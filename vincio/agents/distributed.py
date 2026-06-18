"""Lock-free distributed coordination for durable graphs (agents/distributed).

The :class:`~vincio.agents.graph.StateGraph` checkpoints every super-step on the
shared store, which makes a run *resumable*. This module makes it
*safe to resume from more than one worker at once* — the missing piece for
horizontal scale — without a control plane and without forking the engine.

Two cooperating guards, both riding the same checkpoint records:

* a **TTL "running" lease** per graph thread, so a healthy worker holds the
  thread exclusively and a crashed worker's lease expires and is reclaimed; and
* **optimistic-concurrency (CAS)** on the checkpoint *version*, so even in the
  window where a lease was reclaimed mid-step, the stale worker's next commit
  loses the race and aborts instead of double-executing — the winner's
  checkpoint stands.

:class:`GraphCoordinator` is the protocol. :class:`InMemoryGraphCoordinator`
is deterministic (clock-injectable) for the single-process default and tests;
:class:`RedisGraphCoordinator` makes the lease + version durable across
processes for real multi-worker deployments. :class:`DistributedCheckpointer`
wraps any :class:`~vincio.agents.graph.Checkpointer` store with the
coordinator, so a graph compiled against it runs distributed with no change to
node code — and a run can move between the single-process and distributed
backends without losing its evidence ledger or trace, because the lease/CAS
metadata lives on the same checkpoint records.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from ..core.errors import CheckpointConflictError
from ..core.utils import new_id
from .graph import Checkpoint, Checkpointer

__all__ = [
    "GraphCoordinator",
    "InMemoryGraphCoordinator",
    "RedisGraphCoordinator",
    "DistributedCheckpointer",
]


@runtime_checkable
class GraphCoordinator(Protocol):
    """Lease + version coordination for one graph thread.

    All four operations are atomic with respect to concurrent callers. ``commit``
    couples lease ownership *and* the version CAS into a single check, so a
    worker whose lease was reclaimed cannot commit even if the versions happen
    to line up.
    """

    def acquire(self, thread_id: str, owner: str, *, ttl_s: float, version_hint: int = 0) -> bool:
        """Grant the thread to ``owner`` if free/expired/already theirs."""
        ...

    def renew(self, thread_id: str, owner: str, *, ttl_s: float) -> bool:
        """Extend ``owner``'s lease; ``False`` if they no longer hold it."""
        ...

    def release(self, thread_id: str, owner: str) -> None:
        """Release the lease if ``owner`` holds it (no-op otherwise)."""
        ...

    def version(self, thread_id: str) -> int:
        """Current committed head version (``0`` before the first commit)."""
        ...

    def commit(self, thread_id: str, owner: str, *, expected_version: int) -> int:
        """Atomically check lease + CAS the version, returning the new version.

        Raises :class:`CheckpointConflictError` if ``owner`` does not hold the
        lease or ``expected_version`` is stale.
        """
        ...


class InMemoryGraphCoordinator:
    """Process-local coordinator — deterministic and clock-injectable.

    The default for the single-process path and for offline tests. Lease expiry
    uses ``clock`` (monotonic by default); inject a fake clock to drive
    expiry deterministically.
    """

    def __init__(self, *, clock: Any = None) -> None:
        self._clock = clock or time.monotonic
        # thread_id -> (owner, expires_at_monotonic)
        self._leases: dict[str, tuple[str, float]] = {}
        self._versions: dict[str, int] = {}

    def _holder(self, thread_id: str) -> str | None:
        entry = self._leases.get(thread_id)
        if entry is None:
            return None
        owner, expires_at = entry
        if self._clock() >= expires_at:
            return None
        return owner

    def acquire(self, thread_id: str, owner: str, *, ttl_s: float, version_hint: int = 0) -> bool:
        holder = self._holder(thread_id)
        if holder is not None and holder != owner:
            return False
        self._leases[thread_id] = (owner, self._clock() + ttl_s)
        # Seed the version on first sight so a worker that restarts (or a fresh
        # coordinator) lines up with the head it recovered from the store.
        if thread_id not in self._versions:
            self._versions[thread_id] = version_hint
        return True

    def renew(self, thread_id: str, owner: str, *, ttl_s: float) -> bool:
        if self._holder(thread_id) != owner:
            return False
        self._leases[thread_id] = (owner, self._clock() + ttl_s)
        return True

    def release(self, thread_id: str, owner: str) -> None:
        if self._holder(thread_id) == owner:
            self._leases.pop(thread_id, None)

    def version(self, thread_id: str) -> int:
        return self._versions.get(thread_id, 0)

    def commit(self, thread_id: str, owner: str, *, expected_version: int) -> int:
        if self._holder(thread_id) != owner:
            raise CheckpointConflictError(
                f"worker {owner!r} no longer holds the lease on thread {thread_id!r}",
                thread_id=thread_id,
                expected_version=expected_version,
                actual_version=self._versions.get(thread_id, 0),
            )
        current = self._versions.get(thread_id, 0)
        if current != expected_version:
            raise CheckpointConflictError(
                f"stale checkpoint commit on thread {thread_id!r} "
                f"(expected v{expected_version}, head is v{current})",
                thread_id=thread_id,
                expected_version=expected_version,
                actual_version=current,
            )
        self._versions[thread_id] = current + 1
        return current + 1


# Atomic "check lease owner and CAS the version" as one Redis round-trip.
_COMMIT_LUA = """
local lease = redis.call('GET', KEYS[1])
if lease ~= ARGV[1] then return -1 end
local current = tonumber(redis.call('GET', KEYS[2]) or '0')
if current ~= tonumber(ARGV[2]) then return -2 end
redis.call('SET', KEYS[2], current + 1)
return current + 1
"""

# Release only if we still own the lease.
_RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""

# Renew only if we still own the lease.
_RENEW_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('PEXPIRE', KEYS[1], ARGV[2]) end
return 0
"""


class RedisGraphCoordinator:
    """Durable cross-process coordinator backed by Redis.

    The lease is a ``SET key owner NX PX ttl`` key (so expiry is enforced by
    Redis, surviving worker crashes); the version is a plain counter. Commit,
    renew, and release run as Lua scripts so the lease check and the mutation
    are a single atomic step — the property the CAS guarantee depends on.

    Requires ``pip install "vincio[redis]"``. Used only when you opt into a
    multi-process deployment; the in-memory coordinator is the default.
    """

    def __init__(self, url: str = "redis://localhost:6379/0", *, prefix: str = "vincio:graph:") -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - depends on environment
            from ..core.errors import StorageError

            raise StorageError(
                'Redis coordination requires: pip install "vincio[redis]"'
            ) from exc
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._prefix = prefix

    def _lease_key(self, thread_id: str) -> str:
        return f"{self._prefix}lease:{thread_id}"

    def _version_key(self, thread_id: str) -> str:
        return f"{self._prefix}version:{thread_id}"

    def acquire(self, thread_id: str, owner: str, *, ttl_s: float, version_hint: int = 0) -> bool:
        key = self._lease_key(thread_id)
        ttl_ms = int(ttl_s * 1000)
        if self._redis.set(key, owner, nx=True, px=ttl_ms):
            self._redis.set(self._version_key(thread_id), version_hint, nx=True)
            return True
        # Re-entrant: the same worker may already hold the lease (e.g. resuming).
        if self._redis.get(key) == owner:
            self._redis.pexpire(key, ttl_ms)
            return True
        return False

    def renew(self, thread_id: str, owner: str, *, ttl_s: float) -> bool:
        result = self._redis.eval(
            _RENEW_LUA, 1, self._lease_key(thread_id), owner, int(ttl_s * 1000)
        )
        return bool(result)

    def release(self, thread_id: str, owner: str) -> None:
        self._redis.eval(_RELEASE_LUA, 1, self._lease_key(thread_id), owner)

    def version(self, thread_id: str) -> int:
        raw = self._redis.get(self._version_key(thread_id))
        return int(raw) if raw is not None else 0

    def commit(self, thread_id: str, owner: str, *, expected_version: int) -> int:
        result = int(
            self._redis.eval(
                _COMMIT_LUA,
                2,
                self._lease_key(thread_id),
                self._version_key(thread_id),
                owner,
                expected_version,
            )
        )
        if result == -1:
            raise CheckpointConflictError(
                f"worker {owner!r} no longer holds the lease on thread {thread_id!r}",
                thread_id=thread_id,
                expected_version=expected_version,
                actual_version=self.version(thread_id),
            )
        if result == -2:
            raise CheckpointConflictError(
                f"stale checkpoint commit on thread {thread_id!r}",
                thread_id=thread_id,
                expected_version=expected_version,
                actual_version=self.version(thread_id),
            )
        return result


class DistributedCheckpointer(Checkpointer):
    """A :class:`Checkpointer` that lease-guards and CAS-commits each super-step.

    Drop-in for the durable graph: compile a graph against it and the engine
    acquires the thread lease at run start (raising if another live worker holds
    it — that is the anti-double-execution guard), CAS-commits every checkpoint
    write through the coordinator, and releases on terminal. The checkpoint
    records and history are byte-compatible with the base checkpointer, so a
    thread written by the distributed path resumes on the single-process path
    and vice versa.
    """

    def __init__(
        self,
        store: Any = None,
        *,
        coordinator: GraphCoordinator | None = None,
        owner: str | None = None,
        lease_ttl_s: float = 30.0,
    ) -> None:
        super().__init__(store)
        self.coordinator: GraphCoordinator = coordinator or InMemoryGraphCoordinator()
        self.owner = owner or new_id("worker")
        self.lease_ttl_s = lease_ttl_s
        self._expected: dict[str, int] = {}

    def on_thread_start(self, thread_id: str) -> None:
        latest = self.latest(thread_id)
        hint = latest.version if latest is not None else 0
        granted = self.coordinator.acquire(
            thread_id, self.owner, ttl_s=self.lease_ttl_s, version_hint=hint
        )
        if not granted:
            raise CheckpointConflictError(
                f"thread {thread_id!r} is leased by another worker; refusing to "
                "double-execute (retry after the lease TTL or resume elsewhere)",
                thread_id=thread_id,
            )
        self._expected[thread_id] = self.coordinator.version(thread_id)

    def on_thread_end(self, thread_id: str) -> None:
        self.coordinator.release(thread_id, self.owner)
        self._expected.pop(thread_id, None)

    def renew(self, thread_id: str) -> bool:
        """Extend the lease mid-run for long-lived super-steps."""
        return self.coordinator.renew(thread_id, self.owner, ttl_s=self.lease_ttl_s)

    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        thread_id = checkpoint.thread_id
        # Extend the lease for the next super-step before committing this one;
        # if we have already lost it, commit's lease check fails fast below.
        self.coordinator.renew(thread_id, self.owner, ttl_s=self.lease_ttl_s)
        expected = self._expected.get(thread_id, self.coordinator.version(thread_id))
        new_version = self.coordinator.commit(thread_id, self.owner, expected_version=expected)
        self._expected[thread_id] = new_version
        checkpoint.version = new_version
        checkpoint.lease_owner = self.owner
        return super().save(checkpoint)
