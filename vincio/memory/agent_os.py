"""Agent memory OS (1.10): self-editing memory as permissioned tools + a pager.

A MemGPT/Letta-class self-editing memory, but built on Vincio's *audited* write
pipeline rather than beside it. The agent edits its own memory through four
first-class tools — ``memory_append`` / ``memory_replace`` / ``memory_search`` /
``memory_archive`` — each of which rides the existing permission, RBAC, and
audit-chain path (every write is policy-checked and recorded). A
context-pressure pager keeps a small **core memory** block in context and pages
the overflow to the **archival** store (status ``archived``, excluded from
normal recall), promoting archived items back when they become relevant.

Nothing here bypasses governance: ``memory_append`` is ``MemoryEngine.remember``
(privacy / stability / contradiction / confidence checks), ``memory_replace`` is
the audited ``edit``, and ``memory_archive`` is a lifecycle transition the engine
already understands.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..core.tokens import count_tokens
from ..core.types import MemoryScope

if TYPE_CHECKING:
    from .engine import MemoryEngine

__all__ = ["MemoryOS", "memory_tools"]

_OWNER_KEY = {
    MemoryScope.USER: "user_id",
    MemoryScope.AGENT: "agent_id",
    MemoryScope.SESSION: "session_id",
    MemoryScope.TENANT: "tenant_id",
    MemoryScope.ORGANIZATION: "tenant_id",
}


class MemoryOS:
    """Self-editing core/archival memory over the audited write pipeline."""

    def __init__(
        self,
        memory: MemoryEngine,
        *,
        scope: MemoryScope | str = MemoryScope.AGENT,
        owner_id: str = "agent",
        max_core_tokens: int = 2000,
    ) -> None:
        self.memory = memory
        self.scope = MemoryScope(scope)
        self.owner_id = owner_id
        self.max_core_tokens = max_core_tokens
        self.core_ids: list[str] = []

    def _owner_kwargs(self) -> dict[str, Any]:
        key = _OWNER_KEY.get(self.scope)
        return {key: self.owner_id} if key else {}

    # -- the four self-editing operations (exposed as tools) -----------------

    def append(self, content: str, *, type: str | None = None, importance: float = 0.8) -> str:
        """Write a new memory through the guarded pipeline; add it to core."""
        item = self.memory.remember(
            content, scope=self.scope, type=type, confidence=importance, **self._owner_kwargs()
        )
        self.core_ids.append(item.id)
        self.page()
        return item.id

    def replace(self, memory_id: str, content: str) -> bool:
        """Replace a memory's content (re-passes the write policy, audited)."""
        self.memory.edit(memory_id, content=content)
        return True

    def search(self, query: str, *, top_k: int = 5) -> list[str]:
        """Recall active memories (core + non-archived) relevant to *query*."""
        results = self.memory.search(query, top_k=top_k, **self._owner_kwargs())
        return [r.item.content for r in results]

    def archive(self, memory_id: str) -> bool:
        """Page a memory out of core into the archival store (status archived)."""
        item = self.memory.store.get(memory_id)
        if item is None:
            return False
        item.status = "archived"
        self.memory.store.put(item)
        if memory_id in self.core_ids:
            self.core_ids.remove(memory_id)
        self.memory._record(
            "memory_archive", resource=memory_id, details={"scope": item.scope.value}
        )
        return True

    # -- context-pressure pager ---------------------------------------------

    def core_items(self) -> list[Any]:
        items = [self.memory.store.get(mid) for mid in self.core_ids]
        return [i for i in items if i is not None and i.status not in ("archived", "deleted")]

    def core_context(self) -> str:
        """Render the current core-memory block for inclusion in a prompt."""
        return "\n".join(f"- {i.content}" for i in self.core_items())

    def core_tokens(self) -> int:
        return count_tokens(self.core_context())

    def page(self) -> int:
        """Evict the least-important core memories to archival until under budget.

        Returns the number of items paged out. Importance is the memory's
        confidence (heavily-used, confirmed memories survive longer)."""
        paged = 0
        while self.core_tokens() > self.max_core_tokens and len(self.core_ids) > 1:
            items = self.core_items()
            if not items:
                break
            victim = min(items, key=lambda i: i.confidence)
            if not self.archive(victim.id):
                break
            paged += 1
        return paged

    def page_in(self, query: str, *, top_k: int = 3) -> int:
        """Promote relevant archived memories back into core (status active)."""
        promoted = 0
        archived = [
            i
            for i in self.memory.store.all_items(scope=self.scope, statuses=("archived",))
            if i.owner_id == self.owner_id
        ]
        if not archived:
            return 0
        from ..context.scoring import lexical_similarity

        ranked = sorted(archived, key=lambda i: lexical_similarity(query, i.content), reverse=True)
        for item in ranked[:top_k]:
            item.status = "active"
            self.memory.store.put(item)
            if item.id not in self.core_ids:
                self.core_ids.append(item.id)
            promoted += 1
        self.page()
        return promoted

    def tools(self) -> list[Callable[..., Any]]:
        return memory_tools(self)


def memory_tools(os: MemoryOS) -> list[Callable[..., Any]]:
    """The four memory operations as named callables for ``app.add_tool``.

    Register the writes with a memory-write permission and read with read-only,
    so self-editing memory rides the same RBAC + audit path as any other tool::

        os = app.enable_memory_os(owner_id="agent-1")
        # memory_append / memory_replace / memory_archive are registered as writes
    """

    def memory_append(content: str, importance: float = 0.8) -> str:
        """Append a new long-term memory (returns its id)."""
        return os.append(content, importance=importance)

    def memory_replace(memory_id: str, content: str) -> bool:
        """Replace the content of an existing memory by id."""
        return os.replace(memory_id, content)

    def memory_search(query: str, top_k: int = 5) -> list[str]:
        """Search active memories relevant to a query."""
        return os.search(query, top_k=top_k)

    def memory_archive(memory_id: str) -> bool:
        """Archive a memory, paging it out of the in-context core."""
        return os.archive(memory_id)

    return [memory_append, memory_replace, memory_search, memory_archive]
