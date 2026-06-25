"""Coverage-hardening tests for :mod:`vincio.settlement.exchange`.

These target the uncovered error paths and edge branches of the pull-based
attestation exchange: malformed/mis-typed wire envelopes, the client ``fetch``
poll-and-terminal-state handling, ``_as_exchange`` coercion of every accepted
connection shape, ``_peer_items`` normalization of iterables, the revocation
verify/dedup branches, held-artifact merging, and the ``_task_output`` message
fallback. Everything runs offline against the deterministic MockProvider and
in-process A2A fabric — no mocks, no network.
"""

from __future__ import annotations

import json

import pytest

from vincio import (
    AttestationExchange,
    ContextApp,
    ReputationBundle,
    attestation_a2a_server,
    gather_reputation,
)
from vincio.a2a import connect_a2a_in_process
from vincio.a2a.protocol import A2AArtifact, A2AMessage, A2APart, A2ATask, A2ATaskStatus
from vincio.a2a.server import A2AServer
from vincio.core.errors import SettlementError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.settlement.exchange import (
    _as_exchange,
    _decode_bundle,
    _decode_query,
    _peer_items,
    _task_output,
)


def _contract(buyer: str, seller: str = "vendor", price: float = 0.10) -> Contract:
    return Contract(
        buyer=buyer, seller=seller, terms=ContractTerms(scope="work", price_usd=price)
    ).seal()


def _org(name: str, *, settled: int = 3, subject: str = "vendor") -> ContextApp:
    app = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    app.use_settlement_book()
    for _ in range(settled):
        app.settle(_contract(name, subject), cost_usd=0.05)
    return app


def _buyer(name: str = "buyer") -> ContextApp:
    app = ContextApp(name=name, provider=MockProvider(default_text="ok"))
    app.use_reputation_ledger()
    return app


# -- _decode_query: malformed / mis-typed envelopes ---------------------------


def test_decode_query_rejects_non_json() -> None:
    with pytest.raises(SettlementError, match="malformed attestation-exchange query"):
        _decode_query("this is not json {")


def test_decode_query_rejects_non_envelope_json() -> None:
    # Valid JSON, but lacks the attestation-exchange envelope key.
    with pytest.raises(SettlementError, match="not an attestation-exchange query"):
        _decode_query(json.dumps({"some_other_key": {"subject": "vendor"}}))


def test_decode_query_rejects_json_scalar() -> None:
    # Valid JSON that is not a dict — the `.get` guard refuses it.
    with pytest.raises(SettlementError, match="not an attestation-exchange query"):
        _decode_query(json.dumps("just-a-string"))


def test_decode_query_extracts_the_subject() -> None:
    text = json.dumps({"vincio_attestation_exchange": {"kind": "gather", "subject": "v"}})
    assert _decode_query(text) == "v"


# -- _decode_bundle: malformed / mis-typed replies ----------------------------


def test_decode_bundle_rejects_non_json() -> None:
    with pytest.raises(SettlementError, match="malformed attestation-exchange reply"):
        _decode_bundle("}{not json")


def test_decode_bundle_rejects_non_envelope_reply() -> None:
    with pytest.raises(SettlementError, match="not an attestation-exchange bundle"):
        _decode_bundle(json.dumps({"unrelated": 1}))


def test_decode_bundle_rejects_json_list() -> None:
    # A JSON list is not a dict; the envelope guard refuses it.
    with pytest.raises(SettlementError, match="not an attestation-exchange bundle"):
        _decode_bundle(json.dumps([1, 2, 3]))


# -- AttestationExchange.fetch: poll path and terminal-state error ------------


class _ScriptedClient:
    """A minimal A2A client surface returning pre-scripted tasks per send/poll."""

    def __init__(self, send_task: A2ATask, poll_task: A2ATask | None = None) -> None:
        self._send_task = send_task
        self._poll_task = poll_task
        self.polled: list[str] = []

    async def send(self, _text: str) -> A2ATask:
        return self._send_task

    async def poll_task(self, task_id: str) -> A2ATask:
        self.polled.append(task_id)
        return self._poll_task if self._poll_task is not None else self._send_task


def _completed_task(bundle: ReputationBundle) -> A2ATask:
    output = json.dumps({"vincio_attestation_exchange": bundle.to_wire()})
    return A2ATask(
        status=A2ATaskStatus(state="completed"),
        artifacts=[A2AArtifact(parts=[A2APart(kind="text", text=output)])],
    )


async def test_fetch_polls_a_working_task_until_completed() -> None:
    bundle = ReputationBundle(subject="vendor")
    submitted = A2ATask(id="t-1", status=A2ATaskStatus(state="working"))
    completed = _completed_task(bundle)
    client = _ScriptedClient(submitted, completed)
    exchange = AttestationExchange(client, peer_id="acme")

    got = await exchange.fetch("vendor")

    assert got.subject == "vendor"
    assert client.polled == ["t-1"]  # the working task was polled by id


async def test_fetch_raises_on_a_non_completed_terminal_state() -> None:
    failed = A2ATask(id="t-2", status=A2ATaskStatus(state="failed"))
    exchange = AttestationExchange(_ScriptedClient(failed), peer_id="acme")
    with pytest.raises(SettlementError, match="attestation peer acme ended in failed") as ei:
        await exchange.fetch("vendor")
    assert ei.value.details == {"peer_id": "acme", "state": "failed"}


async def test_fetch_unknown_peer_id_renders_question_mark() -> None:
    failed = A2ATask(id="t-3", status=A2ATaskStatus(state="rejected"))
    exchange = AttestationExchange(_ScriptedClient(failed))  # no peer_id
    with pytest.raises(SettlementError, match=r"attestation peer \? ended in rejected"):
        await exchange.fetch("vendor")


# -- _as_exchange: every accepted connection shape ----------------------------


def test_as_exchange_backfills_peer_id_on_a_bare_exchange() -> None:
    exchange = AttestationExchange(_ScriptedClient(A2ATask()))  # peer_id == ""
    out = _as_exchange(exchange, peer_id="acme")
    assert out is exchange
    assert out.peer_id == "acme"  # backfilled


def test_as_exchange_preserves_an_existing_peer_id() -> None:
    exchange = AttestationExchange(_ScriptedClient(A2ATask()), peer_id="orig")
    out = _as_exchange(exchange, peer_id="other")
    assert out is exchange
    assert out.peer_id == "orig"  # not overwritten


def test_as_exchange_wraps_an_in_process_server() -> None:
    acme = _org("acme", settled=3)
    server = acme.serve_attestations()
    out = _as_exchange(server, peer_id="acme")
    assert isinstance(out, AttestationExchange)
    assert out.peer_id == "acme"


async def test_as_exchange_uses_a_raw_client_directly() -> None:
    # A bare object exposing send / poll_task is accepted as a client.
    bundle = ReputationBundle(subject="vendor")
    client = _ScriptedClient(_completed_task(bundle))
    out = _as_exchange(client, peer_id="acme")
    assert isinstance(out, AttestationExchange)
    assert out.client is client
    assert (await out.fetch("vendor")).subject == "vendor"


def test_as_exchange_rejects_an_unusable_connection() -> None:
    with pytest.raises(SettlementError, match="not an AttestationExchange") as ei:
        _as_exchange(object(), peer_id="bogus")
    assert ei.value.details["peer_id"] == "bogus"
    assert ei.value.details["type"] == "object"


# -- _peer_items: iterable normalization --------------------------------------


def test_peer_items_sorts_a_dict_by_id() -> None:
    assert _peer_items({"z": 1, "a": 2}) == [("a", 2), ("z", 1)]


def test_peer_items_accepts_tuple_pairs() -> None:
    assert _peer_items([("b", 10), ("a", 20)]) == [("a", 20), ("b", 10)]


def test_peer_items_accepts_list_pairs() -> None:
    assert _peer_items([["b", 10], ["a", 20]]) == [("a", 20), ("b", 10)]


def test_peer_items_reads_peer_id_attribute() -> None:
    class _Conn:
        def __init__(self, pid: str) -> None:
            self.peer_id = pid

    a, b = _Conn("zeta"), _Conn("acme")
    items = _peer_items([a, b])
    assert [pid for pid, _ in items] == ["acme", "zeta"]
    assert items[0][1] is b


def test_peer_items_falls_back_to_org_id_attribute() -> None:
    class _OrgConn:
        peer_id = ""  # empty -> falls through to org_id

        def __init__(self, oid: str) -> None:
            self.org_id = oid

    items = _peer_items([_OrgConn("globex")])
    assert items[0][0] == "globex"


def test_peer_items_handles_none() -> None:
    assert _peer_items(None) == []


# -- _task_output: artifact-then-message fallback -----------------------------


def test_task_output_reads_the_first_text_artifact() -> None:
    task = A2ATask(artifacts=[A2AArtifact(parts=[A2APart(kind="text", text="hello")])])
    assert _task_output(task) == "hello"


def test_task_output_falls_back_to_the_status_message() -> None:
    # No artifacts -> the status message text is used.
    task = A2ATask(
        status=A2ATaskStatus(state="completed", message=A2AMessage(parts=[A2APart(text="msg-out")]))
    )
    assert _task_output(task) == "msg-out"


def test_task_output_empty_when_nothing_present() -> None:
    assert _task_output(A2ATask()) == ""


# -- revocation gossip: verify / dedup branches in gather ---------------------


async def test_gossiped_revocation_deduplicated_across_two_peers() -> None:
    acme = _org("acme", settled=4)
    att = acme.attest_reputation("vendor")
    acme.revoke_attestation(att, reason="regressed")
    server = acme.serve_attestations()
    buyer = _buyer()
    # The same server reachable under two ids gossips the same revocation twice.
    result = await buyer.agather_reputation(
        "vendor", peers={"acme": server, "acme-mirror": server}, weight=False
    )
    assert result.revocations_gathered == 1  # deduped by content hash
    mirror = result.visit_for("acme-mirror")
    assert mirror is not None and mirror.duplicates >= 1


async def test_a_forged_revocation_from_a_peer_is_refused() -> None:
    acme = _org("acme", settled=4)
    att = acme.attest_reputation("vendor")
    rev = acme.revoke_attestation(att, reason="regressed")
    rev.signatures[0].signature = "deadbeef"  # forge the issuer signature
    server = acme.serve_attestations(revocations=[rev])
    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor", peers={"acme": server}, verify_with=acme.contract_signer, weight=False
    )
    assert result.revocations_gathered == 0  # the forged revocation is dropped


# -- held-artifact merge dedup branches ---------------------------------------


async def test_held_attestation_already_gathered_is_not_double_counted() -> None:
    acme = _org("acme", settled=3)
    # The importer already holds the very attestation acme will gossip.
    held = acme.attest_reputation("vendor")
    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor",
        peers={"acme": acme.serve_attestations(attestations=[held])},
        held_attestations=[held],
        weight=False,
    )
    assert result.attestations_gathered == 1  # the held copy dedups against the gossiped one


async def test_held_attestation_merges_when_not_already_gathered() -> None:
    acme = _org("acme", settled=3)
    globex = _org("globex", settled=2)
    # The importer already holds globex's attestation out of band; acme gossips its own.
    held = globex.attest_reputation("vendor")
    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor",
        peers={"acme": acme.serve_attestations()},
        held_attestations=[held],
        weight=False,
    )
    assert result.attestations_gathered == 2  # the fresh held copy is folded in
    assert held in result.attestations
    assert result.standing("vendor").issuers == ["acme", "globex"]


async def test_held_revocation_merges_when_not_already_seen() -> None:
    acme = _org("acme", settled=3)
    globex = _org("globex", settled=2)
    # The held revocation is acme's, but only globex (which holds no revocation) is a
    # peer — so the held revocation is genuinely fresh, not a dedup of a gossiped one.
    att = acme.attest_reputation("vendor")
    held_rev = acme.revoke_attestation(att, reason="held out of band")
    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor",
        peers={"globex": globex.serve_attestations()},
        held_revocations=[held_rev],
        weight=False,
    )
    # The fresh held revocation is folded into the gathered set.
    assert result.revocations_gathered == 1
    assert held_rev in result.revocations
    # globex's evidence still stands (the revocation targets acme's absent claim).
    assert result.standing("vendor").issuers == ["globex"]


async def test_held_revocation_already_gossiped_is_not_double_counted() -> None:
    acme = _org("acme", settled=3)
    att = acme.attest_reputation("vendor")
    gossiped_rev = acme.revoke_attestation(att, reason="regressed")
    buyer = _buyer()
    # acme gossips the revocation AND the importer holds the same one -> dedup (false
    # branch of the held-revocation merge).
    result = await buyer.agather_reputation(
        "vendor",
        peers={"acme": acme.serve_attestations()},
        held_revocations=[gossiped_rev],
        weight=False,
    )
    assert result.revocations_gathered == 1  # not double-counted


# -- GatheredReputation.peers_visited -----------------------------------------


async def test_peers_visited_counts_allowed_and_denied_alike() -> None:
    acme = _org("acme", settled=3)
    buyer = _buyer()
    # bogus is considered (a visit) though it never contributes.
    result = await buyer.agather_reputation(
        "vendor", peers={"acme": acme.serve_attestations(), "bogus": object()}, weight=False
    )
    assert result.peers_visited == 2
    assert result.peers_reachable == 1


# -- attestation_a2a_server defaults ------------------------------------------


async def test_server_defaults_owner_when_book_has_no_owner() -> None:
    # A book-like object with a falsy owner falls back to the "attestor" default.
    class _BareBook:
        owner = ""

        def attest(self, _subject: str, *, config=None, sign=True):  # noqa: ANN001
            raise SettlementError("no admissible history")

    server = attestation_a2a_server(_BareBook())
    assert server.card.name == "attestor"
    exchange = AttestationExchange(connect_a2a_in_process(server), peer_id="p")
    bundle = await exchange.fetch("vendor")
    assert bundle.is_empty  # no admissible history -> attestation-free bundle


async def test_module_gather_with_tuple_peers() -> None:
    acme = _org("acme", settled=3)
    # Exercise the iterable-of-pairs branch of _peer_items through the public API.
    result = await gather_reputation("vendor", peers=[("acme", acme.serve_attestations())])
    assert result.attestations_gathered == 1
    assert result.visit_for("acme").attestations == 1


def test_server_is_an_a2a_server() -> None:
    acme = _org("acme", settled=3)
    server = acme.serve_attestations()
    assert isinstance(server, A2AServer)


# -- GatheredReputation reads: contributed / contributing / weight / standing -


async def test_gather_reads_expose_per_peer_contribution_and_standing() -> None:
    acme = _org("acme", settled=3)
    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor",
        peers={"acme": acme.serve_attestations(), "bogus": object()},
        weight=False,
    )
    # contributed property: acme yes, bogus no.
    assert result.visit_for("acme").contributed is True
    assert result.visit_for("bogus").contributed is False
    assert result.peers_contributing == ["acme"]
    # weight / standing delegate to the assembled prior.
    assert 0.0 < result.weight("vendor") <= 1.0
    assert result.standing("vendor").issuers == ["acme"]
    # An unknown member has no standing and floors to the minimum weight.
    assert result.standing("nobody") is None
    assert result.weight("nobody") == result.reputation.weight("nobody")


# -- directory deny + bounded fan-out skip branches (self-contained) ----------


async def test_directory_denied_peer_is_skipped_with_reason() -> None:
    from vincio.a2a import AgentCard

    acme = _org("acme", settled=3)
    evil = _org("evil", settled=9)
    buyer = _buyer()
    directory = buyer.agent_directory(allow=["acme"])
    directory.register(AgentCard(name="acme", description="peer"))
    directory.register(AgentCard(name="evil", description="peer"))
    result = await buyer.agather_reputation(
        "vendor",
        peers={"acme": acme.serve_attestations(), "evil": evil.serve_attestations()},
        directory=directory,
    )
    evil_visit = result.visit_for("evil")
    assert evil_visit is not None
    assert evil_visit.allowed is False
    assert evil_visit.reachable is False
    assert evil_visit.reason  # a denial reason is pinpointed
    assert result.peers_contributing == ["acme"]


async def test_bounded_fan_out_records_the_skip_reason() -> None:
    acme = _org("acme", settled=3)
    globex = _org("globex", settled=2)
    buyer = _buyer()
    result = await buyer.agather_reputation(
        "vendor",
        peers={"acme": acme.serve_attestations(), "globex": globex.serve_attestations()},
        max_peers=1,
        weight=False,
    )
    assert result.peers_reachable == 1
    # globex sorts after acme, so it is the one bounded out.
    globex_visit = result.visit_for("globex")
    assert globex_visit is not None
    assert globex_visit.reachable is False
    assert globex_visit.reason == "fan-out bound reached"


# -- _verifies hash-fail returns False (no verify_with) -----------------------


async def test_a_tampered_attestation_is_refused_without_a_verifier() -> None:
    acme = _org("acme", settled=3)
    tampered = acme.attest_reputation("vendor")
    tampered.settled = 99  # the stored hash no longer recomputes
    server = acme.serve_attestations(attestations=[tampered])
    buyer = _buyer()
    # No verify_with, so the hash-recompute branch alone must refuse it.
    result = await buyer.agather_reputation("vendor", peers={"acme": server}, weight=False)
    assert result.attestations_gathered == 0
    assert result.visit_for("acme").attestations == 0


# -- _task_output skips an empty artifact then falls back ---------------------


def test_task_output_skips_empty_artifact_for_the_message() -> None:
    # First artifact has no text -> loop continues; status message is used.
    task = A2ATask(
        artifacts=[A2AArtifact(parts=[A2APart(kind="data", data={"x": 1})])],
        status=A2ATaskStatus(
            state="completed", message=A2AMessage(parts=[A2APart(text="from-message")])
        ),
    )
    assert _task_output(task) == "from-message"
