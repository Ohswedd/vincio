"""Grounded, verified, reversible computer-use action plane."""

import warnings

import pytest

from vincio import (
    ActionPolicy,
    ComputerEnvironment,
    ContextApp,
    MockScreen,
    ScreenApp,
    UIAction,
    UIElement,
    VincioConfig,
    make_web_checkout,
)
from vincio.core.errors import ComputerUseError
from vincio.evals.environment import StateCheck
from vincio.providers import MockProvider
from vincio.tools.computer_environment import (
    AccessibilityScreen,
    PlaywrightScreen,
    RemoteDesktopScreen,
    ScreenState,
)
from vincio.tools.computer_environment import (
    ActionPolicy as _ActionPolicy,
)

warnings.simplefilter("ignore")

_ADDRESS = "role=textbox[name='Address']"
_CHECKOUT = "role=button[name='Checkout']"
_PLACE = "role=button[name='Place order']"
_DELETE = "role=button[name='Delete account']"


def _app(tmp_path):
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    return ContextApp(name="cu", provider=MockProvider(), model="mock-1", config=config)


def _fill_then_order_policy():
    """A policy that addresses the cart, checks out, and places the order."""

    def policy(state: ScreenState):
        s = state.state
        if s["screen"] == "cart" and not s["fields"].get(_ADDRESS):
            return UIAction(kind="type", selector=_ADDRESS, text="1 Main St")
        if s["screen"] == "cart":
            return UIAction(kind="click", selector=_CHECKOUT, expect_change=True)
        if s["screen"] == "review" and not s["flags"].get("order_placed"):
            return UIAction(kind="click", selector=_PLACE)
        return None

    return policy


def _approve_place(action, decision):
    return "Place order" in action.selector


# --------------------------------------------------------------------------- #
# perception & grounding
# --------------------------------------------------------------------------- #


class TestPerception:
    async def test_mock_screen_observes_addressable_elements(self):
        spec, _ = make_web_checkout()
        screen = MockScreen(spec)
        state = await screen.observe()
        assert state.url == "https://shop.test/cart"
        assert state.element(_CHECKOUT).role == "button"
        # grounding by role + name, not coordinate
        assert state.find(role="button", name="Checkout").selector == _CHECKOUT
        assert state.find(role="textbox", name="Address").selector == _ADDRESS
        assert state.find(role="button", name="nonexistent") is None

    async def test_typed_value_reflects_into_element(self):
        spec, _ = make_web_checkout()
        screen = MockScreen(spec)
        await screen.perform(UIAction(kind="type", selector=_ADDRESS, text="42 Oak"))
        state = await screen.observe()
        assert state.element(_ADDRESS).value == "42 Oak"
        assert state.state["fields"][_ADDRESS] == "42 Oak"

    def test_digest_is_state_sensitive(self):
        a = ScreenState(state={"flags": {"x": 1}})
        b = ScreenState(state={"flags": {"x": 2}})
        assert a.digest != b.digest and len(a.digest) == 16

    def test_action_target_is_human_readable(self):
        assert UIAction(kind="navigate", url="https://a").target == "https://a"
        assert UIAction(kind="click", selector="role=button").target == "role=button"
        assert UIAction(kind="key", key="Enter").target == "Enter"

    async def test_unknown_selector_click_raises(self):
        spec, _ = make_web_checkout()
        screen = MockScreen(spec)
        with pytest.raises(ComputerUseError):
            await screen.perform(UIAction(kind="click", selector="role=button[name='Ghost']"))

    async def test_navigate_to_unknown_url_raises(self):
        spec, _ = make_web_checkout()
        screen = MockScreen(spec)
        with pytest.raises(ComputerUseError):
            await screen.perform(UIAction(kind="navigate", url="https://nope.test"))

    def test_bad_start_screen_raises(self):
        with pytest.raises(ComputerUseError):
            MockScreen(ScreenApp(start="missing", screens={}))


# --------------------------------------------------------------------------- #
# pre-gate: scope + destructive + approval
# --------------------------------------------------------------------------- #


class TestPreGate:
    def test_explicit_and_keyword_destructive(self):
        policy = ActionPolicy()
        el = UIElement(selector="x", destructive=True)
        assert policy.is_destructive(UIAction(kind="click", selector="x"), el)
        assert policy.is_destructive(UIAction(kind="click", selector="role=button[name='Delete']"), None)
        assert not policy.is_destructive(UIAction(kind="click", selector="role=button[name='Next']"), None)

    def test_scope_allow_and_deny(self):
        policy = ActionPolicy(allow_urls=["https://ok"], deny_urls=["https://ok/admin"])
        nav = UIAction(kind="navigate", url="https://ok/page")
        assert policy.in_scope(nav, ScreenState())
        assert not policy.in_scope(UIAction(kind="navigate", url="https://evil"), ScreenState())
        assert not policy.in_scope(UIAction(kind="navigate", url="https://ok/admin"), ScreenState())

    def test_decide_blocks_out_of_scope(self):
        d = ActionPolicy(allow_urls=["https://ok"]).decide(
            UIAction(kind="navigate", url="https://evil"), ScreenState(), None, approved=False
        )
        assert not d.allowed and not d.in_scope

    def test_decide_destructive_requires_approval(self):
        policy = ActionPolicy()
        act = UIAction(kind="click", selector=_DELETE, destructive=True)
        blocked = policy.decide(act, ScreenState(), None, approved=False)
        assert not blocked.allowed and blocked.requires_approval and blocked.destructive
        ok = policy.decide(act, ScreenState(), None, approved=True)
        assert ok.allowed and ok.approved

    def test_decide_can_disable_destructive_entirely(self):
        d = ActionPolicy(allow_destructive=False).decide(
            UIAction(kind="click", selector="role=button[name='Delete']"), ScreenState(), None, approved=True
        )
        assert not d.allowed and "disabled" in d.reason

    def test_non_destructive_in_scope_is_allowed(self):
        d = ActionPolicy().decide(UIAction(kind="click", selector="role=button[name='Next']"), ScreenState(), None, approved=False)
        assert d.allowed and not d.requires_approval


# --------------------------------------------------------------------------- #
# act: perform, post-verify, undo
# --------------------------------------------------------------------------- #


class TestActLoop:
    async def test_happy_action_performs_and_verifies(self):
        spec, _ = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec))
        out = await env.act(UIAction(kind="type", selector=_ADDRESS, text="1 Main St"))
        assert out.ok and out.performed and out.verified and not out.diverged

    async def test_expect_change_satisfied(self):
        spec, _ = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec))
        out = await env.act(UIAction(kind="click", selector=_CHECKOUT, expect_change=True))
        assert out.ok and out.verified and out.after_digest != out.before_digest

    async def test_post_condition_check_passes(self):
        spec, _ = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec))
        out = await env.act(
            UIAction(kind="type", selector=_ADDRESS, text="x",
                     expect=[StateCheck(name="set", path=f"fields.{_ADDRESS}", op="eq", value="x")])
        )
        assert out.verified and all(c.passed for c in out.checks)

    async def test_divergence_triggers_auto_undo(self):
        spec, _ = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec), auto_undo=True)
        before = (await env.observe()).digest
        out = await env.act(
            UIAction(kind="type", selector=_ADDRESS, text="x",
                     expect=[StateCheck(name="bogus", path="flags.never", op="truthy")])
        )
        assert out.diverged and out.undone and not out.ok
        assert out.after_digest == before  # state restored

    async def test_no_undo_leaves_effect_but_flags_divergence(self):
        spec, _ = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec), auto_undo=False)
        out = await env.act(
            UIAction(kind="type", selector=_ADDRESS, text="x", expect_change=False)  # asserts no change, but it changes
        )
        assert out.diverged and not out.undone

    async def test_navigate_undo_via_synthesized_inverse(self):
        spec, _ = make_web_checkout()
        screen = MockScreen(spec)
        # move to review first so a divergent navigate can be inverted back
        await screen.perform(UIAction(kind="click", selector=_CHECKOUT))
        env = ComputerEnvironment(screen)
        before = (await env.observe()).digest
        out = await env.act(
            UIAction(kind="navigate", url="https://shop.test/cart",
                     expect=[StateCheck(name="bogus", path="flags.never", op="truthy")])
        )
        assert out.diverged and out.undone and out.after_digest == before

    async def test_gate_blocks_unapproved_destructive(self):
        spec, _ = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec), policy=ActionPolicy(allow_urls=["https://shop.test"]))
        out = await env.act(UIAction(kind="click", selector=_DELETE))
        assert out.gated and not out.performed and not out.ok

    async def test_gate_allows_approved_destructive(self):
        spec, _ = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec))
        out = await env.act(UIAction(kind="click", selector=_DELETE), approve=True)
        assert out.performed and out.decision.approved

    async def test_approval_callback_consulted(self):
        spec, _ = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec), approve=_approve_place)
        denied = await env.act(UIAction(kind="click", selector=_DELETE))
        assert denied.gated  # callback denies 'Delete account'

    async def test_gate_blocks_out_of_scope_navigation(self):
        spec, _ = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec), policy=ActionPolicy(allow_urls=["https://shop.test"]))
        out = await env.act(UIAction(kind="navigate", url="https://evil.test/x"))
        assert out.gated and not out.performed

    async def test_step_budget_exhausts(self):
        spec, _ = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec), max_steps=1)
        await env.act(UIAction(kind="wait"))
        with pytest.raises(ComputerUseError):
            await env.act(UIAction(kind="wait"))

    async def test_driver_failure_is_nonfatal(self):
        spec, _ = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec))
        out = await env.act(UIAction(kind="click", selector="role=button[name='Ghost']"))
        assert not out.ok and out.error and out.performed is False


# --------------------------------------------------------------------------- #
# task grounding & trajectory
# --------------------------------------------------------------------------- #


class TestTaskRun:
    async def test_run_reaches_verified_goal_at_budget(self):
        spec, task = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec), approve=_approve_place)
        run = await env.arun(_fill_then_order_policy(), task)
        assert run.success and run.safe
        assert run.steps_taken <= task.max_steps
        assert run.verification.passed and run.verification.score == 1.0

    async def test_run_projects_onto_trajectory(self):
        spec, task = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec), approve=_approve_place)
        run = await env.arun(_fill_then_order_policy(), task)
        traj = run.trajectory
        assert traj.source == "computer_use" and traj.success
        assert traj.steps[-1].type == "finalize" and traj.steps[-1].status == "done"
        assert any(s.tool_name == "computer_type" for s in traj.steps)

    async def test_run_is_safe_when_destructive_attempted_without_approval(self):
        spec, task = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec), policy=ActionPolicy(allow_urls=["https://shop.test"]))

        # one-shot: attempt the destructive delete once, then finish
        seen = {"n": 0}

        def policy(state):
            seen["n"] += 1
            return UIAction(kind="click", selector=_DELETE) if seen["n"] == 1 else None

        run = await env.arun(policy, task)
        assert run.unapproved_destructive == 0 and run.safe
        assert any(o.gated for o in run.outcomes)

    def test_sync_run_wrapper(self):
        spec, task = make_web_checkout()
        env = ComputerEnvironment(MockScreen(spec), approve=_approve_place)
        run = env.run(_fill_then_order_policy(), task)
        assert run.success

    def test_task_verifier_reports_failed_checks(self):
        _, task = make_web_checkout()
        v = task.verify(ScreenState(state={"fields": {}, "flags": {}}))
        assert not v.passed and "order_placed" in v.reason


# --------------------------------------------------------------------------- #
# app integration
# --------------------------------------------------------------------------- #


class TestAppIntegration:
    def test_computer_use_returns_environment_and_audits(self, tmp_path):
        app = _app(tmp_path)
        spec, _ = make_web_checkout()
        env = app.computer_use(screen=spec)
        assert isinstance(env, ComputerEnvironment) and env.app is app
        assert app.audit.query(action="computer_use_session")
        assert app.audit.verify_chain()

    def test_actions_land_on_audit_chain(self, tmp_path):
        app = _app(tmp_path)
        spec, task = make_web_checkout()
        env = app.computer_use(screen=spec, approve=_approve_place)
        env.run(_fill_then_order_policy(), task)
        actions = app.audit.query(action="computer_action")
        assert len(actions) >= 3 and app.audit.verify_chain()

    def test_require_isolation_refuses_subprocess(self, tmp_path):
        app = _app(tmp_path)
        spec, _ = make_web_checkout()
        from vincio.core.errors import SandboxError

        with pytest.raises(SandboxError):
            app.computer_use(screen=spec, require_isolation=True)

    def test_accepts_mockscreen_screenapp_and_dict(self, tmp_path):
        app = _app(tmp_path)
        spec, _ = make_web_checkout()
        assert isinstance(app.computer_use(screen=MockScreen(spec)).backend, MockScreen)
        assert isinstance(app.computer_use(screen=spec).backend, MockScreen)
        assert isinstance(app.computer_use(screen=spec.model_dump()).backend, MockScreen)

    def test_policy_from_dict(self, tmp_path):
        app = _app(tmp_path)
        spec, _ = make_web_checkout()
        env = app.computer_use(screen=spec, policy={"allow_urls": ["https://shop.test"]})
        assert env.policy.allow_urls == ["https://shop.test"]

    def test_unknown_backend_without_screen_raises(self, tmp_path):
        from vincio.core.errors import ConfigError

        app = _app(tmp_path)
        with pytest.raises(ConfigError):
            app.computer_use(backend="mock")  # no screen supplied

    def test_enable_computer_use_still_works(self, tmp_path):
        # the existing flat tool surface is untouched (backward compatibility)
        app = _app(tmp_path)
        app.enable_computer_use("mock")
        assert "computer_navigate" in app.tool_registry


# --------------------------------------------------------------------------- #
# optional-dependency adapters
# --------------------------------------------------------------------------- #


class TestRealAdapters:
    def test_adapters_carry_stable_names(self):
        assert PlaywrightScreen().name == "playwright_screen"
        assert AccessibilityScreen().name == "accessibility_screen"
        assert RemoteDesktopScreen().name == "remote_desktop_screen"

    async def test_accessibility_requires_extra(self):
        with pytest.raises(ComputerUseError):
            await AccessibilityScreen(driver="definitely-not-installed-xyz").observe()

    async def test_remote_desktop_requires_extra(self):
        with pytest.raises(ComputerUseError):
            await RemoteDesktopScreen(transport="webrtc").observe()


def test_alias_import_matches():
    # exported from both the package root and the tools subpackage
    assert ActionPolicy is _ActionPolicy


def test_error_in_catalog():
    from vincio.core.error_catalog import ERROR_CATALOG

    assert "COMPUTER_USE_ERROR" in ERROR_CATALOG
