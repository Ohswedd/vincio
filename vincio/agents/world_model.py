"""World-model / simulation-based planning (agents/world_model).

The stateful :class:`~vincio.evals.environment.Environment` contract and the
test-time-search :class:`~vincio.optimize.test_time.Verifier` already let an agent
*evaluate* a trajectory against the live world. The rung this module adds is
letting an agent **learn a model of its tools and environment and plan against
it** — searching in an *imagined* rollout before committing an action to the real
world, so a wrong move costs a simulated step, not a live one.

Three pieces, all deterministic and offline (no model required):

- :class:`WorldModel` — a learned dynamics model fit from recorded
  reset/step :class:`Transition`\\ s. It learns, per tool, the *parameterized*
  state effect (which paths change, and whether the new value is a constant, an
  argument, or a numeric step) under a *learned precondition* (the discriminative
  state field that decides which effect fires). So from a refund that succeeded on
  a *cancelled* order and failed on a *processing* one, it predicts a refund on a
  *delivered* order will succeed — generalizing over arguments and preconditions,
  not memorizing transitions. :meth:`WorldModel.predict` returns a
  :class:`PredictedStep`; :meth:`WorldModel.imagine` rolls a whole plan forward
  without touching a tool.

- :class:`ModelPredictivePlanner` — a receding-horizon (MPC) planner that searches
  imagined rollouts under the world model with the test-time-search **beam**,
  commits the best first action to the *real* environment, observes, and re-plans.
  Scoring prefers the shortest, cheapest plan that reaches the goal (cost-aware
  action selection), and the search is bounded by the same kind of budget the
  orchestrator enforces.

- :class:`CalibrationReport` — the world model earns planning weight only after its
  predicted next states and rewards track the real environment within a tolerance
  (:meth:`WorldModel.calibrate`), the way a judge ensemble earns gating weight. An
  uncalibrated model is refused by the planner by default, so a plan is never
  searched against a world the model does not actually predict.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from ..evals.environment import (
    EnvAction,
    Environment,
    EnvObservation,
    StateCheck,
    TaskVerification,
)
from ..providers.base import run_sync

__all__ = [
    "Transition",
    "PredictedStep",
    "WorldModel",
    "CalibrationReport",
    "MPCStep",
    "MPCResult",
    "ModelPredictivePlanner",
    "record_transitions",
    "task_goal_value",
]


# ---------------------------------------------------------------------------
# Recorded experience
# ---------------------------------------------------------------------------


class Transition(BaseModel):
    """One recorded ``(observation, action) → next_observation`` step.

    The training datum the :class:`WorldModel` is fit from. ``observation`` and
    ``next_observation`` are deep snapshots (the recorder copies them) so a
    transition is independent of the live, mutable world it came from.
    """

    observation: EnvObservation
    action: EnvAction
    next_observation: EnvObservation
    reward: float = 0.0
    ok: bool = True
    done: bool = False


def record_transitions(
    env: Environment,
    action_sequences: Iterable[Sequence[EnvAction]],
    *,
    include_failures: bool = True,
) -> list[Transition]:
    """Drive ``env`` through each action sequence, recording every tool step.

    The exploration data a :class:`WorldModel` learns from. Each sequence runs
    from a fresh :meth:`~vincio.evals.environment.Environment.reset`; only tool
    actions are recorded (message/finish actions do not mutate the world). When
    ``include_failures`` is false, steps the environment rejected (``ok`` false)
    are dropped — keep them (the default) so the model learns *pre*conditions, not
    just effects.
    """
    out: list[Transition] = []
    for sequence in action_sequences:
        env.reset()
        obs = env.observe()
        for action in sequence:
            if action.kind != "tool" or not action.tool:
                continue
            before = obs.model_copy(deep=True)
            result = env.step(action)
            after = result.observation.model_copy(deep=True)
            if include_failures or result.ok:
                out.append(
                    Transition(
                        observation=before,
                        action=action,
                        next_observation=after,
                        reward=result.reward,
                        ok=result.ok,
                        done=result.done,
                    )
                )
            obs = after
            if result.done:
                break
    return out


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class PredictedStep(BaseModel):
    """The world model's prediction for one ``(observation, action)``.

    ``observation`` is the predicted next observation; ``reward`` / ``ok`` /
    ``done`` are the predicted consequence. ``confidence`` in ``[0, 1]`` is how
    much to trust the prediction (``1`` for an exact learned precondition match,
    lower for a fallback, ``0`` for an action the model has never seen, in which
    case the prediction is the identity — no change). ``known`` is whether the
    model had ever seen the action's tool/argument signature.
    """

    observation: EnvObservation
    reward: float = 0.0
    ok: bool = True
    done: bool = False
    confidence: float = 0.0
    known: bool = False
    reason: str = ""


# A learned value template: how the new value at a changed path is produced.
#   ("const", v)      — a literal constant
#   ("arg", name)     — the value of action argument ``name``
#   ("add", k)        — the prior numeric value at the path plus ``k``
_ValueTpl = tuple[str, Any]
# A learned effect outcome: the set of path → value-template changes it applies,
# the paths it removes, and the predicted ok/reward/done.
_Outcome = tuple  # (changes: tuple[(path_tpl, value_tpl)], removed: tuple[path_tpl], ok)


_PLACEHOLDER = "{%s}"


def _flatten(state: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict/list state to ``{dotted_path: scalar_value}`` leaves."""
    out: dict[str, Any] = {}
    if isinstance(state, dict):
        for key, value in state.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    elif isinstance(state, (list, tuple)):
        for index, value in enumerate(state):
            out.update(_flatten(value, f"{prefix}{index}."))
    else:
        out[prefix[:-1]] = state  # strip trailing '.'
    return out


def _parameterize_path(path: str, args: dict[str, Any]) -> str:
    """Replace any path segment equal to an argument value with ``{argname}``."""
    segments = path.split(".")
    str_args = {str(v): k for k, v in args.items() if isinstance(v, (str, int, float))}
    return ".".join(_PLACEHOLDER % str_args[s] if s in str_args else s for s in segments)


def _instantiate_path(template: str, args: dict[str, Any]) -> str:
    """Substitute ``{argname}`` segments in a path template with argument values."""
    segments = template.split(".")
    out: list[str] = []
    for seg in segments:
        if seg.startswith("{") and seg.endswith("}"):
            out.append(str(args.get(seg[1:-1], seg)))
        else:
            out.append(seg)
    return ".".join(out)


def _get_path(state: dict[str, Any], path: str) -> tuple[bool, Any]:
    node: Any = state
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, (list, tuple)):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return False, None
        else:
            return False, None
    return True, node


def _set_path(state: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node: Any = state
    for part in parts[:-1]:
        nxt = node.get(part) if isinstance(node, dict) else None
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _del_path(state: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    node: Any = state
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(parts[-1], None)


class _Effect:
    """A learned, parameterized effect for one ``(tool, argument-names)`` signature.

    Holds a small table mapping the value of a learned *discriminative* state
    field (the precondition) to the outcome it produces, plus a most-common
    fallback. ``key_fields`` is empty for an unconditional effect.
    """

    __slots__ = ("tool", "key_fields", "table", "default", "rewards", "support")

    def __init__(self) -> None:
        self.tool: str = ""
        self.key_fields: tuple[str, ...] = ()
        self.table: dict[tuple[tuple[str, Any], ...], _Outcome] = {}
        self.default: _Outcome | None = None
        self.rewards: dict[_Outcome, float] = {}
        self.support: int = 0


class WorldModel:
    """A deterministic, offline-learned dynamics model of a tool environment.

    Fit from recorded :class:`Transition`\\ s with :meth:`fit` (or the
    constructor's ``transitions=``). For each tool it learns a *parameterized*
    effect — which state paths change, and whether each new value is a constant,
    an argument, or a numeric step — gated by the *discriminative* state field
    that decides which effect fires (the learned precondition). :meth:`predict`
    returns the model's :class:`PredictedStep` for one action; :meth:`imagine`
    rolls a plan forward without touching the live environment.

    The model never invents tools: an action whose signature it has not seen is
    predicted as the identity (no change) with zero confidence, so a planner can
    tell a known move from a guess.
    """

    def __init__(self, transitions: Iterable[Transition] | None = None) -> None:
        self._effects: dict[tuple[str, tuple[str, ...]], _Effect] = {}
        self._vocabulary: list[EnvAction] = []
        self.calibration: CalibrationReport | None = None
        if transitions is not None:
            self.fit(transitions)

    # -- fitting --------------------------------------------------------------

    def fit(self, transitions: Iterable[Transition]) -> WorldModel:
        """Learn per-tool parameterized effects from recorded transitions."""
        groups: dict[tuple[str, tuple[str, ...]], list[Transition]] = {}
        seen: dict[str, EnvAction] = {}
        for tr in transitions:
            sig = (tr.action.tool, tuple(sorted(tr.action.arguments)))
            groups.setdefault(sig, []).append(tr)
            vocab_key = f"{tr.action.tool}|{sorted(tr.action.arguments.items())}"
            seen.setdefault(vocab_key, tr.action.model_copy(deep=True))
        self._effects = {sig: self._learn_effect(sig[0], trs) for sig, trs in groups.items()}
        self._vocabulary = list(seen.values())
        self.calibration = None
        return self

    def _learn_effect(self, tool: str, transitions: list[Transition]) -> _Effect:
        # Per transition: parameterized raw delta + flattened, parameterized pre-state.
        raw: list[dict[str, Any]] = []
        path_samples: dict[str, list[tuple[dict[str, Any], Any, Any]]] = {}
        for tr in transitions:
            args = dict(tr.action.arguments)
            before = _flatten(tr.observation.state)
            after = _flatten(tr.next_observation.state)
            changes: dict[str, tuple[Any, Any]] = {}
            for path, new_value in after.items():
                old_value = before.get(path, None)
                if path not in before or old_value != new_value:
                    tpl = _parameterize_path(path, args)
                    changes[tpl] = (old_value, new_value)
                    path_samples.setdefault(tpl, []).append((args, old_value, new_value))
            removed = tuple(
                sorted(_parameterize_path(p, args) for p in before if p not in after)
            )
            pre_feats = {_parameterize_path(p, args): v for p, v in before.items()}
            raw.append(
                {
                    "args": args,
                    "changes": changes,
                    "removed": removed,
                    "ok": tr.ok,
                    "done": tr.done,
                    "reward": tr.reward,
                    "pre": pre_feats,
                }
            )

        # Resolve a value template per changed path across the whole signature.
        value_tpl: dict[str, _ValueTpl] = {
            path: self._resolve_value_template(samples)
            for path, samples in path_samples.items()
        }

        effect = _Effect()
        effect.tool = tool
        effect.support = len(transitions)

        def outcome_of(entry: dict[str, Any]) -> _Outcome:
            changes = tuple(
                sorted((path, value_tpl[path]) for path in entry["changes"])
            )
            return (changes, entry["removed"], entry["ok"])

        outcomes = [outcome_of(entry) for entry in raw]
        # Mean reward + done per outcome (deterministic over the recorded data).
        reward_acc: dict[_Outcome, list[float]] = {}
        for entry, outcome in zip(raw, outcomes, strict=True):
            reward_acc.setdefault(outcome, []).append(entry["reward"])
        effect.rewards = {o: round(sum(v) / len(v), 6) for o, v in reward_acc.items()}

        distinct = list(dict.fromkeys(outcomes))
        common_default = Counter(outcomes).most_common(1)[0][0]
        effect.default = common_default

        if len(distinct) <= 1:
            effect.key_fields = ()
            effect.table = {(): distinct[0]} if distinct else {}
            return effect

        # Outcomes diverge → learn the discriminative precondition field(s).
        key_fields = self._select_discriminative_fields(raw, outcomes)
        effect.key_fields = key_fields
        table: dict[tuple[tuple[str, Any], ...], _Outcome] = {}
        for entry, outcome in zip(raw, outcomes, strict=True):
            key = tuple((f, entry["pre"].get(f)) for f in key_fields)
            table[key] = outcome  # last write wins; consistent by construction
        effect.table = table
        return effect

    @staticmethod
    def _resolve_value_template(
        samples: list[tuple[dict[str, Any], Any, Any]],
    ) -> _ValueTpl:
        """Decide whether a path's new value is an argument, a step, or a constant."""
        # An argument: every new value equals the same argument's value.
        arg_names = set(samples[0][0])
        for name in sorted(arg_names):
            if all(args.get(name) == new for args, _old, new in samples):
                return ("arg", name)
        # A constant: every new value is identical.
        news = [new for _a, _o, new in samples]
        if all(n == news[0] for n in news):
            return ("const", news[0])
        # A numeric step: new = old + k for a fixed k.
        if all(isinstance(o, (int, float)) and isinstance(n, (int, float)) for _a, o, n in samples):
            steps = {n - o for _a, o, n in samples}
            if len(steps) == 1:
                return ("add", next(iter(steps)))
        return ("const", news[-1])

    @staticmethod
    def _select_discriminative_fields(
        raw: list[dict[str, Any]], outcomes: list[_Outcome]
    ) -> tuple[str, ...]:
        """Pick the state field(s) whose value determines which outcome fires.

        Prefers a single field that (a) maps each value consistently to one
        outcome, (b) has a value that repeats (so it generalizes, unlike a unique
        id), and (c) takes more than one value. Falls back to the combined set of
        fields present in every transition when no single field separates them.
        """
        common = set.intersection(*[set(entry["pre"]) for entry in raw]) if raw else set()
        candidates: list[tuple[int, int, str]] = []
        for field in sorted(common):
            mapping: dict[Any, _Outcome] = {}
            counts: Counter[Any] = Counter()
            consistent = True
            for entry, outcome in zip(raw, outcomes, strict=True):
                value = entry["pre"].get(field)
                counts[value] += 1
                if value in mapping and mapping[value] != outcome:
                    consistent = False
                    break
                mapping[value] = outcome
            repeats = any(c >= 2 for c in counts.values())
            if consistent and repeats and len(counts) > 1:
                # Rank: fewer distinct values (more general), more repeats first.
                candidates.append((len(counts), -max(counts.values()), field))
        if candidates:
            candidates.sort()
            return (candidates[0][2],)
        # No single separating field — use the conjunction of the *informative*
        # common fields (drop constants, which carry no signal).
        non_constant = [
            field
            for field in sorted(common)
            if len({entry["pre"].get(field) for entry in raw}) > 1
        ]
        return tuple(non_constant) if non_constant else tuple(sorted(common))

    # -- prediction -----------------------------------------------------------

    def predict(self, observation: EnvObservation, action: EnvAction) -> PredictedStep:
        """Predict the next observation, reward, and ``ok`` for one action."""
        if action.kind != "tool" or not action.tool:
            return PredictedStep(
                observation=observation.model_copy(deep=True),
                done=action.kind == "finish",
                confidence=1.0,
                known=True,
                reason=f"{action.kind} action does not mutate the world",
            )
        sig = (action.tool, tuple(sorted(action.arguments)))
        effect = self._effects.get(sig)
        if effect is None:
            return PredictedStep(
                observation=observation.model_copy(deep=True),
                ok=False,
                confidence=0.0,
                known=False,
                reason=f"unseen action signature {sig}",
            )
        args = dict(action.arguments)
        key = tuple((f, _read_feature(observation.state, f, args)) for f in effect.key_fields)
        outcome = effect.table.get(key)
        if outcome is not None:
            confidence = 1.0
            reason = "exact precondition match" if effect.key_fields else "unconditional effect"
        else:
            outcome = effect.default
            confidence = 0.5
            reason = "precondition unseen; fell back to most-common effect"
        if outcome is None:  # pragma: no cover - default is set whenever support > 0
            return PredictedStep(
                observation=observation.model_copy(deep=True), known=True, reason="no effect"
            )
        changes, removed, ok = outcome
        next_state = copy.deepcopy(observation.state)
        for path_tpl, value_tpl in changes:
            path = _instantiate_path(path_tpl, args)
            _set_path(next_state, path, _apply_value(value_tpl, next_state, path, args))
        for path_tpl in removed:
            _del_path(next_state, _instantiate_path(path_tpl, args))
        next_obs = observation.model_copy(deep=True)
        next_obs.state = next_state
        next_obs.step = observation.step + 1
        return PredictedStep(
            observation=next_obs,
            reward=effect.rewards.get(outcome, 1.0 if ok else 0.0),
            ok=ok,
            confidence=confidence,
            known=True,
            reason=reason,
        )

    def imagine(
        self, observation: EnvObservation, actions: Sequence[EnvAction]
    ) -> list[PredictedStep]:
        """Roll a whole plan forward under the model, threading predicted states."""
        steps: list[PredictedStep] = []
        obs = observation.model_copy(deep=True)
        for action in actions:
            step = self.predict(obs, action)
            steps.append(step)
            obs = step.observation
            if step.done:
                break
        return steps

    def vocabulary(self) -> list[EnvAction]:
        """The distinct actions seen during :meth:`fit` (the planner's repertoire)."""
        return [a.model_copy(deep=True) for a in self._vocabulary]

    @property
    def trusted(self) -> bool:
        """Whether the last :meth:`calibrate` earned the model planning weight."""
        return self.calibration is not None and self.calibration.trusted

    # -- calibration ----------------------------------------------------------

    def calibrate(
        self,
        transitions: Iterable[Transition],
        *,
        reward_tolerance: float = 0.1,
        min_state_accuracy: float = 0.9,
    ) -> CalibrationReport:
        """Score the model against held-out transitions; earn it planning weight.

        Compares the predicted next state (exact match), reward (absolute error),
        and ``ok`` flag against each recorded transition. The model is *trusted*
        only when its next-state accuracy clears ``min_state_accuracy`` and its
        mean reward error stays within ``reward_tolerance`` — the gate the planner
        checks before searching against it.
        """
        held = list(transitions)
        if not held:
            report = CalibrationReport(
                n=0,
                trusted=False,
                reward_tolerance=reward_tolerance,
                min_state_accuracy=min_state_accuracy,
                reason="no transitions to calibrate against",
            )
            self.calibration = report
            return report
        state_hits = 0
        ok_hits = 0
        reward_err = 0.0
        for tr in held:
            pred = self.predict(tr.observation, tr.action)
            if pred.observation.state == tr.next_observation.state:
                state_hits += 1
            if pred.ok == tr.ok:
                ok_hits += 1
            reward_err += abs(pred.reward - tr.reward)
        n = len(held)
        state_accuracy = round(state_hits / n, 6)
        ok_accuracy = round(ok_hits / n, 6)
        reward_mae = round(reward_err / n, 6)
        trusted = state_accuracy >= min_state_accuracy and reward_mae <= reward_tolerance
        weight = round(state_accuracy, 6) if trusted else 0.0
        reason = (
            f"state accuracy {state_accuracy:.3f} ≥ {min_state_accuracy:.3f} and "
            f"reward MAE {reward_mae:.3f} ≤ {reward_tolerance:.3f}"
            if trusted
            else (
                f"state accuracy {state_accuracy:.3f} (need ≥ {min_state_accuracy:.3f}); "
                f"reward MAE {reward_mae:.3f} (need ≤ {reward_tolerance:.3f})"
            )
        )
        report = CalibrationReport(
            n=n,
            state_accuracy=state_accuracy,
            reward_mae=reward_mae,
            ok_accuracy=ok_accuracy,
            reward_tolerance=reward_tolerance,
            min_state_accuracy=min_state_accuracy,
            trusted=trusted,
            weight=weight,
            reason=reason,
        )
        self.calibration = report
        return report


def _read_feature(state: dict[str, Any], field_template: str, args: dict[str, Any]) -> Any:
    found, value = _get_path(state, _instantiate_path(field_template, args))
    return value if found else None


def _apply_value(value_tpl: _ValueTpl, state: dict[str, Any], path: str, args: dict[str, Any]) -> Any:
    kind, payload = value_tpl
    if kind == "arg":
        return args.get(payload)
    if kind == "add":
        found, old = _get_path(state, path)
        base = old if (found and isinstance(old, (int, float))) else 0
        return base + payload
    return payload  # const


class CalibrationReport(BaseModel):
    """The verdict of :meth:`WorldModel.calibrate`, the model's planning weight.

    ``state_accuracy`` is the fraction of held-out transitions whose next state
    the model predicted exactly; ``reward_mae`` is the mean absolute reward error;
    ``ok_accuracy`` the fraction of correct success/failure predictions.
    ``trusted`` is the gate the planner checks; ``weight`` in ``[0, 1]`` is the
    planning weight earned (``0`` when untrusted)."""

    n: int = 0
    state_accuracy: float = 0.0
    reward_mae: float = 0.0
    ok_accuracy: float = 0.0
    reward_tolerance: float = 0.1
    min_state_accuracy: float = 0.9
    trusted: bool = False
    weight: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Model-predictive (simulation-based) planning
# ---------------------------------------------------------------------------


def task_goal_value(checks: Sequence[StateCheck]) -> Callable[[EnvObservation], float]:
    """A goal-value function: the fraction of an environment task's checks an
    observation's state satisfies (the planner's default verifier)."""
    checks = list(checks)

    def value(observation: EnvObservation) -> float:
        if not checks:
            return 0.0
        passed = sum(1 for c in checks if c.evaluate(observation.state).passed)
        return passed / len(checks)

    return value


class MPCStep(BaseModel):
    """The record of one real, committed step of a model-predictive plan."""

    real_step: int
    action: EnvAction
    imagined_plan: list[EnvAction] = Field(default_factory=list)
    imagined_value: float = 0.0
    imagined_reward: float = 0.0
    model_confidence: float = 0.0
    real_reward: float = 0.0
    ok: bool = True
    reason: str = ""


class MPCResult(BaseModel):
    """The outcome of driving a :class:`ModelPredictivePlanner` to a verified end."""

    task_id: str = ""
    success: bool = False
    real_steps: int = 0
    committed: list[EnvAction] = Field(default_factory=list)
    steps: list[MPCStep] = Field(default_factory=list)
    verification: TaskVerification
    final_value: float = 0.0
    calibrated: bool = False
    planning_weight: float = 0.0
    reason: str = ""


class ModelPredictivePlanner:
    """Plan by searching imagined rollouts under a :class:`WorldModel` (MPC).

    At each real step the planner expands a beam of imagined action sequences
    under the world model (reusing the test-time-search beam), scores each
    imagined state with ``goal_value``, commits only the **first** action of the
    best plan to the real environment, observes the true result, and re-plans —
    so model error is corrected every step and a wrong move costs a simulated
    step, not a live one. The beam score prefers the shortest, cheapest plan that
    reaches the goal (cost-aware action selection via ``action_cost`` /
    ``length_penalty``); set ``reward_weight`` above zero to additionally maximize
    the model's predicted cumulative reward (a return-seeking objective).

    ``actions`` is the candidate repertoire — an explicit list, a
    ``proposer(observation) -> list[EnvAction]`` callable, or ``None`` to use the
    model's learned :meth:`WorldModel.vocabulary`. ``goal_value`` scores an
    observation toward the goal in ``[0, 1]``; when ``None`` the planner derives
    it from the environment task's checks. By default the planner refuses an
    uncalibrated model (``require_calibrated``), the way a judge ensemble must earn
    its gating weight before it counts.
    """

    def __init__(
        self,
        model: WorldModel,
        *,
        actions: list[EnvAction] | Callable[[EnvObservation], list[EnvAction]] | None = None,
        goal_value: Callable[[EnvObservation], float] | None = None,
        horizon: int = 4,
        beam_width: int = 4,
        max_real_steps: int = 12,
        goal_bar: float = 1.0,
        length_penalty: float = 0.01,
        reward_weight: float = 0.0,
        action_cost: Callable[[EnvAction], float] | None = None,
        cost_weight: float = 0.0,
        require_calibrated: bool = True,
    ) -> None:
        self.model = model
        self._actions = actions
        self.goal_value = goal_value
        self.horizon = max(1, horizon)
        self.beam_width = max(1, beam_width)
        self.max_real_steps = max(1, max_real_steps)
        self.goal_bar = goal_bar
        self.length_penalty = length_penalty
        self.reward_weight = reward_weight
        self.action_cost = action_cost
        self.cost_weight = cost_weight
        self.require_calibrated = require_calibrated

    def _propose(self, observation: EnvObservation) -> list[EnvAction]:
        if callable(self._actions):
            return list(self._actions(observation))
        if self._actions is not None:
            return list(self._actions)
        return self.model.vocabulary()

    def _value_fn(self, env: Environment) -> Callable[[EnvObservation], float]:
        if self.goal_value is not None:
            return self.goal_value
        checks = getattr(getattr(env, "task", None), "checks", [])
        return task_goal_value(checks)

    async def aplan(self, env: Environment) -> MPCResult:
        """Drive the planner against the real environment to a verified end state."""
        if self.require_calibrated and not self.model.trusted:
            from ..core.errors import AgentEngineError

            raise AgentEngineError(
                "world model is not calibrated for planning; call WorldModel.calibrate(...) "
                "until it is trusted, or pass require_calibrated=False to plan on an "
                "uncalibrated model"
            )
        value_fn = self._value_fn(env)
        obs = env.reset()
        steps: list[MPCStep] = []
        committed: list[EnvAction] = []
        for real_step in range(self.max_real_steps):
            if value_fn(obs) >= self.goal_bar:
                break
            plan, imagined_value = await self._search(obs, value_fn)
            if not plan:
                break
            first = plan[0]
            prediction = self.model.predict(obs, first)
            result = env.step(first)
            committed.append(first)
            steps.append(
                MPCStep(
                    real_step=real_step,
                    action=first,
                    imagined_plan=[a.model_copy(deep=True) for a in plan],
                    imagined_value=round(imagined_value, 6),
                    imagined_reward=round(prediction.reward, 6),
                    model_confidence=round(prediction.confidence, 6),
                    real_reward=round(result.reward, 6),
                    ok=result.ok,
                    reason=prediction.reason,
                )
            )
            obs = result.observation
            if result.done:
                break
        verification = env.verify()
        final_value = value_fn(env.observe())
        return MPCResult(
            task_id=getattr(getattr(env, "task", None), "id", ""),
            success=verification.passed,
            real_steps=len(committed),
            committed=committed,
            steps=steps,
            verification=verification,
            final_value=round(final_value, 6),
            calibrated=self.model.trusted,
            planning_weight=(self.model.calibration.weight if self.model.calibration else 0.0),
            reason=verification.reason,
        )

    def plan(self, env: Environment) -> MPCResult:
        """Synchronous :meth:`aplan`."""
        return run_sync(self.aplan(env))

    async def _search(
        self, observation: EnvObservation, value_fn: Callable[[EnvObservation], float]
    ) -> tuple[list[EnvAction], float]:
        """Beam-search imagined rollouts; return the best plan and its score."""
        from ..optimize.test_time import CallableVerifier, SearchBudget, TestTimeSearch

        root = _ImaginedNode(observation=observation, plan=())

        def expand(node: _ImaginedNode) -> list[_ImaginedNode]:
            if len(node.plan) >= self.horizon or value_fn(node.observation) >= self.goal_bar:
                return []
            successors: list[_ImaginedNode] = []
            for action in self._propose(node.observation):
                prediction = self.model.predict(node.observation, action)
                successors.append(
                    _ImaginedNode(
                        observation=prediction.observation,
                        plan=(*node.plan, action),
                        cost=node.cost + self._cost(action),
                        reward=node.reward + prediction.reward,
                    )
                )
            return successors

        def score(candidate: Any) -> float:
            node: _ImaginedNode = candidate.output
            base = value_fn(node.observation)
            return (
                base
                + self.reward_weight * node.reward
                - self.length_penalty * len(node.plan)
                - self.cost_weight * node.cost
            )

        search = TestTimeSearch(
            lambda _i: None,
            verifier=CallableVerifier(score, name="world_model"),
            budget=SearchBudget(max_candidates=self.beam_width * self.horizon * 8),
        )
        result = await search.beam_search(
            root=root,
            expand=expand,
            beam_width=self.beam_width,
            max_depth=self.horizon,
            state_text=lambda node: "→".join(a.tool for a in node.plan),
        )
        if result.best is None:
            return [], 0.0
        best_node: _ImaginedNode = result.best.output
        return list(best_node.plan), result.best.score

    def _cost(self, action: EnvAction) -> float:
        return self.action_cost(action) if self.action_cost is not None else 0.0


class _ImaginedNode:
    """One node in the imagined-rollout beam: a predicted state + the plan to it."""

    __slots__ = ("observation", "plan", "cost", "reward")

    def __init__(
        self,
        *,
        observation: EnvObservation,
        plan: tuple[EnvAction, ...],
        cost: float = 0.0,
        reward: float = 0.0,
    ) -> None:
        self.observation = observation
        self.plan = plan
        self.cost = cost
        self.reward = reward
