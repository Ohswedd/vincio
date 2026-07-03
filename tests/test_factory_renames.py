"""The 7.5 factory-prefix normalization: ``make_*``/``create_*`` → ``build_*``.

Every renamed public factory keeps its old name as a
:func:`vincio.stability.deprecated_alias` until 8.0 removes it. These tests
pin the contract: the old name warns and forwards to the new one, the new
name is silent, both names stay exported, and the ``vincio migrate 8.0``
table delivers the mechanical rewrite. Exercising every alias here also keeps
the reachability gate green (an alias with no caller would otherwise trip it).
"""

from __future__ import annotations

import warnings

import pytest

import vincio
from vincio.core.errors import FineTuneError
from vincio.skills.skill import Skill, SkillScript
from vincio.stability import StabilityLevel, VincioDeprecationWarning, stability_of

# old alias name -> (module path, new canonical name)
FACTORY_RENAMES = {
    "make_retail_environment": ("vincio.evals.environment", "build_retail_environment"),
    "make_counter_environment": ("vincio.evals.environment", "build_counter_environment"),
    "make_vault_environment": ("vincio.evals.environment", "build_vault_environment"),
    "make_agent_solver": ("vincio.evals.benchmarks", "build_agent_solver"),
    "make_env_solver": ("vincio.evals.benchmarks", "build_env_solver"),
    "make_web_checkout": ("vincio.tools.computer_environment", "build_web_checkout"),
    "make_finetune_backend": ("vincio.providers.finetune", "build_finetune_backend"),
    "create_metadata_store": ("vincio.storage.base", "build_metadata_store"),
    "make_script_handler": ("vincio.skills.scripts", "build_script_handler"),
    "make_query_contract": ("vincio.data.query", "build_query_contract"),
}


def _symbols(old: str) -> tuple[object, object]:
    import importlib

    module_path, new = FACTORY_RENAMES[old]
    module = importlib.import_module(module_path)
    return getattr(module, old), getattr(module, new)


@pytest.mark.parametrize("old", sorted(FACTORY_RENAMES))
def test_alias_carries_the_deprecation_record(old):
    alias, target = _symbols(old)
    record = stability_of(alias)
    assert record["level"] is StabilityLevel.DEPRECATED
    assert record["since"] == "7.5"
    assert record["removed_in"] == "8.0"
    assert record["alternative"] == FACTORY_RENAMES[old][1]
    assert alias.__wrapped__ is target
    assert alias.__name__ == old
    # the build_* target itself is stable and silent
    assert stability_of(target)["level"] is StabilityLevel.STABLE


@pytest.mark.parametrize("old", sorted(FACTORY_RENAMES))
def test_both_names_stay_exported_from_the_defining_module(old):
    import importlib

    module_path, new = FACTORY_RENAMES[old]
    module = importlib.import_module(module_path)
    assert old in module.__all__
    assert new in module.__all__


def _minimal_skill() -> tuple[Skill, SkillScript]:
    script = SkillScript(name="hello", path="scripts/hello.py")
    return Skill(name="demo", description="demo skill", scripts=[script], path="."), script


def test_each_alias_warns_and_forwards():
    """Every old name emits VincioDeprecationWarning and returns the target's result."""
    from vincio.data.query import make_query_contract
    from vincio.evals.benchmarks import make_agent_solver, make_env_solver
    from vincio.evals.environment import (
        make_counter_environment,
        make_retail_environment,
        make_vault_environment,
    )
    from vincio.providers.finetune import make_finetune_backend
    from vincio.skills.scripts import make_script_handler
    from vincio.storage.base import create_metadata_store
    from vincio.tools.computer_environment import make_web_checkout

    with pytest.warns(VincioDeprecationWarning, match="make_retail_environment"):
        env = make_retail_environment()
    assert env.name == "retail"
    with pytest.warns(VincioDeprecationWarning, match="make_counter_environment"):
        env = make_counter_environment()
    assert env.name == "counter"
    with pytest.warns(VincioDeprecationWarning, match="make_vault_environment"):
        env = make_vault_environment()
    assert env.name == "vault"
    with pytest.warns(VincioDeprecationWarning, match="make_agent_solver"):
        solver = make_agent_solver(lambda prompt: "x")
    assert callable(solver)
    with pytest.warns(VincioDeprecationWarning, match="make_env_solver"):
        solver = make_env_solver(lambda obs: None)
    assert callable(solver)
    with pytest.warns(VincioDeprecationWarning, match="make_web_checkout"):
        app_spec, task = make_web_checkout()
    assert app_spec.name == "shop"
    assert task.id == "place_order"
    with (
        pytest.warns(VincioDeprecationWarning, match="make_finetune_backend"),
        pytest.raises(FineTuneError),
    ):
        make_finetune_backend(object())
    with pytest.warns(VincioDeprecationWarning, match="create_metadata_store"):
        store = create_metadata_store("memory://")
    assert store.count("runs") == 0
    skill, script = _minimal_skill()
    with pytest.warns(VincioDeprecationWarning, match="make_script_handler"):
        handler = make_script_handler(skill, script)
    assert callable(handler)
    with pytest.warns(VincioDeprecationWarning, match="make_query_contract"):
        contract = make_query_contract()
    assert contract is not None


def test_build_names_are_silent():
    """The canonical build_* names never emit a deprecation warning."""
    from types import SimpleNamespace

    from vincio.data.query import build_query_contract
    from vincio.evals.benchmarks import build_agent_solver, build_env_solver
    from vincio.evals.environment import (
        build_counter_environment,
        build_retail_environment,
        build_vault_environment,
    )
    from vincio.providers.finetune import build_finetune_backend
    from vincio.skills.scripts import build_script_handler
    from vincio.storage.base import build_metadata_store
    from vincio.tools.computer_environment import build_web_checkout

    skill, script = _minimal_skill()
    with warnings.catch_warnings():
        warnings.simplefilter("error", VincioDeprecationWarning)
        build_retail_environment()
        build_counter_environment()
        build_vault_environment()
        build_agent_solver(lambda prompt: "x")
        build_env_solver(lambda obs: None)
        build_web_checkout()
        build_finetune_backend(SimpleNamespace(name="openai"))
        build_metadata_store("memory://")
        build_script_handler(skill, script)
        build_query_contract()


def test_alias_result_matches_target_result():
    from vincio.evals.environment import build_counter_environment, make_counter_environment

    fresh = build_counter_environment()
    with pytest.warns(VincioDeprecationWarning):
        aliased = make_counter_environment()
    assert aliased.observe() == fresh.observe()


def test_top_level_reexports_are_the_same_objects():
    from vincio.evals import environment
    from vincio.providers import finetune
    from vincio.tools import computer_environment

    assert vincio.build_retail_environment is environment.build_retail_environment
    assert vincio.build_finetune_backend is finetune.build_finetune_backend
    assert vincio.build_web_checkout is computer_environment.build_web_checkout
    for name in (
        "build_retail_environment",
        "make_retail_environment",
        "build_finetune_backend",
        "make_finetune_backend",
        "build_web_checkout",
        "make_web_checkout",
    ):
        assert name in vincio.__all__


def test_subpackage_reexports_carry_both_names():
    import vincio.data
    import vincio.evals
    import vincio.providers
    import vincio.skills
    import vincio.storage
    import vincio.tools

    expected = {
        vincio.evals: (
            "build_retail_environment",
            "make_retail_environment",
            "build_counter_environment",
            "make_counter_environment",
            "build_vault_environment",
            "make_vault_environment",
            "build_agent_solver",
            "make_agent_solver",
            "build_env_solver",
            "make_env_solver",
        ),
        vincio.tools: ("build_web_checkout", "make_web_checkout"),
        vincio.providers: ("build_finetune_backend", "make_finetune_backend"),
        vincio.storage: ("build_metadata_store", "create_metadata_store"),
        vincio.skills: ("build_script_handler", "make_script_handler"),
        vincio.data: ("build_query_contract", "make_query_contract"),
    }
    for module, names in expected.items():
        for name in names:
            assert name in module.__all__, f"{module.__name__} is missing {name}"
            assert getattr(module, name) is not None
