import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from constract.compare import (
    ROOT, ComparisonPrompter, INSTRUCTION, TargetGraphEvaluator,
    clean_scene, make_contract, prepared_initial, replay_plan, main,
    run_method, APIConfig, write_json,
)
from high_level_planner.scene_graph.graph import SceneGraph


@pytest.fixture
def scenes():
    return tuple(clean_scene(SceneGraph.from_json(ROOT / "inputs" / name))
                 for name in ("initial_scene_graph.json", "terminal_scene_graph.json"))


@pytest.fixture
def actions():
    return json.loads((Path(__file__).parent / "known_plan.json").read_text(encoding="utf-8"))


def test_shared_evidence_and_no_image_input(scenes):
    initial, terminal = scenes
    contract = make_contract(initial, terminal, 0.04)
    evaluator = TargetGraphEvaluator(initial, terminal, 0.04)
    state = prepared_initial(initial, evaluator)
    prompter = ComparisonPrompter(initial, terminal, contract, 16)
    report = evaluator.evaluate(state, INSTRUCTION)
    direct = prompter.direct_prompt(state, report)
    graph = prompter.generation_prompt(state, INSTRUCTION, report, num_branches=3)
    assert json.loads(direct.rsplit("INPUT:\n", 1)[1]) == json.loads(graph.rsplit("INPUT:\n", 1)[1])
    assert "image_url" not in direct
    assert "consecutive steps" in direct and "alternative next actions" in graph
    assert all(contract["target_identity_and_relation_checks"].values())


def test_known_plan_reaches_same_target_and_cost(scenes, actions):
    initial, terminal = scenes
    before = deepcopy(initial.to_dict())
    result = replay_plan(initial, terminal, actions, 0.04)
    assert result["status"] == "passed", result
    assert result["goal_reached"]
    assert result["total_cost"] == pytest.approx(22.866314494980355)
    assert result["valid_prefix_length"] == 9
    assert initial.to_dict() == before
    assert result["physical_execution_verified"] is False


def test_blocked_book_close_fails_and_stops(scenes, actions):
    result = replay_plan(*scenes, [actions[2], *actions], 0.04)
    assert result["status"] == "invalid_plan"
    assert result["valid_prefix_length"] == 0
    assert result["steps"][1]["status"] == "not_replayed_after_failure"
    assert result["total_cost"] is None


def test_wrong_cup_support_fails(scenes, actions):
    altered = deepcopy(actions)
    altered[6]["target"] = "book_sam3"
    result = replay_plan(*scenes, altered, 0.04)
    assert not result["goal_reached"]
    assert result["status"] == "invalid_plan"


def test_missing_final_action_is_not_success(scenes, actions):
    result = replay_plan(*scenes, actions[:-1], 0.04)
    assert result["status"] == "goal_not_reached"
    assert not result["goal_reached"]


def test_prepare_does_not_require_credentials_or_call_model(tmp_path, monkeypatch):
    monkeypatch.delenv("TABLESIM_REASONING_API_TOKEN", raising=False)
    output = tmp_path / "prepared"
    assert main(["--prepare-only", "--output-dir", str(output)]) == 0
    data = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert data["results"]["direct"]["status"] == "prepared_not_run"
    assert data["results"]["graph"]["api_called"] is False
    assert (output / "index.html").exists()
    assert not list(output.rglob("audit_*.json"))


def test_both_runners_use_same_verifier_with_saved_answers(scenes, actions, tmp_path, monkeypatch):
    # Offline integration only. Synthetic audit records remain inside pytest's temp folder.
    class SavedAnswers:
        def __init__(self, config):
            self.config = config
            self.index = 0

        def generate(self, prompt, **kwargs):
            assert not kwargs.get("images")
            chosen = actions if "one complete ordered plan" in prompt else [actions[self.index]]
            self.index += 1
            write_json(Path(self.config.audit_dir) / f"audit_{self.index:04d}.json", {
                "complete": True, "http_status": 200, "response_model": "kimi-k3",
                "offline_test_fixture": True,
            })
            return [json.dumps(chosen)]

    monkeypatch.setattr("constract.compare.OpenAICompatibleLanguageModel", SavedAnswers)
    initial, terminal = scenes
    contract = make_contract(initial, terminal, 0.04)
    args = SimpleNamespace(max_actions=16, position_tolerance=0.04, max_states=128, beam_width=3, num_branches=3)
    results = [run_method(method, tmp_path / method, initial, terminal, contract, APIConfig(api_token="test-only"), args)
               for method in ("direct", "graph")]
    assert all(r["goal_reached"] for r in results), results
    assert results[0]["total_cost"] == pytest.approx(results[1]["total_cost"])
    assert results[1]["replay_cost_matches_ucs"]
    assert results[0]["api"]["request_count"] == 1
    assert results[1]["api"]["request_count"] == 9
