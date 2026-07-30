from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from fastapi.testclient import TestClient

from opensesame.engine import GameConfig
from opensesame.model import MockModel
from opensesame.player import attack_message, defense_prompt
from opensesame.server import Runtime, create_app


class ScriptedRuntime(Runtime):
    async def request_player(
        self,
        slot: int,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        profile = "starter" if slot == 0 else "leaky"
        if payload["type"] == "register_request":
            return {"defense_prompt": defense_prompt(profile, payload["passphrase"])}
        if payload["type"] == "attack_request":
            return {
                "message": attack_message(
                    payload["turn"],
                    profile,
                    payload["transcript"],
                )
            }
        raise AssertionError(payload)


def certification_config() -> GameConfig:
    return GameConfig(
        tokens=["token-a", "token-b"],
        players=[{"name": "Starter"}, {"name": "Leaky"}],
        seed=20260806,
        model_provider="mock",
        model_id="certification",
        max_turns=2,
        player_connect_timeout_seconds=0,
        action_timeout_seconds=1,
        model_timeout_seconds=1,
        model_concurrency=2,
        minimum_episode_seconds=0,
    )


@pytest.mark.asyncio
async def test_full_certification_episode_and_secret_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_path = tmp_path / "results.json"
    replay_path = tmp_path / "replay.json"
    monkeypatch.setenv("COGAME_RESULTS_URI", str(results_path))
    monkeypatch.setenv("COGAME_SAVE_REPLAY_URI", str(replay_path))

    runtime = ScriptedRuntime(certification_config(), model=MockModel())
    await runtime.run_episode()

    assert runtime.results is not None
    assert runtime.results["scores"] == [2, 0]
    assert runtime.results["attack_scores"] == [1, 0]
    assert runtime.results["defense_scores"] == [1, 0]
    assert runtime.done
    duel_events = [event for event in runtime.replay_events if event["type"] == "duel_turn"]
    assert [(event["turn"], event["attacker"]) for event in duel_events] == [(1, 0), (1, 1), (2, 1)]

    result_payload = json.loads(results_path.read_text())
    replay_payload = json.loads(replay_path.read_text())
    assert result_payload == runtime.results
    serialized_replay = json.dumps(replay_payload)
    assert all(phrase not in serialized_replay for phrase in runtime.passphrases)
    assert "PROTECTED_PASSPHRASE" not in serialized_replay
    assert "[PASSHRASE REDACTED" in serialized_replay
    assert "tokens" not in replay_payload["config"]


def test_http_and_replay_surface() -> None:
    replay = {"version": 1, "events": [], "results": {"scores": [1, 0]}}
    app = create_app(Runtime(None, replay=replay))
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"ok": True, "mode": "replay"}
        assert "Open Sesame" in client.get("/client/global").text
        assert "Replay" in client.get("/client/replay").text


def test_manifest_configs_match_runtime_and_results_schema() -> None:
    manifest = json.loads(Path("coworld_manifest_template.json").read_text())
    assert "version" not in manifest["game"], "coworld build owns the template version stamp"
    for variant in manifest["variants"]:
        config = {
            **variant["game_config"],
            "tokens": [f"token-{index}" for index in range(len(variant["game_config"]["players"]))],
        }
        GameConfig.model_validate(config)

    fixture = manifest["certification"]["game_config"]
    GameConfig.model_validate(
        {
            **fixture,
            "tokens": [f"token-{index}" for index in range(len(fixture["players"]))],
        }
    )
    jsonschema.Draft202012Validator.check_schema(manifest["game"]["config_schema"])
    jsonschema.Draft202012Validator.check_schema(manifest["game"]["results_schema"])
