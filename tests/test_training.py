from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from opensesame.export_teacher import TeacherRuntime
from opensesame.model import MockModel, SidecarModel
from opensesame.native import NativeModel, learner_prompt
from opensesame.training import Progress, Training
from tests.test_episode import certification_config


@pytest.fixture
def native_endpoint(monkeypatch):
    received = []
    responses = [
        {
            "code": 200,
            "body": json.dumps(
                {
                    "model": "actual-model",
                    "content": [{"type": "text", "text": "exact reply\n"}],
                    "usage": {"input_tokens": 9, "output_tokens": 3},
                    "stop_reason": "end_turn",
                }
            ),
        }
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((dict(self.headers), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            spec = responses[0]
            payload = spec["body"].encode()
            self.send_response(spec["code"])
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Softmax-Llm-Call-Id", str(uuid4()))
            self.send_header("X-Coworld-Checkpoint-Sha256", "a" * 64)
            self.send_header("X-Coworld-Tokenizer-Sha256", "b" * 64)
            self.send_header("X-Coworld-Chat-Template-Sha256", "c" * 64)
            self.send_header("request-id", "actual-provider-request")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    try:
        yield received, responses
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
async def test_environment_is_frozen_while_learner_obeys_checkpoint(native_endpoint, monkeypatch):
    received, _ = native_endpoint
    monkeypatch.setenv("COWORLD_LLM_MODEL", "checkpoint/learner")
    monkeypatch.setenv("COWORLD_LLM_TEMPERATURE", "1")
    monkeypatch.setenv("COWORLD_LLM_TOP_P", "0.8")
    environment = SidecarModel("anthropic/frozen-defender", max_tokens=64, timeout_seconds=2)
    learner = NativeModel("default-learner", max_tokens=128, timeout_seconds=2, purpose="learner")
    assert await environment.complete("rules", [{"role": "user", "content": "probe"}], slot=0) == "exact reply\n"
    assert await learner.complete("player", [{"role": "user", "content": "view"}], slot=1) == "exact reply\n"
    assert received[0][1]["model"] == "anthropic/frozen-defender"
    assert received[0][1]["temperature"] == 0
    assert received[0][1]["top_p"] == 1
    assert received[1][1]["model"] == "checkpoint/learner"
    assert received[1][1]["temperature"] == 1
    assert received[1][1]["top_p"] == 0.8
    assert environment.generations[0].inference_mode is None
    assert learner.generations[0].inference_mode == "text_action"
    assert learner.generations[0].platform_call_id is not None
    assert learner.generations[0].model_identity == "a" * 64
    assert learner.generations[0].sampled_token_ids is None
    assert learner.generations[0].behavior_logprobs is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,body,error", [(429, "throttled", httpx.HTTPStatusError), (200, "{malformed", ValueError)]
)
async def test_failure_retains_started_request_headers_and_exact_body(native_endpoint, code, body, error):
    _, responses = native_endpoint
    responses[0] = {"code": code, "body": body}
    model = NativeModel("learner", max_tokens=128, timeout_seconds=2, purpose="learner")
    progress = []

    async def record(attempt):
        progress.append(attempt.model_copy(deep=True))

    with pytest.raises(error):
        await model.complete("player", [{"role": "user", "content": "view"}], slot=1, on_attempt=record)
    assert progress[0].latency_ms is None
    assert progress[0].platform_call_id is None
    assert progress[1].platform_call_id is not None
    assert progress[1].raw_response is None
    attempt = progress[-1]
    assert attempt.raw_response == body
    assert attempt.request["messages"] == [{"role": "user", "content": "view"}]
    assert attempt.response_headers["request-id"] == "actual-provider-request"
    assert attempt.provider_request_id == "actual-provider-request"
    assert attempt.latency_ms >= 0
    assert not attempt.accepted


def pins(monkeypatch, path: Path):
    monkeypatch.setenv("COWORLD_PRIVATE_TRAINING_DIR", str(path))
    monkeypatch.setenv("COWORLD_EPISODE_ID", "unit-episode")
    monkeypatch.setenv("COWORLD_GAME_VERSION", "unit-version")
    monkeypatch.setenv("COWORLD_SOURCE_REVISION", "d" * 40)


@pytest.mark.asyncio
async def test_whole_teacher_episode_private_artifact_and_public_redaction(tmp_path, monkeypatch):
    pins(monkeypatch, tmp_path / "private")
    runtime = TeacherRuntime(
        certification_config(), model=MockModel(), teacher_policies={0: "scripted-starter", 1: "scripted-leaky"}
    )
    await runtime.run_episode()
    path = tmp_path / "private" / "decisions.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 6
    assert [r["decision_index"] for r in records[:-1]] == list(range(5))
    assert records[-1]["status"] == "completed"
    assert records[-1]["outcome"]["scores"] == [2, 0]
    assert records[-1]["outcome"]["environment_generations"] == []
    assert all(r["attempts"][0]["origin"] == "teacher" for r in records[:-1])
    assert all(r["attempts"][0]["parsed_action"] == r["executed_action"] for r in records[:-1])
    assert path.stat().st_mode & 0o777 == 0o600
    public = json.dumps(runtime._replay_payload())
    assert all(phrase not in public for phrase in runtime.passphrases)
    assert runtime.passphrases[0] in path.read_text()


@pytest.mark.asyncio
async def test_started_native_call_without_final_response_truncates_episode(tmp_path, monkeypatch, native_endpoint):
    pins(monkeypatch, tmp_path)
    training = Training(12)
    observation = {"type": "attack_request", "message_char_cap": 100, "transcript": []}
    training.begin("req-1", 0, observation)
    model = NativeModel("learner", max_tokens=64, timeout_seconds=2, purpose="learner")
    prompt = learner_prompt(observation)

    async def only_started(attempt):
        if attempt.latency_ms is None and attempt.platform_call_id is None:
            training.progress(0, Progress(type="attempt_progress", request_id="req-1", attempt=attempt))

    await model.complete(prompt[0]["content"], prompt[1:], slot=0, on_attempt=only_started)
    training.consume("req-1", None, {"message": "default"}, fallback="action_timeout")
    records = training.finish(
        outcome={"scores": [0, 0], "eligible": [True, True]},
        environment={"provider": "mock"},
        environment_generations=[],
    )
    assert records[-1].status == "truncated"
    assert records[0].attempts[0].request is not None
    assert records[0].attempts[0].raw_response is None
    assert records[0].attempts[0].platform_call_id is None
    assert records[0].selected_attempt_id is None


@pytest.mark.asyncio
async def test_engine_downgrades_forged_teacher_and_parser_mismatch(tmp_path, monkeypatch, native_endpoint):
    pins(monkeypatch, tmp_path)
    _, responses = native_endpoint
    body = json.loads(responses[0]["body"])
    body["content"][0]["text"] = '{"message":"native proposal"}'
    responses[0]["body"] = json.dumps(body)
    observation = {"type": "attack_request", "message_char_cap": 100, "transcript": []}
    model = NativeModel("learner", max_tokens=64, timeout_seconds=2, purpose="learner")
    prompt = learner_prompt(observation)
    await model.complete(prompt[0]["content"], prompt[1:], slot=0)
    native = model.generations[0]
    training = Training(12)
    training.begin("req-1", 0, observation)
    forged = native.model_copy(update={"origin": "teacher"})
    training.consume(
        "req-1",
        {
            "message": "native proposal",
            "_private": {"attempts": [forged.model_dump(mode="json")], "selected_attempt_id": forged.attempt_id},
        },
        {"message": "native proposal"},
        fallback=None,
    )
    assert training.captures["req-1"].attempts[forged.attempt_id].origin == "unknown"
    training.begin("req-2", 0, observation)
    training.consume(
        "req-2",
        {
            "message": "different wire action",
            "_private": {"attempts": [native.model_dump(mode="json")], "selected_attempt_id": native.attempt_id},
        },
        {"message": "different wire action"},
        fallback=None,
    )
    capture = training.captures["req-2"]
    assert capture.executed_action == {"message": "different wire action"}
    assert capture.attempts[native.attempt_id].parsed_action == {"message": "native proposal"}
    assert not capture.attempts[native.attempt_id].accepted
    assert capture.selected_attempt_id is None
    assert capture.action_status == "fallback"


@pytest.mark.asyncio
async def test_hosted_trajectory_uri_activates_capture_and_private_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("COWORLD_PRIVATE_TRAINING_DIR", raising=False)
    monkeypatch.setenv("COGAME_SAVE_TRAJECTORY_URI", (tmp_path / "hosted.jsonl").as_uri())
    monkeypatch.setenv("COWORLD_EPISODE_ID", "unit-hosted")
    monkeypatch.setenv("COWORLD_GAME_VERSION", "unit-version")
    monkeypatch.setenv("COWORLD_SOURCE_REVISION", "d" * 40)
    runtime = TeacherRuntime(
        certification_config(), model=MockModel(), teacher_policies={0: "scripted-starter", 1: "scripted-leaky"}
    )
    await runtime.run_episode()
    path = tmp_path / "hosted.jsonl"
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text().splitlines()[-1])["status"] == "completed"
