from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from opensesame.lifecycle import player_loop
from opensesame.model import MockModel
from opensesame.server import Runtime
from tests.test_episode import certification_config
from tests.test_training import native_endpoint as native_endpoint
from tests.test_training import pins


@pytest.mark.asyncio
@pytest.mark.parametrize("final", [False, True])
async def test_actual_socket_stop_evidence_ack_and_clean_eof(final):
    received = []

    async def engine(socket):
        await socket.send(json.dumps({"type": "stop", "stop_id": "owned-stop"}))
        received.append(json.loads(await socket.recv()))
        await socket.send(json.dumps({"type": "evidence_received", "stop_id": "owned-stop"}))
        if final:
            await socket.send(json.dumps({"type": "final", "score": 3}))

    async with (
        websockets.serve(engine, "127.0.0.1", 0, max_size=16 * 1024 * 1024) as server,
        websockets.connect(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as socket,
    ):
        handled = []

        async def recv():
            raw = await anext(socket.__aiter__(), None)
            return json.loads(raw) if raw is not None else None

        async def send(message):
            await socket.send(json.dumps(message))

        async def handle(message):
            handled.append(message)

        result = await player_loop(recv, send, handle, [None])
    assert received == [{"type": "stopped", "stop_id": "owned-stop"}]
    assert result.kind == ("final" if final else "stopped")
    assert handled == ([{"type": "final", "score": 3}] if final else [])


@pytest.mark.asyncio
async def test_unjoined_writer_preserves_writable_private_spool_and_withholds_final(tmp_path, monkeypatch):
    pins(monkeypatch, tmp_path)
    runtime = Runtime(certification_config(), model=MockModel())
    assert runtime.training is not None

    class UncooperativeFuture(asyncio.Future):
        def cancel(self, msg=None):
            return False

    release = UncooperativeFuture()
    started = asyncio.Event()

    async def writer():
        started.set()
        await release

    task = asyncio.create_task(writer())
    runtime._play_task = task
    await started.wait()
    runtime._stop_deadline = asyncio.get_running_loop().time() + 0.05
    await runtime.shutdown()
    assert not runtime.ownership_settled
    assert not runtime.done
    assert not task.done()
    assert not runtime.training.writer.closed
    assert not runtime.training.path.exists()
    snapshot = json.loads(runtime.training.path.with_suffix(".ownership.private").read_text())
    assert snapshot["ownership_settled"] is False
    runtime.training.begin("retained-late-private-fact", 0, {"type": "attack_request", "transcript": []})
    assert "retained-late-private-fact" in runtime.training.spool.read_text()
    release.set_result(None)
    await asyncio.gather(task, return_exceptions=True)
    assert task.done()
    runtime.training.writer.close()


@pytest.mark.asyncio
async def test_actual_native_full_context_progress_crosses_socket_and_engine_admission(native_endpoint, monkeypatch):
    from opensesame.player import run
    from opensesame.training import DecisionAdmission, Progress

    _, responses = native_endpoint
    text = '{"message":"native accepted action"}'
    body = {
        "model": "actual-model",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 32768, "output_tokens": 2},
        "sampling_evidence": {
            "policy_revision": "fixture-owned-model",
            "tokenizer_revision": "fixture-tokenizer",
            "chat_template": "fixture-template",
            "sampling": "full_softmax_temperature_one",
            "enable_thinking": False,
            "max_new_tokens": 2048,
            "max_sequence_length": 34816,
            "sampling_seed": 0,
            "eos_token_ids": [3],
            "prompt_token_ids": [123] * 32768,
            "completion_token_ids": [2, 3],
            "behavior_log_probs": [-0.5, -0.3],
            "response": text,
            "stop_reason": "eos",
        },
    }
    responses[0] = {"code": 200, "body": json.dumps(body)}
    monkeypatch.setenv("OPEN_SESAME_PROFILE", "native")
    monkeypatch.setenv("COWORLD_LLM_TEMPERATURE", "1")
    monkeypatch.setenv("COWORLD_LLM_TOP_P", "1")
    observation = {"type": "attack_request", "message_char_cap": 100, "transcript": []}
    admission = DecisionAdmission()
    admission.begin("actual-window", 1, observation)
    sizes = []

    async def engine(socket):
        await socket.send(json.dumps({"type": "hello", "slot": 1, "protocol_version": "2.0.0"}))
        await socket.send(json.dumps({**observation, "request_id": "actual-window"}))
        async for raw in socket:
            sizes.append(len(raw.encode()))
            message = json.loads(raw)
            if message["type"] == "attempt_progress":
                admission.progress(1, Progress.model_validate(message))
            else:
                assert message["type"] == "attack"
                assert admission.consume("actual-window", message, {"message": message["message"]}, fallback=None)
                await socket.send(json.dumps({"type": "stop", "stop_id": "actual-stop"}))
                assert json.loads(await socket.recv()) == {"type": "stopped", "stop_id": "actual-stop"}
                await socket.send(json.dumps({"type": "evidence_received", "stop_id": "actual-stop"}))
                break

    async with websockets.serve(engine, "127.0.0.1", 0, max_size=16 * 1024 * 1024) as server:
        monkeypatch.setenv("COWORLD_PLAYER_WS_URL", f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}")
        await run()
    capture = admission.captures["actual-window"]
    assert capture.executed_action == {"message": "native accepted action"}
    attempt = capture.attempts[capture.selected_attempt_id]
    assert attempt.accepted
    assert attempt.response_complete is True and attempt.response_reader_joined is True
    assert attempt.raw_response == responses[0]["body"]
    assert len(attempt.prompt_token_ids) == 32768
    assert attempt.sampled_token_ids == [2, 3]
    assert max(sizes) > 65536
    assert max(sizes) < 16 * 1024 * 1024


@pytest.mark.asyncio
@pytest.mark.parametrize("violation", ["early", "stale", "late", "unjoined_reader"])
async def test_engine_denies_invalid_stopped_acknowledgement(violation):
    from opensesame.evidence import Attempt
    from opensesame.training import Progress

    runtime = Runtime(certification_config(), model=MockModel())
    incoming = asyncio.Queue()
    connected = asyncio.Event()
    sent = []

    class Socket:
        async def accept(self):
            connected.set()

        async def send_json(self, message):
            sent.append(message)

        async def close(self, code=1000):
            return None

        async def iter_json(self):
            while True:
                yield await incoming.get()

    reader = asyncio.create_task(runtime.connect_player(0, runtime.config.tokens[0], Socket()))
    await connected.wait()
    await asyncio.sleep(0)
    connection = runtime.connections[0]
    connection.stop_deadline = asyncio.get_running_loop().time() + 1
    if violation != "early":
        connection.stop_id = "engine-owned-stop"
    if violation == "late":
        connection.stop_deadline = asyncio.get_running_loop().time() - 1
    if violation == "unjoined_reader":
        attempt = Attempt(policy="native", response_reader_joined=False)
        connection.native_progress[attempt.attempt_id] = Progress(
            type="attempt_progress", request_id="started", attempt=attempt
        )
    await incoming.put({"type": "stopped", "stop_id": "stale" if violation == "stale" else "engine-owned-stop"})
    with pytest.raises(ValueError, match="acknowledgement"):
        await reader
    assert not connection.stopped.is_set()
    assert not any(message["type"] == "evidence_received" for message in sent)
    assert not runtime.done


@pytest.mark.asyncio
async def test_joined_interruption_seals_only_private_truncation(tmp_path, monkeypatch):
    pins(monkeypatch, tmp_path)
    runtime = Runtime(certification_config(), model=MockModel())
    await runtime.shutdown()
    assert runtime.ownership_settled
    assert not runtime.done
    assert runtime.training.writer.closed
    complete = json.loads(runtime.training.path.read_text())
    assert complete["episode"]["status"] == "truncated"
    assert complete["episode"]["outcome"]["interrupted"] is True
    assert complete["episode"]["participant_outcomes"] is None
    assert "scores" not in complete["episode"]["outcome"]
