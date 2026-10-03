"""Actual partial HTTP bytes and owner joins remain private on interruption."""

from __future__ import annotations

import asyncio
import base64

import pytest
from pydantic import ValidationError

from opensesame.evidence import Attempt
from opensesame.native import NativeModel
from tests.test_training import native_endpoint as native_endpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["deadline", "cancel"])
async def test_partial_invalid_utf8_keeps_received_bytes_and_joins_reader(monkeypatch, interrupt):
    prefix = b'{"private":"SENTINEL\xff'
    captured = asyncio.Event()
    disconnected = asyncio.Event()
    progress = []

    async def handler(reader, writer):
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in header.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            await reader.readexactly(length)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 999\r\nContent-Encoding: identity\r\n\r\n" + prefix)
            await writer.drain()
            assert await reader.read() == b""
            disconnected.set()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}")
    model = NativeModel("fixture/native", max_tokens=64, timeout_seconds=0.5, purpose="learner")

    async def publish(attempt):
        progress.append(attempt.model_copy(deep=True))
        if attempt.response_body_b64 == base64.b64encode(prefix).decode():
            captured.set()

    task = asyncio.create_task(
        model.complete("private rules", [{"role": "user", "content": "private view"}], slot=1, on_attempt=publish)
    )
    try:
        await asyncio.wait_for(captured.wait(), 2)
        if interrupt == "cancel":
            task.cancel()
        with pytest.raises(asyncio.CancelledError if interrupt == "cancel" else TimeoutError):
            await asyncio.wait_for(task, 2)
        await asyncio.wait_for(disconnected.wait(), 2)
        final = progress[-1]
        assert final.response_body_b64 == base64.b64encode(prefix).decode()
        assert final.raw_response is None
        assert final.http_status == 200
        assert final.response_complete is False
        assert final.response_reader_joined is True
        assert final.response is None and final.sampled_token_ids is None
        assert progress[0].response_complete is None
        assert progress[0].response_reader_joined is None
        assert progress[0].http_status is None
        assert progress[0].response_body_b64 is None
    finally:
        server.close()
        await server.wait_closed()


def test_private_native_validation_hides_untrusted_input_and_rejects_coercion():
    with pytest.raises(ValidationError) as error:
        Attempt(policy="native-learner", response_complete="PRIVATE_SCHEMA_SENTINEL")
    assert "PRIVATE_SCHEMA_SENTINEL" not in str(error.value)
    with pytest.raises(ValidationError):
        Attempt(policy="native-learner", prompt_token_ids=[True])
    with pytest.raises(ValidationError):
        Attempt(policy="native-learner", http_status="200")


@pytest.mark.asyncio
async def test_cancellation_suppressing_transport_close_is_never_reader_join(native_endpoint, monkeypatch):
    import httpx

    from opensesame.lifecycle import OwnershipUnsettled, owned_children

    class UncooperativeFuture(asyncio.Future):
        def cancel(self, msg=None):
            return False

    release = UncooperativeFuture()
    factory = httpx.AsyncClient
    close_started = asyncio.Event()
    children = set()
    progress = []

    def client_factory(*args, **kwargs):
        client = factory(*args, **kwargs)
        real_close = client.aclose

        async def uncooperative_close():
            close_started.set()
            try:
                await release
            finally:
                await real_close()

        client.aclose = uncooperative_close
        return client

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    token = owned_children.set(children)
    model = NativeModel("learner", max_tokens=128, timeout_seconds=2, purpose="learner")

    async def record(attempt):
        progress.append(attempt.model_copy(deep=True))

    try:
        with pytest.raises(OwnershipUnsettled):
            await model.complete("player", [{"role": "user", "content": "view"}], slot=1, on_attempt=record)
        assert close_started.is_set()
        assert progress[-1].response_complete is True
        assert progress[-1].response_reader_joined is False
        assert progress[-1].raw_response == native_endpoint[1][0]["body"]
        assert any(not task.done() for task in children)
    finally:
        release.set_result(None)
        done, pending = await asyncio.wait(children, timeout=2)
        assert not pending
        for task in done:
            if not task.cancelled():
                task.result()
        owned_children.reset(token)
