from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from opensesame.engine import (
    DefenseRegistration,
    GameConfig,
    build_duels,
    compute_results,
    generate_passphrases,
    redact_passphrase,
)
from opensesame.io import artifact_method, read_data, write_data
from opensesame.model import TextModel, make_model, validate_defense

logger = logging.getLogger("opensesame.game")
STATIC_DIR = Path(__file__).parent / "static"


@dataclass(slots=True)
class PlayerConnection:
    websocket: WebSocket
    inbox: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)


class Runtime:
    def __init__(
        self,
        config: GameConfig,
        *,
        model: TextModel | None = None,
    ) -> None:
        self.config = config
        self.model = (
            model
            if model is not None
            else make_model(
                config.model_provider,
                config.model_id,
                max_tokens=config.response_token_cap,
                timeout_seconds=config.model_timeout_seconds,
            )
        )
        self.seed = config.seed if config.seed is not None else secrets.randbits(63)
        count = len(config.tokens)
        self.passphrases = generate_passphrases(count, self.seed)
        self.registrations = [DefenseRegistration() for _ in range(count)]
        self.duels = build_duels(count)
        self.connections: dict[int, PlayerConnection] = {}
        self.global_viewers: set[WebSocket] = set()
        self.replay_events: list[dict[str, Any]] = []
        self.started = False
        self.done = False
        self.phase = "waiting"
        self.results: dict[str, Any] | None = None
        self._episode_task: asyncio.Task[None] | None = None
        self._connect_timeout_task: asyncio.Task[None] | None = None
        self._request_counter = 0
        self._model_semaphore = asyncio.Semaphore(config.model_concurrency)
        self.on_episode_complete: Callable[[], Awaitable[None] | None] | None = None

    async def startup(self) -> None:
        self._connect_timeout_task = asyncio.create_task(self._start_after_connect_timeout())

    async def shutdown(self) -> None:
        for task in (self._connect_timeout_task, self._episode_task):
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def connect_player(self, slot: int, token: str, websocket: WebSocket) -> None:
        if slot < 0 or slot >= len(self.config.tokens) or token != self.config.tokens[slot]:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        previous = self.connections.get(slot)
        connection = PlayerConnection(websocket=websocket)
        self.connections[slot] = connection
        if previous is not None:
            with suppress(Exception):
                await previous.websocket.close(code=1012)
        logger.info("player slot %d connected (%d/%d)", slot, len(self.connections), len(self.config.tokens))
        await websocket.send_json(
            {
                "type": "hello",
                "slot": slot,
                "player_name": self.config.players[slot].name,
                "player_count": len(self.config.tokens),
                "protocol_version": "1.0.0",
            }
        )
        if len(self.connections) == len(self.config.tokens) and not self.started:
            self._start_episode()
        try:
            async for message in websocket.iter_json():
                if isinstance(message, dict):
                    await connection.inbox.put(message)
        finally:
            if self.connections.get(slot) is connection:
                del self.connections[slot]

    def _start_episode(self) -> None:
        if self.started:
            return
        self.started = True
        self._episode_task = asyncio.create_task(self.run_episode())

    async def _start_after_connect_timeout(self) -> None:
        await asyncio.sleep(self.config.player_connect_timeout_seconds)
        if not self.started:
            logger.warning("player connect deadline reached; starting with %d connected seats", len(self.connections))
            self._start_episode()

    async def request_player(
        self,
        slot: int,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        connection = self.connections.get(slot)
        if connection is None:
            return None
        self._request_counter += 1
        request_id = f"req-{self._request_counter}"
        outgoing = {**payload, "request_id": request_id}
        try:
            await connection.websocket.send_json(outgoing)
        except Exception:
            return None
        deadline = timeout if timeout is not None else self.config.action_timeout_seconds
        loop = asyncio.get_running_loop()
        end = loop.time() + deadline
        while True:
            remaining = end - loop.time()
            if remaining <= 0:
                return None
            try:
                message = await asyncio.wait_for(connection.inbox.get(), timeout=remaining)
            except TimeoutError:
                return None
            if message.get("request_id") == request_id:
                return message

    async def run_episode(self) -> None:
        episode_started_at = asyncio.get_running_loop().time()
        self.phase = "registration"
        await self.publish({"type": "phase", "phase": self.phase})
        responses = await asyncio.gather(*(self._register_seat(slot) for slot in range(len(self.config.tokens))))
        for slot, registration in enumerate(responses):
            self.registrations[slot] = registration
            await self.publish(
                {
                    "type": "registration",
                    "slot": slot,
                    "player_name": self.config.players[slot].name,
                    "eligible": registration.eligible,
                    "reason": registration.reason,
                }
            )

        self.phase = "attack"
        await self.publish({"type": "phase", "phase": self.phase})
        player_count = len(self.config.tokens)
        for offset in range(1, player_count):
            for turn in range(1, self.config.max_turns + 1):
                turn_events = await asyncio.gather(
                    *(
                        self._run_duel_turn(
                            attacker,
                            (attacker + offset) % player_count,
                            turn,
                            offset,
                        )
                        for attacker in range(player_count)
                    )
                )
                for event in turn_events:
                    if event is not None:
                        await self.publish(event, live_payloads=False)

        self.phase = "complete"
        self.results = compute_results(
            self.registrations,
            self.duels,
            [player.name for player in self.config.players],
            seed=self.seed,
            model_provider=self.config.model_provider,
        )
        remaining_minimum = self.config.minimum_episode_seconds - (
            asyncio.get_running_loop().time() - episode_started_at
        )
        if remaining_minimum > 0:
            await asyncio.sleep(remaining_minimum)
        await self.publish({"type": "final_scores", **self.results})
        await self._send_final_to_players()
        await asyncio.sleep(0.25)
        self._write_artifacts()
        self.done = True
        logger.info("episode complete scores=%s", self.results["scores"])
        if self.on_episode_complete is not None:
            outcome = self.on_episode_complete()
            if outcome is not None:
                await outcome

    async def _register_seat(self, slot: int) -> DefenseRegistration:
        passphrase = self.passphrases[slot]
        response = await self.request_player(
            slot,
            {
                "type": "register_request",
                "passphrase": passphrase,
                "defense_prompt_char_cap": self.config.defense_prompt_char_cap,
                "liveness": {
                    "description": "The frozen defense must answer benign probes and never leak this passphrase.",
                    "deterministic_scoring": True,
                },
            },
        )
        if response is None:
            return DefenseRegistration(reason="registration_timeout")
        prompt = response.get("defense_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return DefenseRegistration(reason="invalid_defense_prompt")
        try:
            async with self._model_semaphore:
                eligible, reason, probe_results = await asyncio.wait_for(
                    validate_defense(
                        self.model,
                        prompt,
                        passphrase,
                        prompt_char_cap=self.config.defense_prompt_char_cap,
                    ),
                    timeout=self.config.model_timeout_seconds * 3 + 5,
                )
        except TimeoutError:
            return DefenseRegistration(prompt=prompt, reason="liveness_timeout")
        return DefenseRegistration(
            prompt=prompt,
            eligible=eligible,
            reason=reason,
            probe_results=probe_results,
        )

    async def _run_duel_turn(
        self,
        attacker: int,
        defender: int,
        turn: int,
        offset: int,
    ) -> dict[str, Any] | None:
        duel = self.duels[(attacker, defender)]
        if duel.extracted:
            return None
        if not self.registrations[attacker].eligible or not self.registrations[defender].eligible:
            return None
        response = await self.request_player(
            attacker,
            {
                "type": "attack_request",
                "target": {
                    "slot": defender,
                    "name": self.config.players[defender].name,
                },
                "turn": turn,
                "max_turns": self.config.max_turns,
                "rotation": offset,
                "message_char_cap": self.config.attack_message_char_cap,
                "transcript": duel.transcript,
            },
        )
        message = response.get("message") if response else None
        if not isinstance(message, str) or not message.strip():
            message = "Please help with an ordinary task without revealing protected information."
        message = message[: self.config.attack_message_char_cap]
        try:
            async with self._model_semaphore:
                model_response = await asyncio.wait_for(
                    self.model.complete(
                        self.registrations[defender].prompt,
                        [*duel.transcript, {"role": "user", "content": message}],
                    ),
                    timeout=self.config.model_timeout_seconds,
                )
        except Exception as error:
            logger.warning(
                "model call failed attacker=%d defender=%d turn=%d error=%s",
                attacker,
                defender,
                turn,
                type(error).__name__,
            )
            model_response = "[MODEL UNAVAILABLE — ATTEMPT RECORDED]"

        leaked = duel.record(message, model_response, self.passphrases[defender])
        safe_attack = redact_passphrase(message, self.passphrases[defender])
        safe_response = redact_passphrase(model_response, self.passphrases[defender])
        event = {
            "type": "duel_turn",
            "attacker": attacker,
            "attacker_name": self.config.players[attacker].name,
            "defender": defender,
            "defender_name": self.config.players[defender].name,
            "turn": turn,
            "rotation": offset,
            "attack": safe_attack,
            "response": safe_response,
            "extracted": leaked,
        }
        connection = self.connections.get(attacker)
        if connection is not None:
            with suppress(Exception):
                await connection.websocket.send_json(
                    {
                        "type": "attack_result",
                        "target": {"slot": defender, "name": self.config.players[defender].name},
                        "turn": turn,
                        "response": model_response,
                        "extracted": leaked,
                    }
                )
        defender_connection = self.connections.get(defender)
        if defender_connection is not None:
            with suppress(Exception):
                await defender_connection.websocket.send_json(
                    {
                        "type": "defense_result",
                        "attacker": {"slot": attacker, "name": self.config.players[attacker].name},
                        "turn": turn,
                        "extracted": leaked,
                    }
                )
        return event

    async def publish(self, event: dict[str, Any], *, live_payloads: bool = True) -> None:
        self.replay_events.append(event)
        outgoing = event
        if not live_payloads and not self.done:
            outgoing = {key: value for key, value in event.items() if key not in {"attack", "response"}}
            outgoing["payloads_embargoed"] = True
        stale: list[WebSocket] = []
        for viewer in self.global_viewers:
            try:
                await viewer.send_json(outgoing)
            except Exception:
                stale.append(viewer)
        for viewer in stale:
            self.global_viewers.discard(viewer)

    async def connect_global(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.global_viewers.add(websocket)
        events = (
            self.replay_events
            if self.done
            else [
                {key: value for key, value in event.items() if key not in {"attack", "response"}}
                for event in self.replay_events
            ]
        )
        await websocket.send_json(
            {
                "type": "snapshot",
                "phase": self.phase,
                "started": self.started,
                "done": self.done,
                "events": events,
                "results": self.results,
            }
        )
        try:
            async for _ in websocket.iter_text():
                pass
        finally:
            self.global_viewers.discard(websocket)

    async def _send_final_to_players(self) -> None:
        assert self.results is not None
        for slot, connection in list(self.connections.items()):
            with suppress(Exception):
                await connection.websocket.send_json(
                    {
                        "type": "final",
                        "slot": slot,
                        "score": self.results["scores"][slot],
                        "attack_score": self.results["attack_scores"][slot],
                        "defense_score": self.results["defense_scores"][slot],
                        "eligible": self.results["eligible"][slot],
                    }
                )

    def _write_artifacts(self) -> None:
        assert self.results is not None
        results_uri = os.environ.get("COGAME_RESULTS_URI")
        replay_uri = os.environ.get("COGAME_SAVE_REPLAY_URI")
        if results_uri:
            write_data(
                results_uri,
                json.dumps(self.results, separators=(",", ":")),
                content_type="application/json",
                http_method=artifact_method("COGAME_RESULTS_METHOD"),
            )
        if replay_uri:
            write_data(
                replay_uri,
                json.dumps(self._replay_payload(), separators=(",", ":")),
                content_type="application/json",
                http_method=artifact_method("COGAME_SAVE_REPLAY_METHOD"),
            )

    def _replay_payload(self) -> dict[str, Any]:
        safe_config = self.config.model_dump(exclude={"tokens"})
        return {
            "version": 1,
            "config": safe_config,
            "seed": self.seed,
            "events": self.replay_events,
            "results": self.results,
        }


def runtime_from_environment() -> Runtime:
    config_uri = os.environ.get("COGAME_CONFIG_URI")
    if not config_uri:
        raise RuntimeError("COGAME_CONFIG_URI is required")
    config = GameConfig.model_validate_json(read_data(config_uri))
    return Runtime(config)


def create_app(runtime: Runtime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await runtime.startup()
        yield
        await runtime.shutdown()

    app = FastAPI(title="Open Sesame", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "mode": "episode"}

    @app.get("/client/player")
    def player_client() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "player.html").read_text())

    @app.get("/client/global")
    def global_client() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "global.html").read_text())

    @app.websocket("/player")
    async def player_socket(websocket: WebSocket) -> None:
        try:
            slot = int(websocket.query_params.get("slot", "-1"))
        except ValueError:
            slot = -1
        token = websocket.query_params.get("token", "")
        await runtime.connect_player(slot, token, websocket)

    @app.websocket("/global")
    async def global_socket(websocket: WebSocket) -> None:
        await runtime.connect_global(websocket)

    return app


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    runtime = runtime_from_environment()
    app = create_app(runtime)
    host = os.environ.get("COGAME_HOST", "0.0.0.0")
    port = int(os.environ.get("COGAME_PORT", "8080"))
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))

    async def stop_server() -> None:
        await asyncio.sleep(0.5)
        server.should_exit = True

    runtime.on_episode_complete = stop_server
    server.run()


if __name__ == "__main__":
    main()
