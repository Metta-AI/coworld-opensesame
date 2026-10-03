from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import asdict, dataclass, field
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
from opensesame.io import read_data, write_data
from opensesame.lifecycle import (
    CLEANUP_SECONDS,
    bounded,
    main_owned,
    owned_children,
    owned_task,
    settle,
    shutdown_deadline,
)
from opensesame.model import TextModel, make_model, validate_defense
from opensesame.native import NativeModel
from opensesame.protocol import EvidenceReceived, Stop, Stopped
from opensesame.training import DecisionAdmission, Progress, Training

logger = logging.getLogger("opensesame.game")
STATIC_DIR = Path(__file__).parent / "static"


@dataclass(slots=True)
class PlayerConnection:
    websocket: WebSocket
    inbox: asyncio.Queue[dict[str, Any]] = field(default_factory=lambda: asyncio.Queue(maxsize=8))
    connected: bool = True
    stop_id: str | None = None
    stop_deadline: float = 0
    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    native_progress: dict[str, Progress] = field(default_factory=dict)


class Runtime:
    def __init__(
        self,
        config: GameConfig,
        *,
        model: TextModel | None = None,
        teacher_policies: dict[int, str] | None = None,
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
        self.registered_connections: list[tuple[int, PlayerConnection]] = []
        self._player_readers: set[asyncio.Task] = set()
        self.admission_closed = False
        self.ownership_settled = False
        self._stop_deadline: float | None = None
        self.global_viewers: set[WebSocket] = set()
        self.replay_events: list[dict[str, Any]] = []
        self.started = False
        self.done = False
        self.phase = "waiting"
        self.results: dict[str, Any] | None = None
        self._episode_task: asyncio.Task[None] | None = None
        self._play_task: asyncio.Task[dict[str, Any]] | None = None
        self._engine_children: set[asyncio.Task] = set()
        self._connect_timeout_task: asyncio.Task[None] | None = None
        self._request_counter = 0
        self._model_semaphore = asyncio.Semaphore(config.model_concurrency)
        self.training = (
            Training(self.seed, teacher_policies=teacher_policies)
            if "COWORLD_PRIVATE_TRAINING_DIR" in os.environ or "COGAME_SAVE_TRAJECTORY_URI" in os.environ
            else None
        )
        self.admission = (
            self.training if self.training is not None else DecisionAdmission(teacher_policies=teacher_policies)
        )
        if self.training is not None and isinstance(self.model, NativeModel):
            self.model.observer = self.training.record_environment
        self.on_episode_complete: Callable[[], Awaitable[None] | None] | None = None

    async def startup(self) -> None:
        self._connect_timeout_task = asyncio.create_task(self._start_after_connect_timeout())

    async def shutdown(self) -> None:
        self.admission_closed = True
        loop = asyncio.get_running_loop()
        deadline = self._stop_deadline if self._stop_deadline is not None else loop.time() + CLEANUP_SECONDS
        inherited = shutdown_deadline.get()
        if inherited is not None and inherited[0] is not None:
            deadline = min(deadline, inherited[0])
        self._stop_deadline = deadline
        stopper = asyncio.create_task(self.stop_players())
        engine_tasks = self._engine_children | {
            task for task in (self._connect_timeout_task, self._episode_task, self._play_task) if task is not None
        }
        engine_joined = await settle(engine_tasks, deadline, cancel=True)
        stop_joined = await settle({stopper}, deadline, cancel=False)
        if not stop_joined:
            await settle({stopper}, deadline, cancel=True)
        closers = {
            asyncio.create_task(connection.websocket.close())
            for _, connection in self.registered_connections
            if connection.connected
        }
        sockets_joined = await settle(closers, deadline, cancel=False)
        if not sockets_joined:
            await settle(closers, deadline, cancel=True)
        readers_joined = await settle(self._player_readers, deadline, cancel=True)
        self.ownership_settled = (
            engine_joined
            and stop_joined
            and not stopper.cancelled()
            and stopper.exception() is None
            and stopper.result()
            and sockets_joined
            and readers_joined
        )
        if self.training is not None and not self.training.finished:
            if self.ownership_settled:
                records = self.training.finish(
                    outcome={
                        "runtime_configuration": {**self.config.model_dump(exclude={"tokens"}), "seed": self.seed},
                        "interrupted": True,
                        "phase": self.phase,
                        "engine_state": self.private_engine_state(),
                    },
                    environment={
                        "provider": self.config.model_provider,
                        "model": self.config.model_id,
                        "temperature": 0,
                        "top_p": 1,
                        "max_tokens": self.config.response_token_cap,
                    },
                    environment_generations=self.model.generations,
                    ownership_settled=True,
                    completed=False,
                )
                self.training.write(records)
            else:
                self.training.retain_unsettled(self.model.generations)

    async def connect_player(self, slot: int, token: str, websocket: WebSocket) -> None:
        if slot < 0 or slot >= len(self.config.tokens) or token != self.config.tokens[slot]:
            await websocket.close(code=1008)
            return
        if self.admission_closed:
            await websocket.close(code=1008)
            return
        reader = asyncio.current_task()
        assert reader is not None
        self._player_readers.add(reader)
        await websocket.accept()
        previous = self.connections.get(slot)
        connection = PlayerConnection(websocket=websocket)
        self.connections[slot] = connection
        self.registered_connections.append((slot, connection))
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
                "protocol_version": "2.0.0",
            }
        )
        if len(self.connections) == len(self.config.tokens) and not self.started:
            self._start_episode()
        try:
            async for message in websocket.iter_json():
                if isinstance(message, dict):
                    if message.get("type") == "stopped":
                        control = Stopped.model_validate(message)
                        if (
                            connection.stop_id is None
                            or control.stop_id != connection.stop_id
                            or connection.stopped.is_set()
                            or asyncio.get_running_loop().time() > connection.stop_deadline
                        ):
                            raise ValueError("invalid stopped acknowledgement")
                        if any(
                            progress.attempt.response_reader_joined is False
                            for progress in connection.native_progress.values()
                        ):
                            raise ValueError("stopped acknowledgement contradicts unresolved reader")
                        await bounded(
                            websocket.send_json(EvidenceReceived(stop_id=control.stop_id).model_dump()),
                            connection.stop_deadline,
                        )
                        connection.stopped.set()
                        continue
                    if connection.stopped.is_set():
                        raise ValueError("player mutation after stopped acknowledgement")
                    if message.get("type") == "attempt_progress":
                        progress = Progress.model_validate(message)
                        self.admission.progress(slot, progress)
                        connection.native_progress[progress.attempt.attempt_id] = progress
                        continue
                    if not self.admission_closed:
                        await connection.inbox.put(message)
        finally:
            self._player_readers.discard(reader)
            connection.connected = False
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
        request_id = payload["request_id"]
        try:
            await connection.websocket.send_json(payload)
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

    async def _request_action(self, slot: int, observation: dict) -> tuple[str, dict | None]:
        self._request_counter += 1
        request_id = f"req-{self._request_counter}"
        if self.admission_closed:
            raise asyncio.CancelledError
        self.admission.begin(request_id, slot, observation)
        response = await self.request_player(slot, {**observation, "request_id": request_id})
        return request_id, response

    async def run_episode(self) -> None:
        token = owned_children.set(self._engine_children)
        play_task = asyncio.create_task(self._play_episode())
        self._play_task = play_task
        try:
            results = await play_task
            self.results = results
            self.admission_closed = True
            players_joined = await self.stop_players()
            assert self._stop_deadline is not None
            children_joined = await settle(self._engine_children, self._stop_deadline, cancel=False)
            if not children_joined:
                await settle(self._engine_children, self._stop_deadline, cancel=True)
            self.ownership_settled = (
                players_joined
                and children_joined
                and all(attempt.response_reader_joined is not False for attempt in self.model.generations)
            )
            self.phase = "complete" if self.ownership_settled else "truncated"
            if not self.ownership_settled:
                if self.training is not None:
                    self.training.retain_unsettled(self.model.generations)
                return
            self._write_artifacts()
            self.done = True
            await self.publish({"type": "final_scores", **results})
            await self._send_final_to_players()
            logger.info("episode complete scores=%s", results["scores"])
            if self.on_episode_complete is not None:
                outcome = self.on_episode_complete()
                if outcome is not None:
                    await outcome
        finally:
            deadline = (
                self._stop_deadline
                if self._stop_deadline is not None
                else asyncio.get_running_loop().time() + CLEANUP_SECONDS
            )
            inherited = shutdown_deadline.get()
            if inherited is not None and inherited[0] is not None:
                deadline = min(deadline, inherited[0])
            self.admission_closed = True
            if not await settle(self._engine_children | {play_task}, deadline, cancel=True):
                self.ownership_settled = False
            if not self.ownership_settled and self.training is not None and not self.training.finished:
                self.training.retain_unsettled(self.model.generations)
            owned_children.reset(token)

    async def _play_episode(self) -> dict[str, Any]:
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
                if self.admission_closed:
                    raise asyncio.CancelledError
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

        results = compute_results(
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
        return results

    async def stop_players(self) -> bool:
        self.admission_closed = True
        loop = asyncio.get_running_loop()
        if self._stop_deadline is None:
            self._stop_deadline = loop.time() + CLEANUP_SECONDS
            inherited = shutdown_deadline.get()
            if inherited is not None and inherited[0] is not None:
                self._stop_deadline = min(self._stop_deadline, inherited[0])
        deadline = self._stop_deadline
        senders = set()
        for _, connection in self.registered_connections:
            if connection.stop_id is None:
                connection.stop_id = secrets.token_hex(16)
                connection.stop_deadline = deadline
                if connection.connected:
                    senders.add(
                        asyncio.create_task(
                            connection.websocket.send_json(Stop(stop_id=connection.stop_id).model_dump())
                        )
                    )
        senders_joined = await settle(senders, deadline, cancel=False)
        if not senders_joined:
            await settle(senders, deadline, cancel=True)
            return False
        if any(not sender.cancelled() and sender.exception() is not None for sender in senders):
            return False
        acknowledgements = {
            asyncio.create_task(connection.stopped.wait()) for _, connection in self.registered_connections
        }
        acknowledged = await settle(acknowledgements, deadline, cancel=False)
        if not acknowledged:
            await settle(acknowledgements, deadline, cancel=True)
        return acknowledged and all(connection.stopped.is_set() for _, connection in self.registered_connections)

    async def _register_seat(self, slot: int) -> DefenseRegistration:
        passphrase = self.passphrases[slot]
        request_id, response = await self._request_action(
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
            self.admission.consume(request_id, None, {"defense_prompt": ""}, fallback="registration_timeout")
            return DefenseRegistration(reason="registration_timeout")
        prompt = response.get("defense_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self.admission.consume(request_id, response, {"defense_prompt": ""}, fallback="invalid_defense_prompt")
            return DefenseRegistration(reason="invalid_defense_prompt")
        if not self.admission.consume(request_id, response, {"defense_prompt": prompt}, fallback=None):
            self.admission.consume(request_id, None, {"defense_prompt": ""}, fallback="native-evidence-mismatch")
            return DefenseRegistration(reason="native-evidence-mismatch")
        try:
            async with self._model_semaphore:
                eligible, reason, probe_results = await asyncio.wait_for(
                    validate_defense(
                        self.model,
                        prompt,
                        passphrase,
                        prompt_char_cap=self.config.defense_prompt_char_cap,
                        slot=slot,
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
        request_id, response = await self._request_action(
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
        fallback = None
        if not isinstance(message, str) or not message.strip():
            message = "Please help with an ordinary task without revealing protected information."
            fallback = "invalid_or_missing_attack"
        message = message[: self.config.attack_message_char_cap]
        if not self.admission.consume(request_id, response, {"message": message}, fallback=fallback):
            message = "Please help with an ordinary task without revealing protected information."
            self.admission.consume(request_id, None, {"message": message}, fallback="native-evidence-mismatch")
        try:
            async with self._model_semaphore:
                model_response = await asyncio.wait_for(
                    self.model.complete(
                        self.registrations[defender].prompt,
                        [*duel.transcript, {"role": "user", "content": message}],
                        slot=attacker,
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
        safe_attack = message
        safe_response = model_response
        for phrase in self.passphrases:
            safe_attack = redact_passphrase(safe_attack, phrase)
            safe_response = redact_passphrase(safe_response, phrase)
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
                "results": self.results if self.done else None,
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

    def private_engine_state(self) -> dict:
        return {
            "registrations": [asdict(registration) for registration in self.registrations],
            "duels": [asdict(duel) for duel in self.duels.values()],
        }

    def _write_artifacts(self) -> None:
        assert self.results is not None
        if self.training is not None:
            records = self.training.finish(
                outcome={
                    **self.results,
                    "runtime_configuration": {**self.config.model_dump(exclude={"tokens"}), "seed": self.seed},
                    "engine_state": self.private_engine_state(),
                },
                environment={
                    "provider": self.config.model_provider,
                    "model": self.config.model_id,
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": self.config.response_token_cap,
                    "liveness_probes": 3,
                },
                environment_generations=self.model.generations,
                ownership_settled=self.ownership_settled,
            )
            self.training.write(records)
        results_uri = os.environ.get("COGAME_RESULTS_URI")
        replay_uri = os.environ.get("COGAME_SAVE_REPLAY_URI")
        if results_uri:
            write_data(
                results_uri,
                json.dumps(self.results, separators=(",", ":")),
                content_type="application/json",
            )
        if replay_uri:
            write_data(
                replay_uri,
                json.dumps(self._replay_payload(), separators=(",", ":")),
                content_type="application/json",
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


class OwnedServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, runtime: Runtime):
        super().__init__(config)
        self.runtime = runtime

    @contextmanager
    def capture_signals(self):
        yield

    async def serve(self, sockets=None):
        completed = False
        try:
            await super().serve(sockets=sockets)
            completed = True
        finally:
            await self.runtime.shutdown()
            if not completed and self.started:
                deadline = asyncio.get_running_loop().time() + CLEANUP_SECONDS
                inherited = shutdown_deadline.get()
                if inherited is not None and inherited[0] is not None:
                    deadline = min(deadline, inherited[0])
                closer = owned_task(self.shutdown(sockets=sockets))
                if not await settle({closer}, deadline, cancel=False):
                    await settle({closer}, deadline, cancel=True)
                    raise TimeoutError("server shutdown ownership unresolved")
                closer.result()


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    runtime = runtime_from_environment()
    app = create_app(runtime)
    host = os.environ.get("COGAME_HOST", "0.0.0.0")
    port = int(os.environ.get("COGAME_PORT", "8080"))
    logging.getLogger("websockets.client").setLevel(logging.WARNING)
    logging.getLogger("websockets.server").setLevel(logging.WARNING)
    server = OwnedServer(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            access_log=False,
            ws_max_size=16 * 1024 * 1024,
            ws_max_queue=8,
            timeout_graceful_shutdown=int(CLEANUP_SECONDS),
        ),
        runtime,
    )

    async def stop_server() -> None:
        await asyncio.sleep(0.5)
        server.should_exit = True

    runtime.on_episode_complete = stop_server
    main_owned(server.serve(), lambda: setattr(server, "should_exit", True))


if __name__ == "__main__":
    main()
