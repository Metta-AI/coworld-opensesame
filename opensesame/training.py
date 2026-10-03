"""Private learner decisions, separate from frozen defender generations."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from opensesame.io import artifact_method, write_data
from opensesame.native import Attempt, learner_prompt, parse_action


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempts: list[Attempt]
    selected_attempt_id: str | None


class Progress(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["attempt_progress"]
    request_id: str
    attempt: Attempt


class Capture(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    decision_id: str
    seat: int
    observation: dict[str, JsonValue]
    attempts: dict[str, Attempt] = Field(default_factory=dict)
    selected_attempt_id: str | None = None
    executed_action: JsonValue = None
    action_status: Literal["accepted", "fallback", "missing"] = "missing"
    fallback_origin: str | None = "player did not respond"


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"] = "1"
    event_type: Literal["decision"] = "decision"
    episode_id: str
    decision_id: str
    decision_index: int
    game: Literal["open-sesame"] = "open-sesame"
    game_version: str
    source_revision: str
    image_digest: str | None
    seat: str
    visibility: Literal["private"] = "private"
    observation: JsonValue
    prompt: JsonValue
    attempts: list[Attempt]
    selected_attempt_id: str | None
    executed_action: JsonValue
    action_status: Literal["accepted", "fallback", "missing"]
    fallback_origin: str | None
    terminal: bool = False


class EpisodeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"] = "1"
    event_type: Literal["episode"] = "episode"
    episode_id: str
    seed_family: str
    game: Literal["open-sesame"] = "open-sesame"
    game_version: str
    source_revision: str
    image_digest: str | None
    status: Literal["completed", "truncated"]
    outcome: JsonValue
    participant_outcomes: JsonValue


class Training:
    def __init__(self, seed: int, *, teacher_policy: str | None = None) -> None:
        self.episode_id = os.environ["COWORLD_EPISODE_ID"]
        self.game_version = os.environ["COWORLD_GAME_VERSION"]
        self.source_revision = os.environ["COWORLD_SOURCE_REVISION"]
        if not self.episode_id or not self.game_version or re.fullmatch(r"[a-f0-9]{40}", self.source_revision) is None:
            raise ValueError("private training requires episode, game version, and full source commit pins")
        self.image_digest = os.environ.get("COWORLD_GAME_IMAGE_DIGEST")
        self.seed_family = f"open-sesame-{seed}"
        self.teacher_policy = teacher_policy
        self.captures: dict[str, Capture] = {}
        self.finished = False

    def begin(self, request_id: str, seat: int, observation: dict) -> None:
        if self.finished or request_id in self.captures:
            raise ValueError("cannot open duplicate or completed decision")
        self.captures[request_id] = Capture(decision_id=request_id, seat=seat, observation=observation)

    def progress(self, seat: int, progress: Progress) -> None:
        if self.finished:
            return
        capture = self.captures[progress.request_id]
        if capture.seat != seat:
            raise ValueError("private attempt belongs to another authenticated seat")
        attempt = progress.attempt.model_copy(deep=True)
        attempt.accepted = False
        if attempt.origin != "model":
            attempt.origin = "unknown"
        if attempt.attempt_id in capture.attempts:
            started = capture.attempts[attempt.attempt_id]
            if attempt.prompt != started.prompt or attempt.request != started.request:
                raise ValueError("native attempt cannot change its started request")
        capture.attempts[attempt.attempt_id] = attempt

    def consume(self, request_id: str, response: dict | None, executed: dict, *, fallback: str | None) -> None:
        capture = self.captures[request_id]
        capture.executed_action = executed
        capture.action_status = "fallback" if fallback else "accepted"
        capture.fallback_origin = fallback
        if response is None:
            return
        if self.teacher_policy is not None:
            text = json.dumps(executed, separators=(",", ":"))
            attempt = Attempt(
                policy=self.teacher_policy,
                origin="teacher",
                inference_mode="text_action",
                prompt=learner_prompt(capture.observation),
                request=None,
                model=None,
                decoder=None,
                response=text,
                parsed_action=parse_action(text, capture.observation) if fallback is None else None,
                accepted=fallback is None,
                rejection_reason=fallback,
            )
            capture.attempts[attempt.attempt_id] = attempt
            capture.selected_attempt_id = attempt.attempt_id if attempt.accepted else None
            return
        if "_private" not in response:
            attempt = Attempt(
                policy="external-action",
                origin="unknown",
                inference_mode="text_action",
                prompt=learner_prompt(capture.observation),
                request=None,
                model=None,
                decoder=None,
                response=json.dumps(executed),
                parsed_action=executed,
                accepted=fallback is None,
                rejection_reason=fallback,
            )
            capture.attempts[attempt.attempt_id] = attempt
            capture.selected_attempt_id = attempt.attempt_id if attempt.accepted else None
            return
        evidence = Evidence.model_validate(response["_private"])
        if len({a.attempt_id for a in evidence.attempts}) != len(evidence.attempts):
            raise ValueError("duplicate native attempt identity")
        for attempt in evidence.attempts:
            self.progress(capture.seat, Progress(type="attempt_progress", request_id=request_id, attempt=attempt))
        if fallback is not None or evidence.selected_attempt_id is None:
            return
        selected = capture.attempts[evidence.selected_attempt_id]
        selected.parsed_action = parse_action(TypeAdapter(str).validate_python(selected.response), capture.observation)
        expected_prompt = learner_prompt(capture.observation)
        if selected.prompt != expected_prompt or selected.parsed_action != executed:
            selected.rejection_reason = "native response/prompt differs from consumed player action"
            capture.action_status = "fallback"
            capture.fallback_origin = "player-parser-mismatch"
            return
        selected.accepted = True
        selected.rejection_reason = None
        capture.selected_attempt_id = selected.attempt_id

    def finish(self, *, outcome: dict, environment: dict, environment_generations: list[Attempt]) -> list:
        if self.finished:
            raise ValueError("episode already finished")
        self.finished = True
        decisions = [
            DecisionRecord(
                episode_id=self.episode_id,
                decision_id=c.decision_id,
                decision_index=index,
                game_version=self.game_version,
                source_revision=self.source_revision,
                image_digest=self.image_digest,
                seat=str(c.seat),
                observation=c.observation,
                prompt=learner_prompt(c.observation),
                attempts=list(c.attempts.values()),
                selected_attempt_id=c.selected_attempt_id,
                executed_action=c.executed_action,
                action_status=c.action_status,
                fallback_origin=c.fallback_origin,
            )
            for index, c in enumerate(self.captures.values())
        ]
        calls_finished = all(
            a.latency_ms is not None for c in self.captures.values() for a in c.attempts.values() if a.origin == "model"
        )
        participants = [
            {"seat": str(seat), "score": score, "eligible": outcome["eligible"][seat]}
            for seat, score in enumerate(outcome["scores"])
        ]
        terminal = EpisodeRecord(
            episode_id=self.episode_id,
            seed_family=self.seed_family,
            game_version=self.game_version,
            source_revision=self.source_revision,
            image_digest=self.image_digest,
            status="completed" if calls_finished else "truncated",
            outcome={
                **outcome,
                "environment_policy": environment,
                "environment_generations": [a.model_dump(mode="json") for a in environment_generations],
            },
            participant_outcomes=participants,
        )
        return [*decisions, terminal]

    def write(self, records: list) -> None:
        payload = "".join(record.model_dump_json() + "\n" for record in records)
        if "COWORLD_PRIVATE_TRAINING_DIR" in os.environ:
            directory = Path(os.environ["COWORLD_PRIVATE_TRAINING_DIR"])
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)
            write_data(str(directory / "decisions.jsonl"), payload, content_type="application/x-ndjson", private=True)
        uri = os.environ.get("COGAME_SAVE_TRAJECTORY_URI")
        if uri:
            write_data(
                uri,
                payload,
                content_type="application/x-ndjson",
                http_method=artifact_method("COGAME_SAVE_TRAJECTORY_METHOD"),
                private=True,
            )
