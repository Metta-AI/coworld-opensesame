"""Private learner decisions, separate from frozen defender generations."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from opensesame.evidence import Attempt, ReceivedHeaderPairs
from opensesame.io import write_data
from opensesame.native import NativeRequest, NativeResponse, learner_prompt, parse_action


class NativeDecoder(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, hide_input_in_errors=True)
    max_tokens: int = Field(strict=True, gt=0)
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    timeout_ms: float = Field(gt=0)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    attempts: list[Attempt]
    selected_attempt_id: str | None


class Progress(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    type: Literal["attempt_progress"]
    request_id: str
    attempt: Attempt
    received_header_pairs: ReceivedHeaderPairs | None = None


class Capture(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, hide_input_in_errors=True)
    decision_id: str
    seat: int
    observation: dict[str, JsonValue]
    attempts: dict[str, Attempt] = Field(default_factory=dict)
    received_header_pairs: dict[str, ReceivedHeaderPairs] = Field(default_factory=dict)
    selected_attempt_id: str | None = None
    action_closed: bool = False
    executed_action: JsonValue = None
    action_status: Literal["accepted", "fallback", "missing"] = "missing"
    fallback_origin: str | None = "player did not respond"


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
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
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
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


class CompleteEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    schema_version: Literal["1"] = "1"
    episode: EpisodeRecord
    decisions: list[DecisionRecord]


class EnvironmentProgress(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    event_type: Literal["environment_progress"] = "environment_progress"
    attempt: Attempt
    received_header_pairs: ReceivedHeaderPairs | None


class UnsettledSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    event_type: Literal["ownership_unsettled"] = "ownership_unsettled"
    episode_id: str
    source_revision: str
    ownership_settled: Literal[False] = False
    captures: list[Capture]
    environment_generations: list[Attempt]


class DecisionAdmission:
    """Authenticated decision windows, enforced independently of recording."""

    def __init__(self, *, teacher_policies: dict[int, str] | None = None) -> None:
        self.teacher_policies = teacher_policies
        self.captures: dict[str, Capture] = {}
        self.attempt_owners: dict[str, str] = {}
        self.call_owners: dict[str, str] = {}
        self.finished = False

    def retain(self, record: BaseModel) -> None:
        return None

    def begin(self, request_id: str, seat: int, observation: dict) -> None:
        if self.finished or request_id in self.captures:
            raise ValueError("cannot open duplicate or completed decision")
        self.captures[request_id] = Capture(decision_id=request_id, seat=seat, observation=observation)
        self.retain(self.captures[request_id])

    def progress(self, seat: int, progress: Progress) -> None:
        if self.finished:
            raise ValueError("private progress arrived after writer seal")
        if progress.request_id not in self.captures:
            raise ValueError("native progress has no authenticated decision window")
        capture = self.captures[progress.request_id]
        if capture.seat != seat:
            raise ValueError("private attempt belongs to another authenticated seat")
        attempt_id = progress.attempt.attempt_id
        if attempt_id in self.attempt_owners and self.attempt_owners[attempt_id] != progress.request_id:
            raise ValueError("native attempt identity reused across decisions")
        if capture.action_closed and attempt_id not in capture.attempts:
            raise ValueError("new attempt started outside its decision window")
        if capture.action_closed and attempt_id in capture.attempts:
            previous = capture.attempts[attempt_id]
            if previous.response_complete is True and previous.response_reader_joined is True:
                raise ValueError("native progress arrived after its terminal decision evidence")
        self.attempt_owners[attempt_id] = progress.request_id
        call_id = progress.attempt.platform_call_id
        if call_id is not None:
            if call_id in self.call_owners and self.call_owners[call_id] != attempt_id:
                raise ValueError("native platform call reused across attempts")
            self.call_owners[call_id] = attempt_id
        if progress.received_header_pairs is not None:
            pairs = progress.received_header_pairs
            if pairs.attempt_id != progress.attempt.attempt_id:
                raise ValueError("received headers belong to another attempt")
            if (
                pairs.attempt_id in capture.received_header_pairs
                and capture.received_header_pairs[pairs.attempt_id] != pairs
            ):
                raise ValueError("received header pairs cannot be rewritten")
            capture.received_header_pairs[pairs.attempt_id] = pairs.model_copy(deep=True)
        attempt = progress.attempt.model_copy(deep=True)
        attempt.accepted = False
        if attempt.origin != "model":
            attempt.origin = "unknown"
        if attempt.attempt_id in capture.attempts:
            started = capture.attempts[attempt.attempt_id]
            for field in ("policy", "origin", "prompt", "request", "decoder"):
                if getattr(attempt, field) != getattr(started, field):
                    raise ValueError("native attempt cannot change its started request")
            for field in ("response_body_b64", "response_headers_b64"):
                before = getattr(started, field)
                after = getattr(attempt, field)
                if before is not None:
                    if after is None or not base64.b64decode(after, validate=True).startswith(
                        base64.b64decode(before, validate=True)
                    ):
                        raise ValueError("received native byte prefix cannot be rewritten")
                    if started.response_complete is True and after != before:
                        raise ValueError("completed native bytes cannot be rewritten")
            if started.response_complete is True and attempt.response_complete is not True:
                raise ValueError("completed response cannot become incomplete")
            for field in (
                "http_status",
                "response_headers",
                "platform_call_id",
                "provider_request_id",
                "model_identity",
                "tokenizer_identity",
                "chat_template_sha256",
            ):
                before = getattr(started, field)
                if before is not None and before != getattr(attempt, field):
                    raise ValueError("received native identity cannot be rewritten")
        capture.attempts[attempt.attempt_id] = attempt
        self.retain(progress)

    def consume(self, request_id: str, response: dict | None, executed: dict, *, fallback: str | None) -> bool:
        capture = self.captures[request_id]
        capture.executed_action = executed
        capture.action_status = "fallback" if fallback else "accepted"
        capture.fallback_origin = fallback
        if response is None:
            capture.action_closed = True
            self.retain(capture)
            return True
        if self.teacher_policies is not None:
            text = json.dumps(executed, separators=(",", ":"))
            attempt = Attempt(
                policy=self.teacher_policies[capture.seat],
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
            capture.action_closed = True
            self.retain(capture)
            return True
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
            capture.action_closed = True
            self.retain(capture)
            return True
        evidence = Evidence.model_validate(response["_private"])
        if len({a.attempt_id for a in evidence.attempts}) != len(evidence.attempts):
            raise ValueError("duplicate native attempt identity")
        for attempt in evidence.attempts:
            self.progress(capture.seat, Progress(type="attempt_progress", request_id=request_id, attempt=attempt))
        capture.action_closed = True
        self.retain(capture)
        if fallback is not None or evidence.selected_attempt_id is None:
            return True
        selected = capture.attempts[evidence.selected_attempt_id]
        selected.parsed_action = parse_action(
            TypeAdapter(str, config=ConfigDict(hide_input_in_errors=True)).validate_python(selected.response),
            capture.observation,
        )
        expected_prompt = learner_prompt(capture.observation)
        if (
            selected.origin != "model"
            or selected.inference_mode != "text_action"
            or selected.prompt != expected_prompt
            or selected.parsed_action != executed
            or selected.response_complete is not True
            or selected.response_reader_joined is not True
            or selected.http_status != 200
        ):
            selected.rejection_reason = "native response/prompt differs from consumed player action"
            capture.executed_action = None
            capture.action_status = "missing"
            capture.fallback_origin = "native-evidence-mismatch"
            return False
        request = NativeRequest.model_validate(selected.request)
        served = NativeResponse.model_validate_json(
            TypeAdapter(str, config=ConfigDict(hide_input_in_errors=True)).validate_python(selected.raw_response)
        )
        sampled = served.sampling_evidence
        decoder = NativeDecoder.model_validate(selected.decoder)
        if (
            request.system != expected_prompt[0]["content"]
            or decoder.max_tokens != request.max_tokens
            or decoder.temperature != request.temperature
            or decoder.top_p != request.top_p
            or selected.prompt_token_ids != (sampled.prompt_token_ids if sampled is not None else None)
            or selected.sampled_token_ids != (sampled.completion_token_ids if sampled is not None else None)
            or selected.behavior_logprobs != (sampled.behavior_log_probs if sampled is not None else None)
            or (
                sampled is not None
                and (request.temperature != 1 or request.top_p != 1 or request.max_tokens != sampled.max_new_tokens)
            )
            or [message.model_dump() for message in request.messages] != expected_prompt[1:]
            or selected.response != "".join(block.text for block in served.content if block.type == "text")
            or selected.model != served.model
            or selected.input_tokens != served.usage.input_tokens
            or selected.output_tokens != served.usage.output_tokens
            or selected.stop_reason != (sampled.stop_reason if sampled is not None else served.stop_reason)
        ):
            selected.rejection_reason = "native request/response differs from captured completion"
            capture.executed_action = None
            capture.action_status = "missing"
            capture.fallback_origin = "native-evidence-mismatch"
            return False
        selected.accepted = True
        selected.rejection_reason = None
        capture.selected_attempt_id = selected.attempt_id
        return True


class Training(DecisionAdmission):
    def __init__(self, seed: int, *, teacher_policies: dict[int, str] | None = None) -> None:
        super().__init__(teacher_policies=teacher_policies)
        self.episode_id = os.environ["COWORLD_EPISODE_ID"]
        self.game_version = os.environ["COWORLD_GAME_VERSION"]
        self.source_revision = os.environ["COWORLD_SOURCE_REVISION"]
        if not self.episode_id or not self.game_version or re.fullmatch(r"[a-f0-9]{40}", self.source_revision) is None:
            raise ValueError("private training requires episode, game version, and full source commit pins")
        self.image_digest = os.environ.get("COWORLD_GAME_IMAGE_DIGEST")
        self.seed_family = f"open-sesame-{seed}"
        if "COWORLD_PRIVATE_TRAINING_DIR" in os.environ:
            directory = Path(os.environ["COWORLD_PRIVATE_TRAINING_DIR"])
            self.path = directory / "decisions.jsonl"
        else:
            uri = urlsplit(os.environ["COGAME_SAVE_TRAJECTORY_URI"])
            if uri.scheme != "file" or uri.netloc or uri.query or uri.fragment:
                raise ValueError("private trajectory requires an absolute local file URI")
            self.path = Path(unquote(uri.path))
        if not self.path.is_absolute():
            raise ValueError("private trajectory path must be absolute")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        self.spool = self.path.with_suffix(".progress.private")
        self.writer = os.fdopen(os.open(self.spool, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")

    def retain(self, record: BaseModel) -> None:
        if self.finished:
            raise ValueError("private writer is sealed")
        self.writer.write(record.model_dump_json() + "\n")
        self.writer.flush()
        os.fsync(self.writer.fileno())

    async def record_environment(self, attempt: Attempt) -> None:
        self.retain(EnvironmentProgress(attempt=attempt, received_header_pairs=attempt._received_header_pairs))

    def retain_unsettled(self, environment_generations: list[Attempt]) -> None:
        snapshot = UnsettledSnapshot(
            episode_id=self.episode_id,
            source_revision=self.source_revision,
            captures=list(self.captures.values()),
            environment_generations=environment_generations,
        )
        write_data(
            str(self.path.with_suffix(".ownership.private")),
            snapshot.model_dump_json() + "\n",
            content_type="application/x-ndjson",
            private=True,
        )
        # The spool stays open and writable until its actual owner joins.

    def finish(
        self,
        *,
        outcome: dict,
        environment: dict,
        environment_generations: list[Attempt],
        ownership_settled: bool,
        completed: bool = True,
    ) -> list:
        if self.finished:
            raise ValueError("episode already finished")
        if not ownership_settled:
            raise ValueError("cannot seal private evidence while ownership is unresolved")
        self.finished = True
        self.writer.close()
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
        participants = (
            [
                {"seat": str(seat), "score": score, "eligible": outcome["eligible"][seat]}
                for seat, score in enumerate(outcome["scores"])
            ]
            if completed
            else None
        )
        terminal = EpisodeRecord(
            episode_id=self.episode_id,
            seed_family=self.seed_family,
            game_version=self.game_version,
            source_revision=self.source_revision,
            image_digest=self.image_digest,
            status="completed" if completed else "truncated",
            outcome={
                **outcome,
                "ownership_settled": ownership_settled,
                "native_received_header_pairs": {
                    capture.decision_id: [
                        pairs.model_dump(mode="json") for pairs in capture.received_header_pairs.values()
                    ]
                    for capture in self.captures.values()
                    if capture.received_header_pairs
                },
                "environment_policy": environment,
                "environment_generations": [a.model_dump(mode="json") for a in environment_generations],
            },
            participant_outcomes=participants,
        )
        return [*decisions, terminal]

    def write(self, records: list[DecisionRecord | EpisodeRecord]) -> None:
        terminal = TypeAdapter(EpisodeRecord).validate_python(records[-1])
        decisions = TypeAdapter(list[DecisionRecord]).validate_python(records[:-1])
        complete = CompleteEpisode(episode=terminal, decisions=decisions)
        write_data(str(self.path), complete.model_dump_json() + "\n", content_type="application/x-ndjson", private=True)
