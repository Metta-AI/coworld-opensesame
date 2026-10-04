"""Private native inference evidence; never part of public state or replay."""

from __future__ import annotations

import base64
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    model_validator,
)


class ReceivedHeaderPairs(BaseModel):
    """Actual HTTP parsed header pairs, retained only as private auxiliary evidence."""

    model_config = ConfigDict(hide_input_in_errors=True, extra="forbid")
    attempt_id: str
    pairs: list[tuple[str, str]]


class Attempt(BaseModel):
    model_config = ConfigDict(
        hide_input_in_errors=True,
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
    )

    attempt_id: str = Field(default_factory=lambda: str(uuid4()))
    policy: str = Field(min_length=1)
    origin: Literal["model", "teacher", "fallback", "human", "unknown"] = "model"
    inference_mode: Literal["text_action"] | None = "text_action"
    _received_header_pairs: ReceivedHeaderPairs | None = PrivateAttr(default=None)
    model: str | None = None
    prompt: JsonValue = None
    request: JsonValue = None
    response: JsonValue = None
    raw_response: str | None = None
    response_headers: dict[str, str] | None = None
    response_body_b64: str | None = None
    response_headers_b64: str | None = None
    response_complete: Annotated[bool, Field(strict=True)] | None = None
    response_reader_joined: Annotated[bool, Field(strict=True)] | None = None
    http_status: int | None = Field(default=None, strict=True, ge=100, le=599)
    provider_request_id: str | None = None
    decoder: JsonValue = None
    platform_call_id: str | None = None
    rejection_reason: str | None = None
    model_identity: str | None = None
    tokenizer_identity: str | None = None
    chat_template_sha256: str | None = None
    accepted: bool = False
    parsed_action: JsonValue = None
    stop_reason: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, strict=True, ge=0)
    output_tokens: int | None = Field(default=None, strict=True, ge=0)
    prompt_token_ids: list[Annotated[int, Field(strict=True, ge=0)]] | None = None
    sampled_token_ids: list[Annotated[int, Field(strict=True, ge=0)]] | None = None
    behavior_logprobs: list[Annotated[float, Field(strict=True, le=0)]] | None = None

    @model_validator(mode="after")
    def sampling_is_actual(self) -> Attempt:
        if self.response_body_b64 is not None:
            body = base64.b64decode(self.response_body_b64, validate=True)
            if isinstance(self.raw_response, str) and body != self.raw_response.encode("utf-8"):
                raise ValueError("received bytes differ from native text body")
        if self.response_headers_b64 is not None:
            base64.b64decode(self.response_headers_b64, validate=True)
        for tokens in (self.prompt_token_ids, self.sampled_token_ids):
            if tokens is not None and any(token < 0 for token in tokens):
                raise ValueError("token identities must be nonnegative")
        if self.behavior_logprobs is not None:
            if self.sampled_token_ids is None or len(self.behavior_logprobs) != len(self.sampled_token_ids):
                raise ValueError("actual token IDs and behavior probabilities must match")
            if any(probability > 0 for probability in self.behavior_logprobs):
                raise ValueError("log probabilities cannot be positive")
        return self
