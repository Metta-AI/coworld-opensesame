"""Engine ownership controls for all hosted player connections."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr


class StopControl(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    stop_id: StrictStr = Field(min_length=1, max_length=128)


class Stop(StopControl):
    type: Literal["stop"] = "stop"


class Stopped(StopControl):
    type: Literal["stopped"] = "stopped"


class EvidenceReceived(StopControl):
    type: Literal["evidence_received"] = "evidence_received"
