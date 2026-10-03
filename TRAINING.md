# Open Sesame private training

The learner submits a frozen defense prompt and attack messages. The game calls
the configured defender model and applies the deterministic extraction oracle.
Defender and liveness generations define the environment, never learner targets.
They ignore learner checkpoint and decoder overrides.

Set `OPEN_SESAME_PROFILE=native` for the shipped Messages learner. It uses
`COWORLD_LLM_ENDPOINT`, `COWORLD_LLM_MODEL`, `COWORLD_LLM_TEMPERATURE`, and
`COWORLD_LLM_TOP_P`. Production and teacher export share the learner prompt,
text parser, private observation, character caps, and complete game engine.
Started requests reach the authenticated socket before HTTP; received headers,
raw bodies, sampling evidence, and final elapsed time update that same attempt.
Failed generations terminate the player and the game applies its bounded
ordinary fallback. No provider credentials or direct-provider fallback exists.

The game captures private JSONL when `COGAME_SAVE_TRAJECTORY_URI` or
`COWORLD_PRIVATE_TRAINING_DIR` is set. It requires `COWORLD_EPISODE_ID`,
`COWORLD_GAME_VERSION`, and a full commit in `COWORLD_SOURCE_REVISION`.
Directories use mode700; local artifacts use mode600 and atomic replacement.
Source-bound model text is parsed independently and compared with the consumed
wire action. Attribution mismatches receive no learner label. External teacher
and human claims become unknown. Public results and replay keep their existing
secret redaction and never contain private generation evidence.

The private terminal outcome includes scores, eligibility, participant outcomes,
the frozen environment configuration, and every started defender generation.
An unfinished learner generation makes the training episode truncated.
Never import defender generations as decision targets or infer token
probabilities from greedy outputs.

From a clean committed checkout, export at least ten complete diagnostic games:

```bash
uv run python -m opensesame.export_teacher --variant duel-2 --games 10 --output /tmp/open-sesame-duel
```

The exporter explicitly uses the deterministic mock environment and alternates
the shipped starter/leaky policies. These labels verify the production engine,
prompt/parser, and episode export. They do not measure prompt quality against
the production defender or establish saved-model strength. Qualify JSONL with
the Coworld training consumer, split by `open-sesame-<seed>` across variants,
and separately review teacher quality and frozen production-model behavior.
Production publication requires the reviewed training runtime and immutable
source/image provenance readback.
