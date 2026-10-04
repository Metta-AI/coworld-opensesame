# Private language training evidence

The `native-learner` runnable uses the Coworld Messages sidecar. Its exact ordinary observations,
prompt, parser and capped action are shared by hosted play and private exports.
The environment model is selected by game configuration and remains frozen at temperature 0 and top-p 1.
Learner model and decoder overrides never select the defender. Missing native endpoints fail loudly.
`local-mock-4` and source-owned mock teacher cohorts are diagnostic configurations; they do not prove Haiku judgment or trained strength.

Set the exact `COWORLD_EPISODE_ID`, `COWORLD_GAME_VERSION` and committed `COWORLD_SOURCE_REVISION`.
`COGAME_SAVE_TRAJECTORY_URI` must be an absolute local file URI; development collectors may select `COWORLD_PRIVATE_TRAINING_DIR`.
The private append-only progress spool retains actual starts, requests, bytes, status, raw header pairs and reader ownership.
Partial invalid UTF-8 remains base64 evidence. No-response fields remain unobserved.
Only joined ownership produces the canonical SDK `CompleteEpisode` JSONL record. An unresolved owner preserves its writable spool
and auxiliary `.ownership.private` snapshot, withholding final results and replay.
All private files are mode 0600 and excluded from Docker/public replay artifacts.

The source-owned collector runs whole episodes with the shipped starter/leaky policy functions:

```sh
uv run --no-sync python -m opensesame.export_teacher --variant league-4 --games 10 --output /absolute/private/corpus
```

Repeat for every manifest variant. Seeds identify `open-sesame-<seed>` families across coupled variants.
The collector preserves whole episodes and emits SHA256 manifests marked unreviewed. It emits no training labels or train/validation split.
Modern Metta training import requires external content-bound source review for scripted teachers.
Model-backed labels require authenticated platform receipts and trusted episode/checkpoint context; game packets are insufficient.
The native process fixtures use synthetic HTTP responses and produce zero authenticated model receipts.
Draw-time sampler fields are preserved only when actually returned; greedy responses have no fabricated probabilities.

Before release, refresh every player to protocol 2, certify the exact game image with Coworld 0.1.56,
verify real inference access and private receipt joins, and review the frozen defender configuration.
Game source merges and diagnostic image certificates do not authorize production upload.
