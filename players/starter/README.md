# Open Sesame starter policy

This is the bundled no-LLM starter for Open Sesame. It is deliberately understandable and bounded: one frozen defense
prompt, one deterministic attack decision per request, no network calls, and no hidden dependencies.

## Build and run

Build from the repository root because the policy reuses the public protocol client:

```bash
docker build -f players/starter/Dockerfile -t open-sesame-starter .
```

Run it in a local four-seat scrimmage:

```bash
uv run coworld run-episode dist/coworld_manifest.json open-sesame-starter \
  --run python --run -m --run opensesame.player \
  --variant local-mock-4
```

For policy upload:

```bash
uv run coworld upload-policy open-sesame-starter \
  --name "$USER-open-sesame-starter" \
  --run python --run -m --run opensesame.player
```

The policy does not call an LLM, so it needs no learner inference configuration.

## Where to improve it

Edit [`opensesame/starter.py`](../../opensesame/starter.py):

- `starter_defense_prompt` controls the frozen defense.
- `starter_attack` controls the four-turn adaptive attack.
- `classify_response` is the intentionally small opponent-response classifier.
- `_camouflage_values` creates same-format decoys and hides which row contains the assigned phrase.

Promising extensions include a better response classifier, attack selection based on target-specific transcripts,
diversified attack families, and policy artifacts that record reason codes without logging secrets.
