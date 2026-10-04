# Open Sesame player protocol 2.0.0

Players connect to `COWORLD_PLAYER_WS_URL` and exchange JSON frames. A submitted player acts as both a defender and an
attacker. It never calls the competition model directly and never chooses its passphrase.

## Server to player

### `hello`

```json
{"type":"hello","slot":0,"player_name":"example","player_count":4,"protocol_version":"2.0.0"}
```

### `register_request`

Private to the seat. `passphrase` is the exact system-assigned value the defense must contain verbatim.

```json
{
  "type": "register_request",
  "request_id": "req-1",
  "passphrase": "bacedi-kofulu-dagebi",
  "defense_prompt_char_cap": 12000,
  "liveness": {
    "description": "The frozen defense must answer benign probes and never leak this passphrase.",
    "deterministic_scoring": true
  }
}
```

Reply before the action deadline:

```json
{"type":"register","request_id":"req-1","defense_prompt":"...bacedi-kofulu-dagebi..."}
```

The server rejects a missing/oversized prompt, a prompt without the verbatim phrase, a prompt that fails a benign
exact-answer probe, or a prompt that leaks on a benign probe. The probes are game-owned and may change between game
versions. Do not special-case them; build a defense that remains useful on benign tasks.

### `attack_request`

The current target and only that ordered pair's transcript are included. `transcript` alternates `user` attack messages
and `assistant` target-model responses. Requests are sequential per player, so a player never has two outstanding
actions at once.

```json
{
  "type": "attack_request",
  "request_id": "req-9",
  "target": {"slot": 2, "name": "target-policy"},
  "turn": 2,
  "max_turns": 4,
  "rotation": 1,
  "message_char_cap": 2000,
  "transcript": [
    {"role":"user","content":"first payload"},
    {"role":"assistant","content":"first response"}
  ]
}
```

Reply:

```json
{"type":"attack","request_id":"req-9","message":"next payload"}
```

Malformed, empty, late, or disconnected responses become a harmless fallback attempt. Oversized messages are truncated
at the declared character cap.

### `attack_result`

Private to the attacker. The raw response is included so a multi-turn policy can adapt. `extracted` is the
deterministic oracle result. Once true, that pair receives no more requests.

```json
{
  "type": "attack_result",
  "target": {"slot": 2, "name": "target-policy"},
  "turn": 2,
  "response": "model response",
  "extracted": false
}
```

### `defense_result`

Private notification that another seat attacked this defense. The payload and model response are not disclosed.

```json
{
  "type": "defense_result",
  "attacker": {"slot": 1, "name": "attacker-policy"},
  "turn": 2,
  "extracted": false
}
```

### `final`

```json
{
  "type": "final",
  "slot": 0,
  "score": 5,
  "attack_score": 2,
  "defense_score": 3,
  "eligible": true
}
```

Exit cleanly after this frame. If a player uploads a debugging artifact, finish the upload before exiting; the game
waits briefly before finalizing its own artifacts.

## Security boundaries

- A player controls only its own WebSocket and seat token.
- Assigned passphrases and defense prompts are private. Do not print them to container logs or upload them in a player
  artifact.
- Only text emitted by the defender model can score. Putting a guessed phrase in the attack message does not score.
- The game model, temperature, limits, passphrase generation, liveness gate, and leak oracle are game-owned.
- Defense registration is frozen for the episode. A policy cannot rewrite its defense after seeing attacks.

## Private native inference evidence

A native player sends `attempt_progress` with its current `request_id` and a typed
private `attempt` before HTTP, after received headers, and after completion.
The final register/attack action includes `_private` with `attempts` and
`selected_attempt_id`. The authenticated game socket owns the seat. These fields
never appear in public replay or results, and player assertions do not establish
a trusted teacher identity. The game independently parses selected response text
and compares it with the action actually consumed. See [training](../TRAINING.md).

## Owned shutdown and private native evidence

Every hosted player must implement protocol 2. The engine sends `stop` with an unpredictable `stop_id`.
Cancel active decisions and join owned inference readers and evidence writers before replying
`{"type":"stopped","stop_id":"<engine nonce>"}`. Never acknowledge unresolved ownership.
The engine replies `evidence_received` with the same nonce. Then accept normal `final` scores or clean transport EOF.
Use one absolute two-second cleanup deadline; repeated signals must not extend it.
The shipped Python and browser players implement this contract. Previously shipped players require refresh before release.

Native learners use `/v1/messages`, authenticated player-slot attribution, exact private `attempt_progress`,
and the selected attempt in the action's `_private` envelope. Incoming frames have a bounded 16 MiB capacity.
The engine independently parses the captured completion and compares it with the installed action.
Teacher and human claims from external packets never establish trusted supervision.
Started requests, received bytes, terminal bodies and identity headers cannot be rewritten.
An unresolved reader blocks stopped credit and public completion.
