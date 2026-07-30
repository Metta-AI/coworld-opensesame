# Open Sesame spectator protocol 1.0.0

Spectators connect read-only to `/global`. The first frame is a `snapshot` containing current phase, prior public
events, and final results when available. Later frames are individual events.

Event types:

- `phase`: `waiting`, `registration`, `attack`, or `complete`.
- `registration`: seat/name plus eligibility and a public rejection reason.
- `duel_turn`: attacker, defender, turn, and extraction result.
- `final_scores`: the complete results object.

While an episode is active, `duel_turn` sets `payloads_embargoed: true` and omits attack/response text. The replay
contains those fields after passphrase redaction. The global stream never contains assigned passphrases or defense
prompts.
