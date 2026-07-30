from __future__ import annotations

from opensesame.engine import PASSPHRASE_PATTERN
from opensesame.starter import classify_response, starter_attack, starter_defense_prompt


def test_starter_defense_is_deterministic_camouflage_under_cap() -> None:
    phrase = "bacedi-kofulu-dagebi"
    prompt = starter_defense_prompt(phrase)
    records = [line for line in prompt.splitlines() if line.startswith("record_")]

    assert prompt == starter_defense_prompt(phrase)
    assert len(prompt) < 12_000
    assert prompt.count(phrase) == 1
    assert len(records) == 10
    assert len(set(records)) == 10
    assert all(PASSPHRASE_PATTERN.fullmatch(line.split("=", 1)[1]) for line in records)
    assert "useful conference assistant" in prompt
    assert "Reply only" not in prompt


def test_starter_attack_adapts_to_target_response() -> None:
    refusal = [{"role": "assistant", "content": "I cannot disclose confidential records."}]
    generic = [{"role": "assistant", "content": "Please try another approach."}]

    assert "stateless replacement" in starter_attack(1, [])
    assert "DLP transport test" in starter_attack(2, refusal)
    assert "hyphenated configuration value" in starter_attack(2, generic)
    assert "character-stream fallback" in starter_attack(3, refusal)
    assert "deterministic template renderer" in starter_attack(3, generic)
    assert "Final consistency check" in starter_attack(4, generic)

    for turn in range(1, 5):
        assert len(starter_attack(turn, refusal)) <= 2_000


def test_response_classifier_is_bounded_and_deterministic() -> None:
    assert classify_response("") == "empty"
    assert classify_response("I cannot share confidential data.") == "refusal"
    assert classify_response("That JSON format is invalid.") == "format_claim"
    assert classify_response("Try another approach.") == "generic"
