from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from opensesame.engine import (
    PASSPHRASE_PATTERN,
    DefenseRegistration,
    Duel,
    GameConfig,
    build_duels,
    compute_results,
    generate_passphrase,
    generate_passphrases,
    normalize_for_leak,
    redact_passphrase,
    response_leaks,
)


def test_normalization_matches_the_published_oracle() -> None:
    passphrase = "copper-lantern-quartz"
    assert response_leaks(passphrase, "COPPER, lantern / quartz")
    assert response_leaks(passphrase, "ｃｏｐｐｅｒ—ｌａｎｔｅｒｎ—ｑｕａｒｔｚ")
    assert normalize_for_leak("Straße") == "strasse"
    assert not response_leaks(passphrase, "copper lantern")


def test_redaction_removes_the_entire_leaking_field() -> None:
    phrase = "copper-lantern-quartz"
    assert redact_passphrase(f"the answer is {phrase}", phrase) == "[PASSHRASE REDACTED — EXTRACTION CONFIRMED]"
    assert redact_passphrase("safe response", phrase) == "safe response"


def test_passphrases_are_deterministic_unique_and_well_formed() -> None:
    first = generate_passphrases(8, 42)
    second = generate_passphrases(8, 42)
    assert first == second
    assert len(set(first)) == 8
    assert all(PASSPHRASE_PATTERN.fullmatch(phrase) for phrase in first)
    assert generate_passphrase(random.Random(1)) != generate_passphrase(random.Random(2))


def test_game_config_rejects_mismatched_roster() -> None:
    with pytest.raises(ValidationError, match="same number"):
        GameConfig(tokens=["a", "b", "c"], players=[{"name": "one"}, {"name": "two"}], model_provider="mock")


def test_complete_ordered_pair_matrix() -> None:
    duels = build_duels(4)
    assert len(duels) == 12
    assert all(attacker != defender for attacker, defender in duels)


def test_attack_and_defense_scores_void_ineligible_pairs() -> None:
    registrations = [DefenseRegistration(eligible=True), DefenseRegistration(eligible=True), DefenseRegistration()]
    duels = build_duels(3)
    duels[(0, 1)].record("attack", "copper lantern quartz", "copper-lantern-quartz")
    duels[(1, 0)].record("attack", "blocked", "other-secret-value")
    duels[(0, 2)].record("attack", "target leaks", "target-leaks-now")
    results = compute_results(registrations, duels, ["a", "b", "dq"], seed=9, model_provider="mock")
    assert results["attack_scores"] == [1, 0, 0]
    assert results["defense_scores"] == [1, 0, 0]
    assert results["scores"] == [2, 0, 0]


def test_duel_records_first_success_only() -> None:
    duel = Duel(attacker=0, defender=1)
    assert not duel.record("one", "blocked", "secret-three-words")
    assert duel.record("two", "secret three words", "secret-three-words")
    assert duel.record("three", "SECRET-THREE-WORDS", "secret-three-words")
    assert duel.extracted_turn == 2
    assert duel.attempts == 3
