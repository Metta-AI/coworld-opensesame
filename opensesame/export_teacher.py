"""Whole production-engine episodes with scripted players and an explicit mock environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from opensesame.engine import GameConfig
from opensesame.io import write_data
from opensesame.lifecycle import OwnershipUnsettled, main_owned
from opensesame.model import MockModel
from opensesame.player import attack_message, defense_prompt
from opensesame.server import Runtime


class TeacherRuntime(Runtime):
    async def request_player(self, slot: int, payload: dict, *, timeout: float | None = None) -> dict:
        profile = "starter" if slot % 2 == 0 else "leaky"
        if payload["type"] == "register_request":
            return {"defense_prompt": defense_prompt(profile, payload["passphrase"])}
        return {"message": attack_message(payload["turn"], profile, payload["transcript"])}


async def export(variant: str, output: Path, games: int, seed_start: int) -> None:
    if games < 10:
        raise ValueError("whole-episode collection requires at least ten complete games")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if dirty:
        raise ValueError("source-pinned teacher export requires a clean committed checkout")
    manifest = json.loads(Path("coworld_manifest_template.json").read_text())
    config = next(v["game_config"] for v in manifest["variants"] if v["id"] == variant)
    collected = []
    for seed in range(seed_start, seed_start + games):
        cfg = GameConfig.model_validate(
            {
                **config,
                "seed": seed,
                "model_provider": "mock",
                "model_id": "mock-certification",
                "minimum_episode_seconds": 0,
                "tokens": [f"teacher-{seat}" for seat in range(len(config["players"]))],
            }
        )
        os.environ.update(
            COWORLD_EPISODE_ID=f"opensesame-teacher-{variant}-{source[:7]}-{seed}",
            COWORLD_SOURCE_REVISION=source,
            COWORLD_GAME_VERSION=f"source-{source}-mock",
            COWORLD_PRIVATE_TRAINING_DIR=str(output / variant / str(seed)),
        )
        runtime = TeacherRuntime(
            cfg,
            model=MockModel(),
            teacher_policies={
                seat: "scripted-starter" if seat % 2 == 0 else "scripted-leaky" for seat in range(len(cfg.tokens))
            },
        )
        try:
            await runtime.run_episode()
        finally:
            await runtime.shutdown()
        if not runtime.done:
            raise OwnershipUnsettled("teacher episode did not settle")
        path = output / variant / str(seed) / "decisions.jsonl"
        with path.open("rb") as source_file:
            digest = hashlib.file_digest(source_file, "sha256").hexdigest()
        collected.append(
            {"path": str(path.relative_to(output)), "sha256": digest, "seed_family": f"open-sesame-{seed}"}
        )
    write_data(
        str(output / f"{variant}.manifest.json"),
        json.dumps(
            {
                "format": "coworld_complete_episodes_v1",
                "game": "open-sesame",
                "variant": variant,
                "source_revision": source,
                "environment_profile": "diagnostic-mock",
                "training_review": "unreviewed",
                "authenticated_model_receipts": 0,
                "episodes": collected,
            },
            separators=(",", ":"),
        )
        + "\n",
        content_type="application/json",
        private=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    main_owned(export(args.variant, args.output, args.games, args.seed_start))


if __name__ == "__main__":
    main()
