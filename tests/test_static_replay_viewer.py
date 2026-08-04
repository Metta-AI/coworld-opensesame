import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).parents[1]
HOOK = REPO / "tools" / "build_replay_viewer.sh"
VIEWER = REPO / "replay_viewer" / "index.html"


def test_manifest_declares_static_replay_viewer_build_hook() -> None:
    manifest = json.loads((REPO / "coworld_manifest_template.json").read_text())

    assert manifest["game"]["replay_viewer"] == {"bundle": "build/static-replay-viewer"}
    assert HOOK.is_file()
    assert os.access(HOOK, os.X_OK)


def test_static_replay_viewer_build_is_clean_and_self_contained(tmp_path: Path) -> None:
    output = tmp_path / "static-replay-viewer"
    output.mkdir()
    (output / "stale-file").write_text("must disappear")

    subprocess.run([HOOK, output], cwd=REPO, check=True)

    assert not (output / "stale-file").exists()
    assert (output / "index.html").read_bytes() == VIEWER.read_bytes()


def test_static_viewer_uses_direct_replay_fetch_and_visible_failures() -> None:
    source = VIEWER.read_text()

    assert 'new URLSearchParams(location.search).get("replay")' in source
    assert 'fetch(replayUrl,{credentials:"omit"})' in source
    assert "response.arrayBuffer()" in source
    assert 'scores.setAttribute("role","alert")' in source
    assert "Replay could not be loaded" in source
    assert "WebSocket" not in source
