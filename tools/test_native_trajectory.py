import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from opensesame.model import MockModel
from opensesame.native import LEARNER_SYSTEM
from opensesame.player import attack_message, defense_prompt

root = Path(__file__).resolve().parents[1]

source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
destination = Path(sys.argv[1])
destination.mkdir(mode=0o700)
flow = "accepted"
archives = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        call_id = str(uuid4())
        learner = body["system"] == LEARNER_SYSTEM
        slot = int(self.headers["X-Coworld-Player-Slot"])
        if learner and flow == "silence":
            archives.append({"call_id": call_id, "slot": slot, "request": body, "response": None, "role": "learner"})
            time.sleep(2)
            return
        code = 200
        if learner:
            view = json.loads(body["messages"][0]["content"])
            profile = "starter" if slot % 2 == 0 else "leaky"
            action = (
                {"defense_prompt": defense_prompt(profile, view["passphrase"])}
                if view["type"] == "register_request"
                else {"message": attack_message(view["turn"], profile, view["transcript"])}
            )
            text = json.dumps(action)
        else:
            text = asyncio.run(MockModel().complete(body["system"], body["messages"], slot=slot))
        payload = {
            "id": call_id,
            "model": body["model"],
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 9, "output_tokens": 3},
            "stop_reason": "end_turn",
        }
        if learner and flow == "sampled":
            payload["sampling_evidence"] = {
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3, 4],
                "behavior_log_probs": [-0.5, -0.6],
                "stop_reason": "end_turn",
            }
        if learner and flow == "greedy-tokens":
            payload["sampling_evidence"] = {
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3, 4],
                "behavior_log_probs": None,
                "stop_reason": "end_turn",
            }
        raw = json.dumps(payload)
        if learner and flow == "malformed":
            raw = "{actual malformed native response"
        if learner and flow == "throttled":
            code, raw = 429, "actual throttled native response"
        encoded = raw.encode()
        archives.append(
            {
                "call_id": call_id,
                "slot": slot,
                "request": body,
                "response": raw,
                "role": "learner" if learner else "environment",
            }
        )
        self.send_response(code)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("X-Softmax-Llm-Call-Id", call_id)
        if learner:
            self.send_header("X-Coworld-Checkpoint-Sha256", "a" * 64)
            self.send_header("X-Coworld-Tokenizer-Sha256", "b" * 64)
            self.send_header("X-Coworld-Chat-Template-Sha256", "c" * 64)
        self.send_header("request-id", "provider-" + call_id)
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args):
        pass


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_port(process, number):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("owned game exited before readiness")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", number)) == 0:
                return
        time.sleep(0.05)
    raise TimeoutError("owned game readiness")


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
server.daemon_threads = True
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
manifest = json.loads((root / "coworld_manifest_template.json").read_text())
cohorts = [("accepted", v["id"]) for v in manifest["variants"]] + [
    (f, "duel-2") for f in ("sampled", "greedy-tokens", "malformed", "throttled", "silence")
]
try:
    for flow, variant in cohorts:
        case = destination / (variant + "-" + flow)
        case.mkdir(mode=0o700)
        archives = []
        config = next(v["game_config"] for v in manifest["variants"] if v["id"] == variant)
        config = {
            **config,
            "seed": 17,
            "model_provider": config["model_provider"],
            "player_connect_timeout_seconds": 5,
            "action_timeout_seconds": 0.1 if flow == "silence" else 1,
            "model_timeout_seconds": 1,
            "minimum_episode_seconds": 0,
            "tokens": ["owned-native-" + str(i) for i in range(len(config["players"]))],
        }
        (case / "config.json").write_text(json.dumps(config))
        (case / "config.json").chmod(0o600)
        number = port()
        environment = {
            **os.environ,
            "COGAME_CONFIG_URI": str(case / "config.json"),
            "COGAME_PORT": str(number),
            "COGAME_HOST": "127.0.0.1",
            "COGAME_RESULTS_URI": str(case / "results.json"),
            "COGAME_SAVE_REPLAY_URI": str(case / "replay.json"),
            "COGAME_SAVE_TRAJECTORY_URI": str(case / "trajectory.jsonl"),
            "COWORLD_EPISODE_ID": "open-sesame-" + variant + "-" + flow + "-" + source[:7],
            "COWORLD_GAME_VERSION": "source-" + source,
            "COWORLD_SOURCE_REVISION": source,
            "COWORLD_LLM_ENDPOINT": "http://127.0.0.1:" + str(server.server_port),
            "COWORLD_LLM_MODEL": "checkpoint/learner-fixture",
            "COWORLD_LLM_TEMPERATURE": "1" if flow == "sampled" else "0",
            "COWORLD_LLM_TOP_P": "1",
            "OPEN_SESAME_LEARNER_TIMEOUT_SECONDS": "2" if flow == "silence" else "0.8",
            "OPEN_SESAME_PROFILE": "native",
        }
        processes = []
        handles = []
        try:
            handle = (case / "game.log").open("w")
            (case / "game.log").chmod(0o600)
            handles.append(handle)
            game = subprocess.Popen(
                [sys.executable, "-m", "opensesame.server"],
                cwd=root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append(game)
            wait_port(game, number)
            for slot, token in enumerate(config["tokens"]):
                handle = (case / f"player-{slot}.log").open("w")
                (case / f"player-{slot}.log").chmod(0o600)
                handles.append(handle)
                player_env = {
                    **environment,
                    "COWORLD_PLAYER_WS_URL": f"ws://127.0.0.1:{number}/player?slot={slot}&token={token}",
                }
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-m", "opensesame.player"],
                        cwd=root,
                        env=player_env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                )
            game.wait(timeout=60)
            assert game.returncode == 0, (case, game.returncode)
            path = case / "trajectory.jsonl"
            events = [json.loads(line) for line in path.read_text().splitlines()]
            assert events[-1]["status"] == ("truncated" if flow == "silence" else "completed"), (case, events[-1])
            attempts = [a for e in events[:-1] for a in e["attempts"]]
            by_id = {a["call_id"]: a for a in archives}
            for a in attempts:
                if a["platform_call_id"] is None:
                    assert flow == "silence" and a["raw_response"] is None and a["latency_ms"] is None
                    assert a["request"] is not None and not a["accepted"]
                    continue
                actual = by_id[a["platform_call_id"]]
                assert (
                    actual["role"] == "learner"
                    and actual["request"] == a["request"]
                    and actual["response"] == a["raw_response"]
                )
            environment_calls = events[-1]["outcome"]["environment_generations"]
            assert all(by_id[a["platform_call_id"]]["role"] == "environment" for a in environment_calls)
            assert all(
                a["request"]["model"] == config["model_id"] and a["request"]["temperature"] == 0
                for a in environment_calls
            )
            assert all(
                a["request"]["model"] == "checkpoint/learner-fixture" for a in attempts if a["origin"] == "model"
            )
            assert path.stat().st_mode & 0o777 == 0o600
            private = json.dumps(events)
            public = (case / "replay.json").read_text()
            assert "checkpoint/learner-fixture" not in public
            if flow in ("accepted", "sampled", "greedy-tokens"):
                assert all(e["action_status"] == "accepted" for e in events[:-1])
                assert all(
                    next(a for a in e["attempts"] if a["attempt_id"] == e["selected_attempt_id"])["parsed_action"]
                    == e["executed_action"]
                    for e in events[:-1]
                )
            (case / "native-call-archives.json").write_text(json.dumps(archives, indent=2))
            (case / "native-call-archives.json").chmod(0o600)
            print(
                json.dumps(
                    {
                        "case": case.name,
                        "status": events[-1]["status"],
                        "decisions": len(events) - 1,
                        "native_calls": len(archives),
                        "environment_calls": len(environment_calls),
                    }
                ),
                flush=True,
            )
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                process.wait(timeout=10)
            for handle in handles:
                handle.close()
finally:
    server.shutdown()
    server.server_close()
    thread.join()
