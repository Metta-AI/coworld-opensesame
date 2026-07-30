from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

HTTP_USER_AGENT = "coworld-opensesame/0.1"


def read_data(uri: str) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        request = Request(uri, headers={"User-Agent": HTTP_USER_AGENT})
        with urlopen(request, timeout=30) as response:
            return response.read()
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).read_bytes()
    if parsed.scheme == "":
        return Path(uri).read_bytes()
    raise ValueError(f"Unsupported URI for reading: {uri}")


def artifact_method(env_var: str) -> Literal["POST", "PUT"]:
    method = os.environ.get(env_var, "PUT").upper()
    if method not in {"POST", "PUT"}:
        raise ValueError(f"{env_var} must be POST or PUT")
    return cast(Literal["POST", "PUT"], method)


def write_data(
    uri: str,
    data: bytes | str,
    *,
    content_type: str,
    http_method: Literal["POST", "PUT"] = "PUT",
) -> None:
    payload = data.encode() if isinstance(data, str) else data
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        request = Request(uri, data=payload, method=http_method)
        request.add_header("Content-Type", content_type)
        request.add_header("User-Agent", HTTP_USER_AGENT)
        with urlopen(request, timeout=60):
            return
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif parsed.scheme == "":
        path = Path(uri)
    else:
        raise ValueError(f"Unsupported URI for writing: {uri}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
