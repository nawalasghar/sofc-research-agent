"""Configuration and secret loading.

- ``config.yaml`` holds paths and tuning knobs and is committed.
- ``.env`` holds API keys and is gitignored.

Nothing in this module talks to the network or imports heavy dependencies, so the
pure-Python tools stay cheap to import.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.yaml"
ENV_FILE = ROOT / ".env"


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing, unreadable, or lacks a required key."""


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        raise ConfigError(f"config.yaml not found at {CONFIG_FILE}")
    import yaml  # deferred: only needed once, and only when config is read

    with CONFIG_FILE.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{CONFIG_FILE} must parse to a mapping, got {type(data).__name__}")
    return data


def get(dotted_key: str, default: Any = None) -> Any:
    """Read a nested config value: ``get("literature.default_k", 5)``."""
    node: Any = load_config()
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def require(dotted_key: str) -> Any:
    value = get(dotted_key, None)
    if value is None:
        raise ConfigError(f"missing required key '{dotted_key}' in {CONFIG_FILE.name}")
    return value


@lru_cache(maxsize=1)
def load_dotenv() -> dict[str, str]:
    """Parse ``.env`` into a dict. Missing file is not an error."""
    values: dict[str, str] = {}
    if not ENV_FILE.exists():
        return values
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, val = line.partition("=")
        if not sep:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        values[key.strip()] = val
    return values


def secret(name: str, default: str | None = None) -> str | None:
    """Look up a secret in ``.env`` first, then the process environment."""
    value = load_dotenv().get(name) or os.environ.get(name)
    return value if value else default


def rag_root() -> Path:
    return Path(str(require("paths.rag_root"))).expanduser()


def index_files() -> tuple[Path, Path]:
    """(faiss index, chunk metadata jsonl) inside the literature RAG project."""
    root = rag_root() / "index"
    return root / "chunks.faiss", root / "chunks.jsonl"


def dataset_csv() -> Path:
    return Path(str(require("paths.dataset_csv"))).expanduser()
