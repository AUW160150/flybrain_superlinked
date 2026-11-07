"""Flybrain Swarm backend. Loads fly_superlinked/.env (if present) into the environment without overriding it."""
import os
from pathlib import Path

_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _v = _v.strip().strip('"').strip("'")
            if _v and _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v
