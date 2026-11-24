"""Explicit local credentials/settings for the goal generation CLI."""

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_API_CONFIG = Path(__file__).resolve().parents[1] / "config" / "goal_generator.local.json"


def load_api_settings(path=None):
    """Read explicit settings, or the repository-local file when available."""
    if path is None:
        path = DEFAULT_API_CONFIG
        if not path.is_file():
            return {}
    try:
        settings = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        raise ValueError("Goal config must be valid JSON") from None
    if not isinstance(settings, dict) or set(settings) - {"reasoning", "image"}:
        raise ValueError("Goal config supports only reasoning and image sections")
    for name, section in settings.items():
        numeric = ({"temperature", "timeout", "stream_total_timeout", "max_retries",
                    "max_output_tokens"} if name == "reasoning" else {"max_retries"})
        extra = {"reasoning_effort", "stream"} if name == "reasoning" else set()
        allowed = {"url", "model", "api_token"} | numeric | extra
        if not isinstance(section, dict) or set(section) - allowed:
            raise ValueError(f"{name} supports only {', '.join(sorted(allowed))}")
        for key, value in section.items():
            if key == "stream":
                if not isinstance(value, bool):
                    raise ValueError("reasoning.stream must be boolean")
                continue
            if key == "reasoning_effort":
                if not isinstance(value, str) or value not in {"low", "high", "max"}:
                    raise ValueError("reasoning.reasoning_effort must be low, high or max")
                continue
            if key in numeric:
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or (key in {"timeout", "stream_total_timeout", "max_output_tokens"} and value <= 0)
                        or (key == "max_retries" and (not isinstance(value, int) or not 0 <= value <= 5))
                        or (key == "max_output_tokens" and not isinstance(value, int))
                        or (key == "temperature" and not 0 <= value <= 2)):
                    raise ValueError(f"Invalid {name}.{key}")
                continue
            if not isinstance(value, str) or (key != "api_token" and not value.strip()):
                raise ValueError(f"{name}.{key} must be a string (nonempty for url/model)")
        if "url" in section:
            try:
                url = urlsplit(section["url"])
                valid = (url.scheme == "https" and url.hostname and not url.username
                         and not url.password and not url.query and not url.fragment)
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"{name}.url must be HTTPS without credentials/query/fragment")
    return settings


def dynamic_reasoning_config(settings, audit_dir):
    """Forward supported runtime options, with streaming for live planning."""
    from dataclasses import replace
    from high_level_planner.reasoning_graph.language_models.openai_compatible import APIConfig

    section = settings.get("reasoning", {})
    base = APIConfig.from_env(model=section.get("model"), base_url=section.get("url"))
    defaults = dict(timeout=120.0, stream=True, stream_total_timeout=600.0,
                    max_retries=2, max_output_tokens=8192,
                    reasoning_effort="low" if base.model.lower() == "kimi-k3" else None)
    defaults.update({key: section[key] for key in (
        "timeout", "temperature", "stream", "stream_total_timeout", "max_retries",
        "max_output_tokens", "reasoning_effort", "api_token") if key in section})
    return replace(base, **defaults, audit_dir=str(audit_dir))


@contextmanager
def configured_tokens(settings, image_config):
    """Scope file credentials to one invocation and restore the environment."""
    overrides = {}
    for section, variable in (("reasoning", "TABLESIM_REASONING_API_TOKEN"),
                              ("image", image_config.token_env)):
        token = settings.get(section, {}).get("api_token", "").strip()
        if token:
            if variable in overrides and overrides[variable] != token:
                raise ValueError("Reasoning and image credentials need separate token_env names")
            overrides[variable] = token
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
