"""One interface for the agent jobs: JSON-schema chat (optionally with an image) and image embedding.

Providers:
  sie       Superlinked SIE Serverless via sie-sdk (open models; model ids in backend/models.yaml)
  anthropic Claude via the official anthropic SDK (frontier comparison; needs ANTHROPIC_API_KEY or FRONTIER_API_KEY)
  openai_compatible  any OpenAI-style /v1/chat/completions endpoint (FRONTIER_BASE_URL, FRONTIER_MODEL)
Every call returns measured tokens and latency so cost is computed from usage x published price.
"""
import base64
import json
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def models():
    return yaml.safe_load((ROOT / "models.yaml").read_text())


@lru_cache(maxsize=1)
def prices():
    return yaml.safe_load((ROOT / "config/prices.yaml").read_text())


@dataclass
class CallResult:
    ok: bool
    data: object = None
    text: str = ""
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    images: int = 0
    latency_s: float = 0.0
    error: str = ""
    attempts: int = 1
    extra: dict = field(default_factory=dict)

    def cost_usd(self):
        return call_cost(self.provider, self.model, self.input_tokens, self.output_tokens, self.images)


def call_cost(provider, model, tin, tout, images=0):
    if provider == "openai_compatible":
        pin = float(os.environ.get("FRONTIER_PRICE_IN", "nan")); pout = float(os.environ.get("FRONTIER_PRICE_OUT", "nan"))
        return (tin * pin + tout * pout) / 1e6
    p = prices().get(provider, {}).get(model)
    if not p:
        return float("nan")
    return (tin * p.get("input_per_mtok", 0) + tout * p.get("output_per_mtok", 0)) / 1e6 + images * p.get("per_1000_images", 0) / 1000


def sie_configured():
    return bool(os.environ.get("SIE_API_KEY"))


def frontier_provider():
    if os.environ.get("FRONTIER_PROVIDER"):
        return os.environ["FRONTIER_PROVIDER"]
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("FRONTIER_API_KEY"):
        return "anthropic"
    return None


def frontier_model():
    return os.environ.get("FRONTIER_MODEL") or ("claude-opus-5-5" if frontier_provider() == "anthropic" else "")


@lru_cache(maxsize=1)
def sie_client():
    from sie_sdk import SIEClient
    return SIEClient(base_url=os.environ.get("SIE_BASE_URL", "https://api.superlinked.com"), api_key=os.environ["SIE_API_KEY"])


@lru_cache(maxsize=1)
def anthropic_client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("FRONTIER_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def _usage(obj, *names):
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if v is not None:
            return int(v)
    return 0


def chat_json(provider, model, system, user_text, schema, schema_name, image_png=None, max_tokens=700, validate=None):
    """Ask for JSON matching `schema`. `validate(data)` may raise; on failure retry once, then fail loudly."""
    last_err, total_in, total_out, t0 = "", 0, 0, time.time()
    for attempt in (1, 2):
        try:
            text, tin, tout = _chat_once(provider, model, system, user_text, schema, schema_name, image_png, max_tokens,
                                         None if attempt == 1 else last_err)
            total_in += tin; total_out += tout
            data = json.loads(_strip_fences(text))
            if validate is not None:
                data = validate(data)
            return CallResult(True, data, text, provider, model, total_in, total_out, int(image_png is not None) * attempt,
                              time.time() - t0, attempts=attempt)
        except Exception as e:  # invalid JSON, schema violation, or API error
            last_err = f"{type(e).__name__}: {e}"[:400]
    return CallResult(False, None, "", provider, model, total_in, total_out, int(image_png is not None) * 2,
                      time.time() - t0, error=last_err, attempts=2)


def _strip_fences(t):
    t = t.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.rsplit("```", 1)[0]
    return t.strip()


def _chat_once(provider, model, system, user_text, schema, schema_name, image_png, max_tokens, retry_note):
    if retry_note:
        user_text = f"{user_text}\n\nYour previous answer was rejected ({retry_note}). Reply with JSON that matches the schema exactly."
    if provider == "sie":
        content = [{"type": "text", "text": user_text}]
        if image_png is not None:
            content.insert(0, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_png).decode()}})
        r = sie_client().chat_completions(
            model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            max_completion_tokens=max_tokens, temperature=0.0,
            response_format={"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}})
        msg = r["choices"][0]["message"]["content"]
        u = r.get("usage", {}) or {}
        return msg, _usage(u, "prompt_tokens", "input_tokens"), _usage(u, "completion_tokens", "output_tokens")
    if provider == "anthropic":
        content = [{"type": "text", "text": user_text}]
        if image_png is not None:
            content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                           "data": base64.standard_b64encode(image_png).decode()}})
        r = anthropic_client().messages.create(
            model=model, max_tokens=4000, system=system, messages=[{"role": "user", "content": content}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
        if r.stop_reason == "refusal":
            raise RuntimeError("model refused")
        text = next(b.text for b in r.content if b.type == "text")
        return text, r.usage.input_tokens, r.usage.output_tokens
    if provider == "openai_compatible":
        import httpx
        content = [{"type": "text", "text": user_text}]
        if image_png is not None:
            content.insert(0, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_png).decode()}})
        r = httpx.post(os.environ["FRONTIER_BASE_URL"].rstrip("/") + "/chat/completions", timeout=120,
                       headers={"Authorization": f"Bearer {os.environ['FRONTIER_API_KEY']}"},
                       json={"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                             "max_tokens": max_tokens, "temperature": 0,
                             "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}}})
        r.raise_for_status(); j = r.json()
        return j["choices"][0]["message"]["content"], _usage(j.get("usage", {}), "prompt_tokens"), _usage(j.get("usage", {}), "completion_tokens")
    raise ValueError(f"unknown provider {provider}")


def embed_images(pngs, model=None):
    """SIE image embeddings (unit-normalised float32 vectors)."""
    import numpy as np
    from PIL import Image
    import io
    from sie_sdk.types import Item
    model = model or models()["place_embedding"]
    t0 = time.time()
    items = [Item(images=[{"data": Image.open(io.BytesIO(p)).convert("RGB"), "format": "png"}]) for p in pngs]
    res = sie_client().encode(model, items)
    res = res if isinstance(res, list) else [res]
    vecs = np.stack([np.asarray(r["dense"], dtype=np.float32) for r in res])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs, CallResult(True, None, "", "sie", model, 0, 0, len(pngs), time.time() - t0)
