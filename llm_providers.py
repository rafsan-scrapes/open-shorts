"""Optional per-job "primary AI provider" for the transcript stages.

Gemini stays the only mandatory key (frame stages + fallback). The user may
pick one of these OpenAI-compatible presets for scoring + detail; on any
failure the job continues on Gemini (sticky, per-call granularity).

Security: the client sends a provider ID, never a URL. The backend maps the
ID to a fixed base URL. A free-form base URL would be an SSRF hole toward
Docker-internal services and 169.254.169.254.
"""
from __future__ import annotations

import re

PROVIDERS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "bynara": "https://router.bynara.id/v1",
    "opencode-zen": "https://opencode.ai/zen/v1",
}

MODEL_RE = re.compile(r"^[A-Za-z0-9._:/@~+\-]{1,200}$")
KEY_MAX_LEN = 500

ZEN_404_HINT = ("This model is not served via /chat/completions on OpenCode Zen. "
                "Only models served via /chat/completions work here; models served "
                "via /responses, /messages or /models/<id> do not.")


def provider_ids() -> list:
    return sorted(PROVIDERS.keys())


def base_url_for(provider_id: str) -> str:
    """Fixed base URL for a provider ID, or raises ValueError for unknown IDs."""
    pid = (provider_id or "").strip().lower()
    try:
        return PROVIDERS[pid]
    except KeyError:
        raise ValueError(f"Unknown LLM provider: {provider_id!r}")


def canonical_id(provider_id: str) -> str:
    pid = (provider_id or "").strip().lower()
    if pid not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider_id!r}")
    return pid


def validate_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        raise ValueError("Model name is empty")
    if len(m) > 200 or not MODEL_RE.match(m):
        raise ValueError(
            "Invalid model name (max 200 chars, charset [A-Za-z0-9._:/@~+-])")
    return m


def validate_key(key: str) -> str:
    k = (key or "").strip()
    if not k:
        raise ValueError("API key is empty")
    if len(k) > KEY_MAX_LEN:
        raise ValueError("API key too long")
    if "\n" in k or "\r" in k:
        raise ValueError("API key must not contain newlines")
    return k


def extra_headers(provider_id: str) -> dict:
    if (provider_id or "").strip().lower() == "openrouter":
        return {"HTTP-Referer": "https://openshorts.app",
                "X-OpenRouter-Title": "OpenShorts"}
    return {}


_KEY_LIKE = re.compile(r"(sk-[A-Za-z0-9\-_]{4,}|Bearer\s+[A-Za-z0-9\-_.~+/=]{4,})")


def scrub(text: str) -> str:
    """Remove anything that looks like a key; truncate to ~300 chars."""
    t = str(text or "")
    t = _KEY_LIKE.sub("[redacted-key]", t)
    if len(t) > 300:
        t = t[:300] + "…"
    return t


def classify(status: int | None, body: str, exc_name: str = "") -> dict:
    """One unit-tested classifier for third-party provider failures.

    Returns {"kind", "retryable", "message"} where kind is one of:
    auth | quota | model_not_found | context_length | format_rejection |
    transient | bad_response | refusal. ``message`` is a short user-facing
    fragment with no secrets (callers scrub again before logging).
    ``status`` is None for transport-level errors (timeout/DNS/reset).
    """
    b = (body or "").lower()
    exc = (exc_name or "").lower()

    def _m(msg):
        return {"message": msg}

    # Transport errors: timeouts, DNS, connection reset -> transient.
    if status is None:
        if any(t in exc or t in b for t in (
                "moderat", "safety", "policy", "refus", "blocked")):
            return {"kind": "refusal", "retryable": False,
                    **_m("the provider refused the request on safety/policy grounds")}
        return {"kind": "transient", "retryable": True,
                **_m(f"network error ({exc or 'connection failed'})")}
    s = int(status)

    if s in (401, 403):
        return {"kind": "auth", "retryable": False,
                **_m("the provider rejected the API key (401/403)")}
    if s == 402 or "insufficient" in b and "credit" in b \
            or "quota" in b and "exhaust" in b or "out of credit" in b \
            or "insufficient_quota" in b or "credits" in b and "exhaust" in b:
        return {"kind": "quota", "retryable": False,
                **_m("the provider reports insufficient credits/quota")}
    if s == 404 or "model not found" in b or "no endpoints found" in b \
            or "model_not_found" in b or "unknown model" in b:
        return {"kind": "model_not_found", "retryable": False,
                **_m("the model name isn't available on that provider")}
    if s == 400 and any(t in b for t in (
            "context", "too long", "max tokens", "token limit",
            "context_length", "reduce the length")):
        return {"kind": "context_length", "retryable": False,
                **_m("the prompt exceeded the model's context length")}
    if s in (400, 422) and any(t in b for t in (
            "response_format", "json_schema", "json_object", "format")):
        return {"kind": "format_rejection", "retryable": False,
                **_m("the provider rejected the response_format")}
    if any(t in b for t in ("moderat", "safety", "policy violation",
                            "content filter", "refus")):
        return {"kind": "refusal", "retryable": False,
                **_m("the provider refused the request on safety/policy grounds")}
    if s in (408, 429, 500, 502, 503, 504):
        return {"kind": "transient", "retryable": True,
                **_m(f"transient provider error (HTTP {s})")}
    if s == 400:
        # Generic 400 (e.g. wrong endpoint flavour on Zen): no retry; callers
        # append the Zen hint for that provider.
        return {"kind": "bad_response", "retryable": False,
                **_m(f"provider error (HTTP {s})")}
    return {"kind": "bad_response" if s >= 400 else "transient",
            "retryable": s in (408, 429, 500, 502, 503, 504),
            **_m(f"provider error (HTTP {s})")}
