"""Text-only LLM backend for the moment picker: any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all speak
``POST {base}/chat/completions``. When ``LLM_BASE_URL`` is set (or
``LLM_PROVIDER=openai``), the transcript scoring and detail passes in
``main.get_viral_clips`` go here instead of Gemini, so a self-hosted install
can run the whole pipeline without a Google key.

What stays on Gemini, because it needs a model that can look at frames or a
video file: the layout picker (``layout_picker.py``), the on-screen content
detector (``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). Without a Gemini key those degrade the way they
already did: the first two return "none", the third fails the job with a clear
message. Nothing in this module is imported by them.

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional, Tuple, Type

import httpx
from pydantic import BaseModel

import llm_providers

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes
HOSTED_TIMEOUT = 120.0  # third-party providers: fail over fast, don't hang the job
HOSTED_MAX_ATTEMPTS = 3
RETRY_AFTER_CAP = 30.0


class ProviderCallError(RuntimeError):
    """A failed third-party provider call, safe to log (no keys)."""

    def __init__(self, kind: str, message: str, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


def provider() -> str:
    """``"openai"`` when a compatible endpoint is configured, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local", "openai-compatible"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    return "openai" if base_url() else "gemini"


def base_url() -> str:
    return (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")


def model_name() -> str:
    return (os.environ.get("LLM_MODEL") or "").strip() or DEFAULT_MODEL


def active() -> bool:
    """True when the moment picker should call the OpenAI-compatible server."""
    return provider() == "openai" and bool(base_url())


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url()}


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def hosted_timeout() -> float:
    try:
        return float(os.environ.get("LLM_HOSTED_TIMEOUT") or HOSTED_TIMEOUT)
    except ValueError:
        return HOSTED_TIMEOUT


def job_provider_config() -> Optional[dict]:
    """Per-job third-party provider snapshot from the child env, or None.

    Set per job by app.py from the X-LLM-* headers (never from the global
    env, so concurrent jobs don't interfere). Falls back to the admin
    ``LLM_BASE_URL`` setup when no per-job snapshot is present.
    """
    pid = (os.environ.get("LLM_PROVIDER_ID") or "").strip().lower()
    if pid:
        return {"provider": pid,
                "base_url": (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/"),
                "model": (os.environ.get("LLM_MODEL") or "").strip(),
                "api_key": (os.environ.get("LLM_API_KEY") or "").strip()}
    return None


def _headers(api_key: Optional[str] = None) -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (api_key if api_key is not None else os.environ.get("LLM_API_KEY") or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    kwargs.setdefault("follow_redirects", False)
    return httpx.Client(timeout=_timeout(), **kwargs)


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


def _retry_after_seconds(resp) -> Optional[float]:
    try:
        raw = resp.headers.get("retry-after")
        if raw is None:
            return None
        secs = float(str(raw).strip().split(",")[0])
        return max(0.0, min(secs, RETRY_AFTER_CAP))
    except (ValueError, TypeError, AttributeError):
        return None


def _scrub_exc(msg: str) -> str:
    return llm_providers.scrub(msg)


def _parse_hosted_body(data: dict, provider: str, url: str) -> str:
    """Extract the assistant text from a 200 body, or raise ProviderCallError."""
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        # OpenRouter reports upstream failures as HTTP 200 + {"error": ...}.
        detail = str((data.get("error") or {}).get("message") or "")[:200]
        kind = llm_providers.classify(502, detail)["kind"]
        raise ProviderCallError(
            kind, f"{provider}: upstream error: {llm_providers.scrub(detail)}", 502)
    choices = (data.get("choices") if isinstance(data, dict) else None) or []
    if not choices:
        raise ProviderCallError(
            "bad_response", f"{provider}: empty/missing choices in response", 200)
    msg = (choices[0].get("message") or {}) if isinstance(choices[0], dict) else {}
    text = msg.get("content") or ""
    if isinstance(text, list):  # some servers return content parts
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    text = (text or "").strip()
    if not text:
        # Content only in a reasoning field, or finish_reason length.
        if msg.get("reasoning"):
            raise ProviderCallError(
                "bad_response", f"{provider}: answer only in reasoning field", 200)
        if (choices[0].get("finish_reason") or "") == "length":
            raise ProviderCallError(
                "bad_response", f"{provider}: response truncated (finish_reason length)", 200)
        raise ProviderCallError(
            "bad_response", f"{provider}: empty content in response", 200)
    # Strip <think>...</think> blocks some models emit around the JSON.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    return text


def _request_once(client, url, body, headers, provider: str):
    try:
        return client.post(url, json=body, headers=headers)
    except Exception as e:
        name = type(e).__name__
        info = llm_providers.classify(None, str(e), name)
        raise ProviderCallError(
            info["kind"], f"{provider}: {info['message']}", None) from None


def _hosted_post(client, url, body, headers, provider: str,
                 sleep_fn=time.sleep) -> httpx.Response:
    """POST with retries for transient errors; honors Retry-After (cap 30s)."""
    last: Optional[ProviderCallError] = None
    for attempt in range(1, HOSTED_MAX_ATTEMPTS + 1):
        resp = _request_once(client, url, body, headers, provider)
        if resp.status_code < 400:
            return resp
        try:
            body_text = resp.text
        except Exception:
            body_text = ""
        # Non-JSON HTML gateway page -> treat like transient 5xx.
        ct = (resp.headers.get("content-type") or "").lower()
        if resp.status_code >= 500 and ("text/html" in ct or "<html" in body_text.lower()[:500]):
            info = {"kind": "transient", "retryable": True,
                    "message": f"transient provider error (HTTP {resp.status_code})"}
        else:
            info = llm_providers.classify(resp.status_code, body_text)
        safe = llm_providers.scrub(body_text)
        hint = ""
        if provider == "opencode-zen" and resp.status_code in (400, 404):
            hint = " " + llm_providers.ZEN_404_HINT
        err = ProviderCallError(
            info["kind"],
            f"{provider}: {info['message']} (HTTP {resp.status_code}: {safe}).{hint}",
            resp.status_code)
        if not info["retryable"] or attempt == HOSTED_MAX_ATTEMPTS:
            raise err
        last = err
        wait = _retry_after_seconds(resp)
        if wait is None:
            wait = min(2 ** (attempt - 1) * 2.0, RETRY_AFTER_CAP)
        print(f"⚠️ {provider} transient error (attempt {attempt}/{HOSTED_MAX_ATTEMPTS}), "
              f"retrying in {wait:g}s: {llm_providers.scrub(body_text)[:150]}")
        sleep_fn(wait)
    raise last  # pragma: no cover


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  config: Optional[dict] = None,
                  sleep_fn=time.sleep,
                  ) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.

    ``config`` (optional) overrides the env: ``{"base_url", "model",
    "api_key", "extra_headers", "provider", "timeout"}``. The env-only path
    (``config=None``) behaves exactly as before for existing setups and tests.
    """
    import gemini_worker  # local import: keeps this module free of the google SDK

    if config is None:
        url = f"{base_url()}/chat/completions"
        model = model or model_name()
        messages = [
            {"role": "system", "content": "You answer with a single JSON object and nothing else."},
            {"role": "user", "content": prompt},
        ]
        last_rejection: Optional[str] = None
        with _client() as client:
            for fmt in _response_formats(schema):
                body = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
                if fmt is not None:
                    body["response_format"] = fmt
                resp = client.post(url, json=body, headers=_headers())
                if fmt is not None and _is_format_rejection(resp):
                    # The server does not know this response_format flavour; the
                    # next loop iteration asks for a looser one.
                    last_rejection = resp.text[:200]
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
                data = resp.json()
                break
            else:
                raise RuntimeError(
                    f"LLM server rejected every response_format variant: {last_rejection}")

        choices = data.get("choices") or []
        text = ""
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""
            if isinstance(text, list):  # some servers return content parts
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        parsed = gemini_worker._parse_json_response_text(text)
        # Validate against the same schema Gemini enforces server-side, so a
        # local model that drops a field fails here with a readable error
        # (retried by the caller) instead of deep inside the clip pipeline.
        validated = schema.model_validate(parsed).model_dump()

        usage = data.get("usage") or {}
        cost = {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "thinking_tokens": 0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "model": model,
            "price_estimated": False,
            "local": True,
        }
        return validated, cost

    # --- Hosted per-job provider path (OpenRouter / Bynara / OpenCode Zen) ---
    provider = str(config.get("provider") or "provider")
    url = f"{str(config['base_url']).rstrip('/')}/chat/completions"
    model = config.get("model") or model or model_name()
    timeout = float(config.get("timeout") or hosted_timeout())
    headers = dict(_headers(config.get("api_key")))
    headers.update(config.get("extra_headers") or {})
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    last_rejection: Optional[str] = None
    schema_retried = False
    with httpx.Client(timeout=timeout, follow_redirects=False,
                      transport=config.get("transport")) as client:
        for fmt in _response_formats(schema):
            body = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
            if fmt is not None:
                body["response_format"] = fmt
            resp = _hosted_post(client, url, body, headers, provider, sleep_fn)
            if fmt is not None and _is_format_rejection(resp):
                last_rejection = llm_providers.scrub(resp.text)
                continue
            try:
                data = resp.json()
            except Exception:
                # Non-JSON body (HTML gateway): one retry as transient.
                if not schema_retried:
                    schema_retried = True
                    sleep_fn(2.0)
                    resp = _hosted_post(client, url, body, headers, provider, sleep_fn)
                    try:
                        data = resp.json()
                    except Exception:
                        raise ProviderCallError(
                            "transient", f"{provider}: non-JSON response body", 502)
                else:
                    raise ProviderCallError(
                        "transient", f"{provider}: non-JSON response body", 502)
            text = _parse_hosted_body(data, provider, url)
            try:
                parsed = gemini_worker._parse_json_response_text(text)
                validated = schema.model_validate(parsed).model_dump()
            except Exception:
                if not schema_retried:
                    # Schema-validation failure: retry once, then fall back.
                    schema_retried = True
                    sleep_fn(1.0)
                    continue
                raise ProviderCallError(
                    "bad_response",
                    f"{provider}: answer did not match the expected schema", 200)
            usage = data.get("usage") or {}
            try:
                inp = int(usage.get("prompt_tokens") or 0)
            except (TypeError, ValueError):
                inp = 0
            try:
                outp = int(usage.get("completion_tokens") or 0)
            except (TypeError, ValueError):
                outp = 0
            cost = {
                "input_tokens": inp,
                "output_tokens": outp,
                "thinking_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
                "model": model,
                "price_estimated": False,
                "local": True,
            }
            return validated, cost
    raise ProviderCallError(
        "bad_response",
        f"{provider}: rejected every response_format variant: "
        f"{llm_providers.scrub(last_rejection or '')}", 400)
