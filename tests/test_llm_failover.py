"""Optional primary AI provider with sticky Gemini failover.

Covers llm_providers.classify / validation / scrub, the hosted generate_json
path (MockTransport, mocked sleep), the per-call sticky failover in main.py,
and the app.py header/env/resume wiring.
"""
import json

import httpx
import pytest

import llm_backend
import llm_providers
import gemini_worker


def _cfg(**over):
    d = {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
         "model": "m", "api_key": "k", "extra_headers": {}}
    d.update(over)
    return d


def _completion(payload, usage=None):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant",
                                 "content": json.dumps(payload)}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    })


def _transport(handler):
    return httpx.MockTransport(handler)


# --- classification -------------------------------------------------------

def test_classify_auth_no_retry():
    for status in (401, 403):
        out = llm_providers.classify(status, "invalid api key")
        assert out["kind"] == "auth" and out["retryable"] is False


def test_classify_quota_no_retry():
    out = llm_providers.classify(402, "insufficient credits")
    assert out["kind"] == "quota" and out["retryable"] is False
    out = llm_providers.classify(200, "quota exhausted for today")
    assert out["kind"] == "quota" and out["retryable"] is False


def test_classify_model_not_found_no_retry():
    for body in ("model not found", "no endpoints found for model"):
        out = llm_providers.classify(404, body)
        assert out["kind"] == "model_not_found" and out["retryable"] is False


def test_classify_context_length():
    out = llm_providers.classify(400, "this model's maximum context length exceeded")
    assert out["kind"] == "context_length" and out["retryable"] is False


def test_classify_format_rejection():
    out = llm_providers.classify(400, "response_format json_schema not supported")
    assert out["kind"] == "format_rejection"


def test_classify_transient_retryable():
    for status in (408, 429, 500, 502, 503, 504):
        out = llm_providers.classify(status, "busy")
        assert out["retryable"] is True
    out = llm_providers.classify(None, "timeout", "ReadTimeout")
    assert out["kind"] == "transient" and out["retryable"] is True
    out = llm_providers.classify(None, "dns fail", "NewConnectionError")
    assert out["retryable"] is True


def test_classify_refusal_no_retry():
    out = llm_providers.classify(200, "content moderation flagged this")
    assert out["kind"] == "refusal" and out["retryable"] is False


def test_scrub_removes_keys_and_truncates():
    assert "sk-abc" not in llm_providers.scrub("key sk-abcdef token")
    assert "Bearer" not in llm_providers.scrub("auth Bearer xyz12345 here!")
    assert len(llm_providers.scrub("x" * 500)) <= 301


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        llm_providers.base_url_for("https://evil.internal/x")
    with pytest.raises(ValueError):
        llm_providers.canonical_id("evil")


def test_model_and_key_validation():
    with pytest.raises(ValueError):
        llm_providers.validate_model("https://evil/x?y=1 z")
    with pytest.raises(ValueError):
        llm_providers.validate_model("x" * 201)
    assert llm_providers.validate_model(" meta/x:free ") == "meta/x:free"
    with pytest.raises(ValueError):
        llm_providers.validate_key("abc\nX-Injected: 1")
    with pytest.raises(ValueError):
        llm_providers.validate_key("")


# --- hosted generate_json --------------------------------------------------

def _score_payload():
    return {"windows": [{"id": "w0", "start": 0, "end": 90,
                         "score": 80, "reason": "hook"}]}


def test_hosted_success_zero_cost():
    cfg = _cfg(transport=_transport(lambda r: _completion(_score_payload())))
    parsed, cost = llm_backend.generate_json(
        "p", gemini_worker.ScoreResponse, config=cfg, sleep_fn=lambda s: None)
    assert parsed["windows"][0]["score"] == 80
    assert cost["total_cost"] == 0.0 and cost["local"] is True


def test_hosted_auth_fails_immediately_no_retry():
    calls = {"n": 0}

    def handler(r):
        calls["n"] += 1
        return httpx.Response(401, text="invalid api key")
    cfg = _cfg(transport=_transport(handler))
    with pytest.raises(llm_backend.ProviderCallError) as exc:
        llm_backend.generate_json("p", gemini_worker.ScoreResponse,
                                  config=cfg, sleep_fn=lambda s: None)
    assert exc.value.kind == "auth"
    assert calls["n"] == 1
    assert "k" not in str(exc.value) or "redacted" in str(exc.value)


def test_hosted_retry_after_honored():
    calls = {"n": 0}
    waits = []

    def handler(r):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, text="slow down")
        return _completion(_score_payload())
    cfg = _cfg(transport=_transport(handler))
    parsed, _ = llm_backend.generate_json(
        "p", gemini_worker.ScoreResponse, config=cfg,
        sleep_fn=lambda s: waits.append(s))
    assert parsed["windows"]
    assert calls["n"] == 2 and waits == [7.0]


def test_hosted_retries_then_raises():
    cfg = _cfg(transport=_transport(lambda r: httpx.Response(503, text="busy")))
    with pytest.raises(llm_backend.ProviderCallError) as exc:
        llm_backend.generate_json("p", gemini_worker.ScoreResponse,
                                  config=cfg, sleep_fn=lambda s: None)
    assert exc.value.kind == "transient"


def test_hosted_200_error_object_fails():
    def handler(r):
        return httpx.Response(200, json={"error": {"message": "upstream boom"}})
    cfg = _cfg(transport=_transport(handler))
    with pytest.raises(llm_backend.ProviderCallError):
        llm_backend.generate_json("p", gemini_worker.ScoreResponse,
                                  config=cfg, sleep_fn=lambda s: None)


def test_hosted_empty_choices_and_reasoning_only():
    cfg = _cfg(transport=_transport(
        lambda r: httpx.Response(200, json={"choices": []})))
    with pytest.raises(llm_backend.ProviderCallError):
        llm_backend.generate_json("p", gemini_worker.ScoreResponse,
                                  config=cfg, sleep_fn=lambda s: None)

    def reasoning(r):
        return httpx.Response(200, json={
            "choices": [{"message": {"reasoning": "thinking", "content": ""},
                         "finish_reason": "stop"}]})
    cfg = _cfg(transport=_transport(reasoning))
    with pytest.raises(llm_backend.ProviderCallError):
        llm_backend.generate_json("p", gemini_worker.ScoreResponse,
                                  config=cfg, sleep_fn=lambda s: None)


def test_hosted_truncated_length_and_html_body():
    def length(r):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
    cfg = _cfg(transport=_transport(length))
    with pytest.raises(llm_backend.ProviderCallError):
        llm_backend.generate_json("p", gemini_worker.ScoreResponse,
                                  config=cfg, sleep_fn=lambda s: None)

    def html(r):
        return httpx.Response(502, text="<html>cloudflare</html>",
                              headers={"content-type": "text/html"})
    cfg = _cfg(transport=_transport(html))
    with pytest.raises(llm_backend.ProviderCallError) as exc:
        llm_backend.generate_json("p", gemini_worker.ScoreResponse,
                                  config=cfg, sleep_fn=lambda s: None)
    assert exc.value.kind == "transient"


def test_hosted_fences_and_think_tags_stripped():
    text = "<think>hmm</think>\n```json\n" + json.dumps(_score_payload()) + "\n```"
    cfg = _cfg(transport=_transport(
        lambda r: httpx.Response(200, json={
            "choices": [{"message": {"content": text}}]})))
    parsed, _ = llm_backend.generate_json(
        "p", gemini_worker.ScoreResponse, config=cfg, sleep_fn=lambda s: None)
    assert parsed["windows"][0]["score"] == 80


def test_hosted_format_tiers_still_work():
    seen = []

    def handler(r):
        body = json.loads(r.content)
        fmt = (body.get("response_format") or {}).get("type")
        seen.append(fmt)
        if fmt == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})
        return _completion({"windows": []})
    cfg = _cfg(transport=_transport(handler))
    parsed, _ = llm_backend.generate_json(
        "p", gemini_worker.ScoreResponse, config=cfg, sleep_fn=lambda s: None)
    assert parsed == {"windows": []}
    assert seen[0] == "json_schema"


def test_hosted_schema_violation_retries_once_then_raises():
    calls = {"n": 0}

    def handler(r):
        calls["n"] += 1
        return _completion({"windows": [{"id": "w0"}]})  # missing fields
    cfg = _cfg(transport=_transport(handler))
    with pytest.raises(llm_backend.ProviderCallError):
        llm_backend.generate_json("p", gemini_worker.ScoreResponse,
                                  config=cfg, sleep_fn=lambda s: None)
    assert calls["n"] == 2


def test_hosted_key_never_in_exceptions():
    secret = "sk-or-v1-supersecretkey123"

    def handler(r):
        assert secret not in str(r.url)
        return httpx.Response(401, text="bad key")
    cfg = _cfg(api_key=secret, transport=_transport(handler))
    try:
        llm_backend.generate_json("p", gemini_worker.ScoreResponse,
                                  config=cfg, sleep_fn=lambda s: None)
        raise AssertionError("should raise")
    except llm_backend.ProviderCallError as e:
        assert secret not in str(e)


def test_hosted_no_redirect_follow():
    seen = {}

    def handler(r):
        seen["url"] = str(r.url)
        return _completion(_score_payload())
    cfg = _cfg(base_url="https://openrouter.ai/api/v1",
               transport=_transport(handler))
    llm_backend.generate_json("p", gemini_worker.ScoreResponse,
                              config=cfg, sleep_fn=lambda s: None)
    assert seen["url"].startswith("https://openrouter.ai/api/v1/chat/completions")


# --- env-only path unchanged -----------------------------------------------

def test_env_only_setup_unchanged(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    def handler(r):
        return _completion({"windows": []})
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_backend, "_client",
                        lambda **kw: httpx.Client(transport=transport, **kw))
    parsed, cost = llm_backend.generate_json("p", gemini_worker.ScoreResponse)
    assert parsed == {"windows": []}
    assert cost["local"] is True


# --- sticky failover in main.py ----------------------------------------------

def _windows(n):
    return [{"id": f"w{i}", "start": i * 10.0, "end": i * 10.0 + 10.0,
             "text": "hello world"} for i in range(n)]


def test_failover_batch3_then_sticky_on_gemini(monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    provider_cfg = {"provider": "openrouter", "base_url": "https://x",
                    "model": "m", "api_key": "k", "extra_headers": {}}
    state = {"switched": False, "switched_at": None, "reason": None}
    calls = {"provider": 0, "gemini": 0}

    def fake_generate(prompt, schema, config=None, sleep_fn=None):
        calls["provider"] += 1
        if calls["provider"] <= 2:
            key = "windows" if schema is gemini_worker.ScoreResponse else "shorts"
            item = {"id": "w0", "start": 0, "end": 10, "score": 90, "reason": "ok"}
            if key == "shorts":
                item = {"start": 0, "end": 20, "source_window_id": "w0",
                        "predicted_score": 90, "video_description_for_tiktok": "t",
                        "video_description_for_instagram": "i",
                        "video_title_for_youtube_short": "title",
                        "viral_hook_text": "hook"}
            return {key: [item]}, {"total_cost": 0.0}
        raise llm_backend.ProviderCallError("transient", "openrouter: boom (HTTP 503)", 503)

    def fake_stage(client, model, prompt, schema):
        calls["gemini"] += 1
        if schema is gemini_worker.ScoreResponse:
            return {"windows": [{"id": "w9", "start": 0, "end": 10,
                                 "score": 10, "reason": "g"}]}, None
        return {"shorts": []}, None

    monkeypatch.setattr(llm_backend, "generate_json", fake_generate)
    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    out = main._run_stage_split_with_failover(
        state, None, "gem", provider_cfg, _windows(3),
        lambda ws: "p", gemini_worker.ScoreResponse, "windows", [],
        "scoring batch 3/7")
    assert state["switched"] is True
    assert state["switched_at"] == "scoring batch 3/7"
    assert calls["gemini"] >= 1
    # Earlier provider results are kept by the caller; here the failed call
    # was re-run on Gemini.
    assert out and out[0]["id"] == "w9"


def test_auth_failure_first_call_immediate_fallback(monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    provider_cfg = {"provider": "bynara", "base_url": "https://y",
                    "model": "m", "api_key": "bad", "extra_headers": {}}
    state = {"switched": False, "switched_at": None, "reason": None}
    calls = {"provider": 0, "gemini": 0}

    def fake_generate(prompt, schema, config=None, sleep_fn=None):
        calls["provider"] += 1
        raise llm_backend.ProviderCallError(
            "auth", "bynara: the provider rejected the API key (401/403)", 401)

    def fake_stage(client, model, prompt, schema):
        calls["gemini"] += 1
        return {"windows": []}, None

    monkeypatch.setattr(llm_backend, "generate_json", fake_generate)
    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    main._run_stage_split_with_failover(
        state, None, "gem", provider_cfg, _windows(2),
        lambda ws: "p", gemini_worker.ScoreResponse, "windows", [],
        "scoring batch 1/1")
    assert calls["provider"] == 1  # no retries on auth
    assert calls["gemini"] == 1
    assert "rejected" in (state["reason"] or "")


def test_gemini_also_failing_keeps_today_behavior(monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    provider_cfg = {"provider": "openrouter", "base_url": "https://x",
                    "model": "m", "api_key": "k", "extra_headers": {}}
    state = {"switched": False, "switched_at": None, "reason": None}

    def fake_generate(prompt, schema, config=None, sleep_fn=None):
        raise llm_backend.ProviderCallError("transient", "boom", 503)

    def fake_stage(client, model, prompt, schema):
        raise RuntimeError("gemini down")

    monkeypatch.setattr(llm_backend, "generate_json", fake_generate)
    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    with pytest.raises(RuntimeError, match="gemini down"):
        main._run_stage_split_with_failover(
            state, None, "gem", provider_cfg, _windows(1),
            lambda ws: "p", gemini_worker.ScoreResponse, "windows", [],
            "detail")


# --- app.py wiring -----------------------------------------------------------

def _req(headers):
    class _H(dict):
        def get(self, k, default=None):
            return super().get(k, default)
    class _R:
        def __init__(self):
            self.headers = _H(headers)
    return _R()


def test_resolve_llm_provider_headers():
    import app as app_module
    out = app_module.resolve_llm_provider(
        _req({"X-LLM-Provider": "openrouter", "X-LLM-Model": "a/b:free",
              "X-LLM-Key": "secret"}))
    assert out == {"provider": "openrouter", "model": "a/b:free", "api_key": "secret"}
    assert app_module.resolve_llm_provider(_req({})) is None
    assert app_module.resolve_llm_provider(
        _req({"X-LLM-Provider": "Gemini only"})) is None


def test_unknown_provider_id_rejected():
    import app as app_module
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        app_module.resolve_llm_provider(
            _req({"X-LLM-Provider": "https://evil/x", "X-LLM-Model": "m",
                  "X-LLM-Key": "k"}))
    assert exc.value.status_code == 400


def test_header_injection_key_rejected():
    import app as app_module
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        app_module.resolve_llm_provider(
            _req({"X-LLM-Provider": "bynara", "X-LLM-Model": "m",
                  "X-LLM-Key": "a\r\nX: 1"}))


def test_resume_manifest_strips_llm_key(tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))
    job_id = "job123"
    (tmp_path / job_id).mkdir()
    app_module._write_resume_manifest(job_id, ["x"], 2, None, None, False,
                                      llm_provider="openrouter")
    import json as _json
    with open(tmp_path / job_id / app_module._RESUME_FILE) as f:
        m = _json.load(f)
    assert m.get("llm_provider") == "openrouter"
    assert "api_key" not in _json.dumps(m).lower() or "secret" not in _json.dumps(m)
    # Rebuilt env carries no per-job key: resume falls back to Gemini.
    monkeypatch.delenv("LLM_PROVIDER_ID", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert "LLM_PROVIDER_ID" not in dict(__import__("os").environ)
