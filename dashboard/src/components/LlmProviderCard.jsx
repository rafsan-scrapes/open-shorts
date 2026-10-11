import React, { useState } from 'react';
import { Bot, Check, Loader2 } from 'lucide-react';
import { apiFetch } from '../lib/api';

const PROVIDERS = [
  { id: 'gemini', label: 'Gemini only (default)' },
  { id: 'openrouter', label: 'OpenRouter', placeholder: 'e.g. meta-llama/llama-3.3-70b-instruct:free', keyPlaceholder: 'sk-or-...' },
  { id: 'bynara', label: 'Bynara', placeholder: 'e.g. deepseek-v4-flash', keyPlaceholder: 'byn_...' },
  { id: 'opencode-zen', label: 'OpenCode Zen', placeholder: 'e.g. big-pickle', keyPlaceholder: 'zen_...' },
];

export default function LlmProviderCard({ provider, setProvider, providerKey, setProviderKey, providerModel, setProviderModel }) {
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);

  const testConnection = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await apiFetch('/api/llm-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, model: providerModel.trim(), api_key: providerKey }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.ok) {
        setTestResult({ ok: true, msg: `OK · ${data.latency_ms} ms` });
      } else {
        const detail = data?.detail;
        const msg = typeof detail === 'string' ? detail : (detail?.message || `Failed (${res.status})`);
        setTestResult({ ok: false, msg: String(msg).slice(0, 300) });
      }
    } catch (e) {
      setTestResult({ ok: false, msg: 'timeout / network error' });
    } finally {
      setTesting(false);
    }
  };

  const meta = PROVIDERS.find((p) => p.id === provider) || PROVIDERS[0];
  const isGemini = provider === 'gemini';

  return (
    <div className="card p-4 sm:p-6 mt-8">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
            <Bot size={16} className="text-brass" />
          </div>
          <h2 className="text-base font-medium text-ink lowercase">AI provider (optional)</h2>
        </div>
        <span className="readout">Optional</span>
      </div>
      <p className="text-xs text-muted mb-4 leading-relaxed">
        Gemini stays required (frame analysis + automatic fallback). An optional third-party
        provider can pick the transcript moments instead. Only the transcript text — not the
        video — is sent to the chosen provider, and some free models/providers may log or
        train on prompts.
      </p>
      <label className="block text-sm text-muted mb-1">Provider</label>
      <select value={provider} onChange={(e) => { setProvider(e.target.value); setTestResult(null); }} className="input-field mb-3">
        {PROVIDERS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
      </select>
      {!isGemini && (
        <>
          <label className="block text-sm text-muted mb-1">API key</label>
          <input type="password" value={providerKey} onChange={(e) => setProviderKey(e.target.value)}
            className="input-field mb-3 font-mono" placeholder={meta.keyPlaceholder || 'API key'} />
          <label className="block text-sm text-muted mb-1">Model name</label>
          <input value={providerModel} onChange={(e) => setProviderModel(e.target.value)}
            className="input-field mb-1 font-mono" placeholder={meta.placeholder || 'model name'} />
          {provider === 'opencode-zen' && (
            <p className="text-xs text-muted mb-2 leading-relaxed">
              Only models served via /chat/completions work here; models served via /responses,
              /messages or /models/&lt;id&gt; do not.
            </p>
          )}
          <button onClick={testConnection} disabled={testing || !providerKey.trim() || !providerModel.trim()}
            className="btn-quiet py-2 px-4 text-sm mt-2 disabled:opacity-45">
            {testing ? <><Loader2 size={14} className="animate-spin" /> testing…</> : 'Test connection'}
          </button>
          {testResult && (
            <p className={`mt-2 text-xs ${testResult.ok ? 'text-ok' : 'text-danger'}`}>
              {testResult.ok ? <><Check size={12} className="inline" /> {testResult.msg}</> : testResult.msg}
            </p>
          )}
        </>
      )}
    </div>
  );
}
