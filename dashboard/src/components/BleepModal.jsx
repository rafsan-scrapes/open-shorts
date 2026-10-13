import React, { useEffect, useState } from 'react';
import { Loader2, VolumeX, Plus, Trash2 } from 'lucide-react';
import { apiFetch } from '../lib/api';
import Modal from './ui/Modal';

// Client-side preview wordlist only — the server (profanity.py) is the source
// of truth on submit. Kept small and in sync by intent, not by import.
const PREVIEW_WORDS = new Set([
    'fuck', 'fucking', 'fucker', 'fucked', 'fucks',
    'shit', 'bitch', 'bitches', 'asshole', 'dick',
    'motherfucker', 'bullshit',
]);

const cleanWord = (t) => (t || '').toLowerCase().replace(/^[^a-z0-9']+|[^a-z0-9']+$/g, '');

const fmt = (ms) => `${(ms / 1000).toFixed(2)}s`;

export default function BleepModal({ isOpen, onClose, onBleep, isProcessing, jobId, clipIndex }) {
    const [captions, setCaptions] = useState(null);
    const [loading, setLoading] = useState(false);
    const [loadError, setLoadError] = useState(null);
    // idx set of auto-detected hits the user kept checked.
    const [checked, setChecked] = useState(null);
    const [extraWords, setExtraWords] = useState('');
    const [manual, setManual] = useState([]);
    const [mStart, setMStart] = useState('');
    const [mEnd, setMEnd] = useState('');
    const [maskCaptions, setMaskCaptions] = useState(true);

    useEffect(() => {
        if (!isOpen || !jobId || clipIndex === undefined) return;
        setLoading(true);
        setLoadError(null);
        setCaptions(null);
        setChecked(null);
        setManual([]);
        apiFetch(`/api/clip/${jobId}/${clipIndex}/transcript`)
            .then((res) => {
                if (!res.ok) throw new Error('transcript unavailable');
                return res.json();
            })
            .then((data) => {
                const caps = data.captions || [];
                setCaptions(caps);
                const hits = caps.filter((c) => PREVIEW_WORDS.has(cleanWord(c.text)));
                setChecked(new Set(hits.map((c) => `${c.startMs}-${c.endMs}`)));
            })
            .catch(() => setLoadError('could not load transcript — you can still add manual ranges below.'))
            .finally(() => setLoading(false));
    }, [isOpen, jobId, clipIndex]);

    if (!isOpen) return null;

    const hits = (captions || []).filter((c) => PREVIEW_WORDS.has(cleanWord(c.text)));
    const key = (c) => `${c.startMs}-${c.endMs}`;
    const toggle = (c) => {
        setChecked((prev) => {
            const next = new Set(prev || []);
            if (next.has(key(c))) next.delete(key(c));
            else next.add(key(c));
            return next;
        });
    };

    const addManual = () => {
        const s = parseFloat(mStart);
        const e = parseFloat(mEnd);
        if (!Number.isFinite(s) || !Number.isFinite(e) || e <= s || s < 0) return;
        setManual((prev) => [...prev, { start: s, end: e }]);
        setMStart('');
        setMEnd('');
    };

    const selectedAuto = hits.filter((c) => (checked || new Set()).has(key(c)));
    const canSubmit = selectedAuto.length + manual.length > 0;

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="md" eyebrow="EDITOR · AUDIO" title="bleep profanity">
            <div className="space-y-5">
                <p className="text-xs text-muted lowercase leading-relaxed">
                    detected bad words are muted to silence (with caption text masked as f***).
                    uncheck any word to keep it, or add manual ranges below.
                </p>

                {loading && (
                    <div className="flex items-center gap-2 text-sm text-muted">
                        <Loader2 size={16} className="animate-spin" /> scanning transcript…
                    </div>
                )}
                {loadError && <p className="text-xs text-warn lowercase">{loadError}</p>}

                {!loading && hits.length === 0 && !loadError && (
                    <p className="text-xs text-muted lowercase">no profanity detected — add a manual range below if needed.</p>
                )}

                {hits.length > 0 && (
                    <div>
                        <p className="eyebrow mb-2">detected ({selectedAuto.length}/{hits.length} muted)</p>
                        <div className="space-y-1.5 max-h-48 overflow-y-auto custom-scrollbar pr-1">
                            {hits.map((c) => (
                                <label
                                    key={key(c)}
                                    className="flex items-center gap-2.5 px-3 py-2 rounded-input border border-rule hover:border-rule2 cursor-pointer transition-colors"
                                >
                                    <input
                                        type="checkbox"
                                        checked={(checked || new Set()).has(key(c))}
                                        onChange={() => toggle(c)}
                                        className="w-4 h-4 accent-[var(--color-accent)] shrink-0"
                                    />
                                    <span className="text-sm text-ink font-medium">{c.text}</span>
                                    <span className="readout ml-auto">{fmt(c.startMs)} – {fmt(c.endMs)}</span>
                                </label>
                            ))}
                        </div>
                    </div>
                )}

                <div>
                    <p className="eyebrow mb-2">extra words to bleep</p>
                    <input
                        type="text"
                        value={extraWords}
                        onChange={(e) => setExtraWords(e.target.value)}
                        className="input-field"
                        placeholder="e.g. damn, crap (comma separated)…"
                    />
                </div>

                <div>
                    <p className="eyebrow mb-2">manual ranges (seconds){manual.length > 0 && ` · ${manual.length} added`}</p>
                    {manual.map((m, i) => (
                        <div key={i} className="flex items-center gap-2 px-3 py-1.5 rounded-input border border-rule mb-1.5">
                            <VolumeX size={14} className="text-muted shrink-0" />
                            <span className="readout">{m.start.toFixed(2)}s – {m.end.toFixed(2)}s</span>
                            <button
                                onClick={() => setManual((prev) => prev.filter((_, j) => j !== i))}
                                aria-label="remove range"
                                className="ml-auto p-1 rounded-full text-muted hover:text-danger transition-colors"
                            >
                                <Trash2 size={14} />
                            </button>
                        </div>
                    ))}
                    <div className="flex gap-2">
                        <input
                            type="number" min="0" step="0.1"
                            value={mStart} onChange={(e) => setMStart(e.target.value)}
                            className="input-field" placeholder="start"
                        />
                        <input
                            type="number" min="0" step="0.1"
                            value={mEnd} onChange={(e) => setMEnd(e.target.value)}
                            className="input-field" placeholder="end"
                        />
                        <button onClick={addManual} className="btn-ghost shrink-0" aria-label="add range">
                            <Plus size={16} />
                        </button>
                    </div>
                </div>

                <label className="flex items-center justify-between cursor-pointer p-3 bg-paper rounded-input border border-rule">
                    <span className="text-sm text-ink2 lowercase">mask captions too (f***)</span>
                    <input
                        type="checkbox"
                        checked={maskCaptions}
                        onChange={(e) => setMaskCaptions(e.target.checked)}
                        className="w-4 h-4 accent-[var(--color-accent)] cursor-pointer"
                    />
                </label>

                <div className="flex gap-2 shrink-0">
                    <button onClick={onClose} className="btn-ghost">cancel</button>
                    <button
                        onClick={() => onBleep({
                            intervals: [
                                ...selectedAuto.map((c) => ({ start: c.startMs / 1000, end: c.endMs / 1000 })),
                                ...manual,
                            ],
                            extra_words: extraWords.split(',').map((w) => w.trim()).filter(Boolean),
                            mask_captions: maskCaptions,
                        })}
                        disabled={isProcessing || !canSubmit}
                        className="btn-primary flex-1"
                    >
                        {isProcessing && <Loader2 size={16} className="animate-spin text-brassink" />}
                        {isProcessing ? 'muting…' : `mute ${selectedAuto.length + manual.length} word${selectedAuto.length + manual.length === 1 ? '' : 's'}`}
                    </button>
                </div>
            </div>
        </Modal>
    );
}
