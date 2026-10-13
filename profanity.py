"""Profanity bleep helpers: detection, caption masking, mute filter.

Standard-library only (like recut.py) so it imports in the thin CI env.
"""

import os
import re

# Small built-in English wordlist. Overridable/extended via env
# PROFANITY_WORDLIST (comma-separated) and per-request extra_words.
DEFAULT_WORDLIST = frozenset({
    "fuck", "fucking", "fucker", "fucked", "fucks",
    "shit", "shits", "shitting",
    "bitch", "bitches",
    "asshole", "dick", "pussy", "cunt",
    "motherfucker", "bullshit",
    "nigger", "nigga", "faggot", "retard",
})

MAX_INTERVALS = 20
MIN_INTERVAL_SECONDS = 0.1
MUTE_PAD_SECONDS = 0.05


def wordlist():
    extra = os.environ.get("PROFANITY_WORDLIST", "")
    words = set(DEFAULT_WORDLIST)
    for w in extra.split(","):
        w = w.strip().lower()
        if w:
            words.add(w)
    return words


def _normalize_word(text):
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", (text or "").lower())


def find_profanity(words, extra_words=()):
    """Filter clip-relative [{w,s,e}, ...] down to profane hits."""
    allowed = wordlist() | {_normalize_word(w) for w in (extra_words or []) if w}
    allowed.discard("")
    hits = []
    for w in words or []:
        norm = _normalize_word(w.get("w", ""))
        if norm and norm in allowed:
            try:
                hits.append({"w": str(w["w"]).strip(),
                             "s": float(w["s"]), "e": float(w["e"])})
            except (KeyError, TypeError, ValueError):
                continue
    hits.sort(key=lambda h: h["s"])
    return hits


def mask_word(word):
    """fuck -> f*** (keep first letter, mask the rest)."""
    w = (word or "").strip()
    if len(w) <= 1:
        return "***" if w else ""
    return w[0] + "***"


def mask_transcript_words(words, hits):
    """Return a copy of clip-relative transcript words with hits masked."""
    hit_keys = {(round(h["s"], 3), round(h["e"], 3)) for h in hits}
    out = []
    for w in words or []:
        try:
            key = (round(float(w["start"]), 3), round(float(w["end"]), 3))
        except (KeyError, TypeError, ValueError):
            out.append(dict(w))
            continue
        if key in hit_keys:
            c = dict(w)
            c["word"] = (" " if str(w.get("word", "")).startswith(" ") else "") \
                + mask_word(str(w.get("word", "")).strip())
            out.append(c)
        else:
            out.append(dict(w))
    return out


def normalize_intervals(intervals, duration=None):
    """Validate/clamp/merge bleep intervals. Returns [{start, end}]."""
    if not isinstance(intervals, (list, tuple)) or not intervals:
        raise ValueError("intervals must be a non-empty list")
    if len(intervals) > MAX_INTERVALS:
        raise ValueError(f"too many intervals (max {MAX_INTERVALS})")
    cleaned = []
    for i, iv in enumerate(intervals):
        try:
            start = float(iv["start"])
            end = float(iv["end"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"interval {i + 1}: start/end must be numbers")
        if start != start or end != end:  # NaN guard
            raise ValueError(f"interval {i + 1}: start/end must be numbers")
        start = max(0.0, round(start - MUTE_PAD_SECONDS, 3))
        end = round(end + MUTE_PAD_SECONDS, 3)
        if duration is not None:
            end = min(float(duration), end)
            start = min(start, float(duration))
        if end - start < MIN_INTERVAL_SECONDS:
            raise ValueError(
                f"interval {i + 1} is shorter than {MIN_INTERVAL_SECONDS}s")
        cleaned.append({"start": start, "end": end})
    cleaned.sort(key=lambda iv: iv["start"])
    merged = []
    for iv in cleaned:
        if merged and iv["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], iv["end"])
        else:
            merged.append(dict(iv))
    return merged


def build_mute_filter(intervals):
    """FFmpeg -af value muting every interval (video stream untouched)."""
    if not intervals:
        raise ValueError("intervals must be a non-empty list")
    parts = []
    for iv in intervals:
        parts.append(
            f"volume=0:enable='between(t,{iv['start']:.3f},{iv['end']:.3f})'")
    return ",".join(parts)
