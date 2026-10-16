"""Tests for profanity bleep intervals: dedupe, post-merge cap, re-bleep union."""
import pytest

import profanity


def _iv(s, e):
    return {"start": s, "end": e}


class TestNormalizeIntervals:
    def test_duplicates_do_not_trip_cap(self):
        # mode='both' sends the same word as auto + manual; retry unions
        # prior + new. 15 words x2 = 30 raw entries must merge to 15.
        base = [_iv(i * 2.0, i * 2.0 + 0.4) for i in range(15)]
        merged_once = profanity.normalize_intervals(base, 60.0)
        assert len(merged_once) == 15
        redo = profanity.normalize_intervals(base + base, 60.0)
        assert len(redo) == 15
        # Re-bleep union: prior (already normalized) + same new intervals.
        union = profanity.normalize_intervals(merged_once + base, 60.0)
        assert len(union) == 15

    def test_overlapping_merge_counts_once(self):
        out = profanity.normalize_intervals(
            [_iv(1.0, 1.5), _iv(1.4, 2.0), _iv(5.0, 5.5)], 60.0)
        assert len(out) == 2

    def test_cap_enforced_post_merge(self):
        many = [_iv(i * 2.0, i * 2.0 + 0.4) for i in range(101)]
        with pytest.raises(ValueError, match="too many intervals"):
            profanity.normalize_intervals(many, 500.0)

    def test_old_cap_now_passes(self):
        # 25 disjoint words used to 400 under the old max-20 pre-merge check.
        many = [_iv(i * 2.0, i * 2.0 + 0.4) for i in range(25)]
        assert len(profanity.normalize_intervals(many, 500.0)) == 25

    def test_build_mute_filter_scales(self):
        intervals = profanity.normalize_intervals(
            [_iv(i * 2.0, i * 2.0 + 0.4) for i in range(25)], 500.0)
        af = profanity.build_mute_filter(intervals)
        assert af.count("volume=0") == 25
