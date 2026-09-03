# -*- coding: utf-8 -*-
"""Shared polite pacing for every network job.

Pacing reduces load; it is not an attempt to evade bot protection. Requests use
longer, irregular pauses and periodic breaks without rotating user agents,
fingerprints, or IP addresses. Every helper accepts an RNG for deterministic
tests.
"""

import random
import time

# Occasional long-tailed pause above the normal delay range.
LONG_TAIL_PROB = 0.15
LONG_TAIL_MULT = 2.5

# A longer break every N requests.
BREAK_EVERY = 15
BREAK_RANGE = (30.0, 90.0)


def human_pause(lo: float, hi: float, rng=random) -> float:
    """Calculate a delay in seconds without sleeping; the result is always >= lo."""
    if rng.random() < LONG_TAIL_PROB:
        return rng.uniform(hi, hi * LONG_TAIL_MULT)
    return rng.uniform(lo, hi)


def long_break(i: int, every: int = BREAK_EVERY, rng=random) -> float | None:
    """Return a long-break duration after request ``i``, or ``None``."""
    if i > 0 and every > 0 and i % every == 0:
        return rng.uniform(*BREAK_RANGE)
    return None


def mean_pause(lo: float, hi: float, every: int = BREAK_EVERY) -> float:
    """Expected delay per request including long tails and periodic breaks."""
    base = (1 - LONG_TAIL_PROB) * (lo + hi) / 2 + LONG_TAIL_PROB * (hi + hi * LONG_TAIL_MULT) / 2
    per_break = (sum(BREAK_RANGE) / 2 / every) if every > 0 else 0.0
    return base + per_break


def polite_sleep(
    i: int, delay_range: tuple[float, float], log=None, rng=random, break_every: int = BREAK_EVERY
) -> float:
    """Sleep after request ``i`` using the shared pacing policy.

    ``break_every`` varies by resource: HTML navigation and static CDN files
    have different normal access patterns.
    """
    brk = long_break(i, every=break_every, rng=rng)
    if brk is not None:
        if log:
            log.info(f"  ☕ break {brk:.0f}s (after {i} requests)")
        time.sleep(brk)
        return brk
    d = human_pause(*delay_range, rng=rng)
    time.sleep(d)
    return d
