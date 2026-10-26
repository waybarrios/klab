"""Static prefetch policy derived from the H1 characterization sweep (superseded by capture-time autotune)."""
import math

L2_BYTES = 4 << 20
MAX_BYTES = int(1.5 * (1 << 20))   # current + next layer must fit in ~0.75 of L2
MAX_M = 32
BLOCK_RATE = 140e3                  # bytes per us issued by one 256-thread prefetch block
MARGIN = 1.25


def plan(nbytes, m):
    """Return prefetch kwargs for a consumer reading `nbytes` of weights with M rows, or None."""
    if nbytes > MAX_BYTES or m > MAX_M:
        return None
    t_est = 2.2 + nbytes / 330e3
    blocks = max(1, math.ceil(MARGIN * nbytes / (BLOCK_RATE * t_est)))
    return {"blocks": blocks, "kernel": "l2_hint", "stride": 128}
