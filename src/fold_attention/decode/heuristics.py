"""The measured rules that choose a decode build for a batch.

Everything here is a fit to H100 SXM measurements: the V format, the
precision gates, the tail's rank, the register caps and residency the
builds reach, and the split of keys over CTAs. None of it changes what the
kernel computes except `refine_for`, `weight_terms_for` and `tail_rank_for`,
which trade accuracy for time and are documented as such.
"""

from __future__ import annotations

import torch

from .config import BN, MR, NT, NWG, smem_bytes

# The live fraction above which the 8-bit V is faster, by head dim, and how
# far each extra eight-row block of the query group moves it. The 8-bit V buys
# bytes with per-tile conversion work, so it pays only where the tile is bound
# by bytes, and a wider group has already spent that issue budget. Fitted on
# GLM-4-9B captures whose wide groups are real query heads (G = 16 and 32).
_V8_LIVE = {128: 0.30, 64: 0.55}
_V8_BUSY = {128: 0.15, 64: 0.45}
# At G <= 8 the 8-bit V is faster down to a live fraction of 0.11 at D = 128
# and 0.28 at D = 64, with equal or lower error at each of those points.
_V8_LIVE_G8 = {128: 0.10, 64: 0.25}


def v8_live_fraction(D: int = 128, G: int = 8) -> float:
    """The live fraction above which the 8-bit V is the faster format."""
    if G <= 8:
        return _V8_LIVE_G8.get(D, _V8_LIVE.get(D, 0.30))
    return _V8_LIVE.get(D, 0.30) + _V8_BUSY.get(D, 0.15) * ((G + 7) // 8 - 1)


def pick_config(
    live_fraction: float,
    nbh: int | None = None,
    seq_len: int | None = None,
    max_len: int | None = None,
    D: int = 128,
    G: int = 8,
    truncate: bool = True,
) -> tuple[bool, int | None]:
    """`(v8, split)` for a live fraction; `split` is None unless the batch's
    `nbh` and `seq_len` are given.

    The live fraction decides V's format because the 8-bit V trades per-tile
    work for bytes, which pays only where the kernel is at its byte floor.
    A caller knows it from column 0 of the previous step's counts.
    """
    v8 = live_fraction >= v8_live_fraction(D, G)
    sp = None
    if nbh is not None and seq_len is not None:
        sp = pick_split(nbh, seq_len, v8, max_len=max_len, D=D, G=G, truncate=truncate)
    return v8, sp


def v_regs_default(v8: bool, D: int, G: int = 8, truncate: bool = True) -> bool:
    """Whether the 8-bit V's value operand comes from registers.

    A lane's load carries `D / 32` channels, four bytes at D = 128 and two at
    D = 64. At D = 128 that beats the widening pass outright; at D = 64 the
    pass, which reads 16-byte units, wins at every depth. The register
    operand holds the whole value fragment live across the tile loop, and at
    G = 64 truncated ptxas spills it, which leaves one CTA per SM where the
    widening arm holds two, so the width is part of the rule.
    """
    if not (v8 and D >= 128):
        return False
    return _arm(D, G, True, True, truncate)[0] < _SPILLS


# The gates under the mass reference, `(refine_k, refine_v)`, by head dim and
# query group class. The target is the fastest depth whose error is at most
# the least accurate BF16 decode kernel's. Checked on capture layers outside
# the reported ones (Qwen3 6/16/36/44, GLM-4 18/36, gpt-oss 3/15, windows with
# the sink, 1K-32K): each gate passes, with 3% of margin, wherever the one it
# would replace does, and holds unless another gains 1% on average over a
# leave-one-layer-out check and loses no more than 1% in any fold. Dense
# builds take one per context band; truncated builds one per depth step,
# `_DEPTH_STEPS` holding each step's lower edge.
_DENSE_K = {
    (128, 1): (12, 12, 12),
    (128, 4): (10, 10, 12),
    (128, 8): (10, 10, 12),
    (128, 16): (10, 12, 12),
    (64, 1): (12, 12, 12),
    (64, 4): (10, 10, 10),
    (64, 8): (12, 12, 10),
}
_DENSE_V8 = {
    (128, 1): ((10, 12), (10, 12), (10, 12)),
    (128, 4): ((10, 10), (10, 10), (10, 10)),
    (128, 8): ((10, 10), (10, 10), (10, 12)),
    (128, 16): ((10, 10), (10, 12), (10, 12)),
    (64, 1): ((10, 10), (10, 10), (12, 8)),
    (64, 4): ((10, 8), (10, 10), (10, 8)),
    (64, 8): ((10, 10), (10, 10), (10, 10)),
}
# measured at depths 10, 12, 13, 14, 16, 18 and 20
_DEPTH_STEPS = (11.0, 12.5, 13.5, 15.0, 17.0, 19.0)
_TRUNC_K = {
    (128, 1): (10, 8, 8, 8, 10, 12, 12),
    (128, 4): (10, 10, 10, 8, 10, 12, 12),
    (128, 8): (10, 8, 8, 8, 10, 12, 12),
    (128, 16): (6, 10, 10, 10, 10, 12, 12),
    (64, 1): (8, 10, 10, 10, 10, 12, 12),
    (64, 4): (6, 8, 8, 8, 10, 10, 10),
    (64, 8): (10, 10, 10, 10, 10, 12, 12),
}
_TRUNC_V8 = {
    (128, 1): ((8, 10), (8, 10), (6, 8), (6, 8), (10, 10), (12, 10), (12, 10)),
    (128, 4): ((8, 10), (6, 8), (6, 8), (6, 8), (10, 10), (10, 10), (10, 10)),
    (128, 8): ((8, 10), (8, 12), (8, 10), (6, 8), (10, 12), (10, 10), (12, 10)),
    (128, 16): ((8, 10), (6, 12), (8, 6), (8, 6), (8, 8), (10, 12), (10, 12)),
    (64, 1): ((6, 6), (6, 6), (6, 6), (10, 10), (10, 10), (10, 10), (12, 10)),
    (64, 4): ((8, 6), (8, 6), (8, 6), (8, 6), (10, 10), (8, 8), (10, 10)),
    (64, 8): ((8, 6), (8, 6), (8, 6), (8, 6), (10, 10), (10, 10), (10, 10)),
}
# the bands' edges, halfway in log2 between the measured contexts 4K and 8K,
# and 16K and 32K
_BAND_EDGES = (5793, 23170)


def refine_for(
    depth: float | None,
    v8: bool = False,
    *,
    seq_len: int,
    head_dim: int = 128,
    group: int = 8,
    tail: bool = True,
) -> tuple[float, float]:
    """`(refine_k, refine_v)` under the mass reference: the depths below the
    row's log-sum-exp inside which K's and V's second planes are read.

    Measured on H100 on Qwen3, GLM-4 and gpt-oss captures (`_DENSE_K`,
    `_DENSE_V8`, `_TRUNC_K`, `_TRUNC_V8`, whose comment gives the target). A
    group wider than 8 at D = 64 takes G = 8's. A build without the tail,
    which is unmeasured below depth 15, keeps 10 there. `refine_v` is unused
    on a bf16 V.
    """
    dk = 64 if head_dim <= 64 else 128
    gc = 1 if group <= 2 else 4 if group <= 4 else 8 if group <= 8 or dk == 64 else 16
    if depth is None:
        band = sum(seq_len > e for e in _BAND_EDGES)
        if v8:
            rk, rv = _DENSE_V8[(dk, gc)][band]
            return float(rk), float(rv)
        return float(_DENSE_K[(dk, gc)][band]), 8.0
    step = sum(depth >= e for e in _DEPTH_STEPS)
    if v8:
        rk, rv = _TRUNC_V8[(dk, gc)][step]
        return float(rk), float(rv)
    if depth < 15.0 and not tail:
        return 10.0, 8.0
    return float(_TRUNC_K[(dk, gc)][step]), 8.0


def refine_bands_for(
    v8: bool = False,
    *,
    head_dim: int = 128,
    group: int = 8,
) -> tuple[float, ...]:
    """The dense gates' length bands for `DecodeConfig.refine_bands`:
    `(edge0, edge1, rk1, rk2, rv1, rv2)`. The kernel picks a row group's band
    from that request's own length, so a request's gates, and its bits, do
    not depend on the requests batched with it. Band 0 is `refine_for`'s at
    the shortest length and is passed as the launch's `refine_k`/`refine_v`.
    """
    (rk1, rv1), (rk2, rv2) = (
        refine_for(None, v8, seq_len=e + 1, head_dim=head_dim, group=group) for e in _BAND_EDGES
    )
    return (float(_BAND_EDGES[0]), float(_BAND_EDGES[1]), rk1, rk2, rv1, rv2)


def weight_terms_for(depth: float | None) -> int:
    """How many bf16 terms a weight is carried in, paired with the depth.

    One bf16 term rounds each weight by up to 2^-9, and at the dense end that
    is the error floor. Two terms carry the rounding error too, and the floor
    falls to that of the logits and V, the same at every reference. The
    second value matmul costs 1-3% of a step, which a cut from depth 15 up
    buys back in keys at equal error on either V format; below it the cut's
    own error is far over either floor.
    """
    if depth is None:
        return 2
    return 2 if depth >= 15.0 else 1


def tail_rank_for(D: int, G: int, v8: bool) -> int:
    """The decode tail's rank: 16, 0 for the block rows alone, or -1 for no
    tail.

    At matched error against FlashAttention-4 and FlashInfer, rank 16 is
    1.03-1.29x at G <= 16. At G = 32 its registers leave 2 CTAs per SM and it
    loses 5-16%; the block rows alone cost nothing in registers there and are
    1.02-1.04x at D = 64 and 0.98x at D = 128. On an 8-bit V the rank term
    costs a CTA per SM or serialises the logit `wgmma` under the cap, so it
    loses time at matched error.
    """
    if v8:
        return -1
    if G <= 16:
        return 16
    return 0 if D < 128 else -1


def front_ownership(total_keys: int, n_groups: int, G: int, D: int, v8: bool, tail: bool) -> int:
    """Which front CTA quantises Q and the new key (`front.mass_front`): 0,
    the append CTA; 1, the mass CTA, which publishes both; 2, the mass CTA
    publishes Q and the append CTA owns K, V and the tail. `total_keys` is
    the batch's keys over every KV head.

    Below 196608 keys, independent producer CTAs hide more latency than
    deduplicating their arithmetic saves. Under 32 row groups of long requests
    an 8-bit V measures otherwise: at D = 64 the append CTA's Q is 2-6%
    faster, and at D = 128 up to G = 4 the mass CTA publishing Q alone is
    1-4% faster."""
    enough_work = total_keys >= 196608
    if tail:
        return 2 if n_groups >= 32 else 0
    if not v8 or (not enough_work and n_groups < 64):
        return 0
    if n_groups < 32 and (D < 128 or G <= 4):
        return 0 if D < 128 else 2
    if G == 8 and D == 128:
        return 2
    if G == 16 and D == 128:
        return 2 if n_groups >= 64 else 0
    return 1


def front_early(n_groups: int, sms: int) -> bool:
    """Whether the front releases its decode as soon as it starts, so the
    decode's prologue overlaps it, rather than when it ends. Early is 1-16%
    faster up to one row group per SM, and at two it is up to 5% faster at 4K
    keys and no slower at 32K. On larger grids the spinning decode CTAs take the SMs
    the front's CTAs need: early loses 1-4.5% for every build at four."""
    return n_groups <= 2 * sms


# Registers ptxas takes uncapped, by (D, truncate, weight terms) then G, for
# the arms (bf16 V, 8-bit V from registers, 8-bit V widened); a width past the
# table spills. Each count is the larger over two grids (64 row groups at
# split 8, 256 at split 3), 1-8 KV heads and every front ownership mode. A
# count moves by up to 20 with the constants a build bakes in; a single split
# folds its index away and moves by -9 to +6, so `_REGS_ONE` holds its counts
# where they differ. A count too low is what spills a grid into a second wave.
_SPILLS = 256
# the counts a single split's build takes where they differ from the tables
_REGS_ONE: dict = {
    "_REGS": {
        (128, 1, 1): {
            4: (56, 95, 80),
            8: (62, 95, 70),
            16: (85, 118, 94),
            32: (148, 186, 158),
            64: (216, 248, 237),
        },
        (128, 1, 2): {4: (55, 95, 64), 32: (156, 188, 166), 64: (216, 248, 237)},
        (128, 0, 2): {
            8: (62, 86, 64),
            16: (90, 115, 92),
            24: (101, 142, 122),
            32: (121, 144, 162),
            48: (170, 212, 198),
            64: (208, 235, 230),
        },
        (64, 1, 1): {8: (47, 72, 62), 16: (76, 93, 84), 48: (187, 212, 200)},
        (64, 1, 2): {8: (48, 64, 56), 16: (76, 93, 83), 48: (187, 213, 200)},
        (64, 0, 2): {8: (48, 64, 56), 16: (72, 92, 85), 32: (126, 141, 126), 48: (178, 202, 188)},
    },
    "_REGS_TAIL": {
        (128, 16, 1): {24: 124},
        (128, 16, 2): {24: 126},
        (128, 0, 1): {24: 116},
        (128, 0, 2): {24: 114},
        (64, 16, 2): {32: 147},
        (64, 0, 2): {8: 47},
    },
    "_REGS_FRONT": {
        (128, 1, 1): {16: (85, 118, 94), 32: (150, 186, 160)},
        (128, 1, 2): {32: (158, 188, 166)},
        (128, 0, 2): {4: (60, 84, 64), 16: (84, 113, 92), 24: (95, 148, 122), 32: (119, 168, 162)},
        (64, 1, 1): {8: (48, 72, 61), 16: (70, 93, 83)},
        (64, 1, 2): {1: (46, 64, 47), 4: (47, 64, 56), 8: (48, 64, 56), 16: (72, 93, 83)},
        (64, 0, 2): {8: (48, 64, 56), 16: (70, 93, 85), 32: (124, 141, 127)},
    },
    "_REGS_TAIL_FRONT": {
        (128, 16, 1): {32: 158},
        (128, 16, 2): {24: 124},
        (128, 0, 1): {24: 118},
        (128, 0, 2): {16: 78, 24: 116},
        (64, 16, 1): {16: 82, 24: 119},
    },
}
_REGS = {
    (128, 1, 1): {
        1: (54, 94, 72),
        4: (56, 96, 80),
        8: (64, 95, 70),
        16: (87, 118, 94),
        24: (108, 144, 122),
        32: (144, 186, 160),
        48: (182, 214, 198),
        64: (220, 248, 237),
    },
    (128, 1, 2): {
        1: (62, 78, 66),
        4: (56, 96, 64),
        8: (64, 95, 64),
        16: (87, 120, 92),
        24: (104, 144, 122),
        32: (148, 186, 168),
        48: (176, 210, 198),
        64: (220, 248, 237),
    },
    (128, 0, 2): {
        1: (46, 80, 70),
        4: (63, 86, 64),
        8: (64, 86, 64),
        16: (94, 115, 92),
        24: (108, 142, 122),
        32: (121, 144, 160),
        48: (172, 212, 198),
        64: (210, 236, 230),
    },
    (64, 1, 1): {
        1: (40, 63, 52),
        4: (48, 64, 62),
        8: (54, 72, 62),
        16: (78, 93, 86),
        24: (110, 128, 119),
        32: (126, 150, 139),
        48: (188, 214, 200),
        64: (232, 245, 242),
    },
    (64, 1, 2): {
        1: (48, 63, 47),
        4: (48, 64, 56),
        8: (55, 64, 56),
        16: (78, 93, 83),
        24: (110, 128, 119),
        32: (126, 148, 139),
        48: (187, 214, 200),
        64: (232, 245, 242),
    },
    (64, 0, 2): {
        1: (48, 63, 54),
        4: (48, 64, 56),
        8: (53, 64, 56),
        16: (76, 92, 85),
        24: (96, 128, 116),
        32: (126, 141, 128),
        48: (176, 201, 189),
        64: (217, 236, 227),
    },
}
# The truncated builds that carry the tail, by (D, rank, weight terms) then G.
# The tail runs on a bf16 V only. A rank the table lacks takes the next one
# up, and one past it the widest, which undercounts.
_REGS_TAIL = {
    (128, 16, 1): {1: 67, 4: 59, 8: 64, 16: 104, 24: 126, 32: 162},
    (128, 16, 2): {1: 77, 4: 71, 8: 72, 16: 124, 24: 128, 32: 168},
    (128, 0, 1): {1: 60, 4: 56, 8: 55, 16: 96, 24: 114, 32: 113},
    (128, 0, 2): {1: 68, 4: 69, 8: 71, 16: 74, 24: 116, 32: 114},
    (64, 16, 1): {1: 52, 4: 56, 8: 64, 16: 83, 24: 117, 32: 146},
    (64, 16, 2): {1: 56, 4: 56, 8: 56, 16: 83, 24: 117, 32: 146},
    (64, 0, 1): {1: 42, 4: 47, 8: 47, 16: 72, 24: 90, 32: 122},
    (64, 0, 2): {1: 48, 4: 48, 8: 48, 16: 64, 24: 91, 32: 120},
}
# The same two for the paged builds behind the front, which serving runs.
_REGS_FRONT = {
    (128, 1, 1): {
        1: (48, 95, 72),
        4: (64, 98, 80),
        8: (56, 114, 70),
        16: (87, 118, 94),
        24: (108, 144, 124),
        32: (148, 186, 160),
    },
    (128, 1, 2): {
        1: (64, 96, 66),
        4: (54, 95, 64),
        8: (56, 114, 64),
        16: (87, 120, 92),
        24: (104, 144, 124),
        32: (152, 186, 168),
    },
    (128, 0, 2): {
        1: (44, 80, 70),
        4: (60, 86, 64),
        8: (62, 86, 64),
        16: (88, 113, 92),
        24: (99, 149, 122),
        32: (120, 168, 160),
    },
    (64, 1, 1): {
        1: (40, 64, 56),
        4: (48, 72, 60),
        8: (54, 72, 61),
        16: (76, 93, 86),
        24: (113, 128, 113),
        32: (126, 150, 139),
    },
    (64, 1, 2): {
        1: (48, 64, 47),
        4: (54, 64, 56),
        8: (56, 64, 56),
        16: (76, 93, 83),
        24: (110, 128, 119),
        32: (125, 148, 139),
    },
    (64, 0, 2): {
        1: (47, 66, 54),
        4: (48, 64, 56),
        8: (51, 64, 56),
        16: (72, 93, 85),
        24: (94, 128, 116),
        32: (124, 141, 128),
    },
}
_REGS_TAIL_FRONT = {
    (128, 16, 1): {1: 67, 4: 57, 8: 58, 16: 104, 24: 124, 32: 162},
    (128, 16, 2): {1: 76, 4: 63, 8: 69, 16: 122, 24: 126, 32: 166},
    (128, 0, 1): {1: 56, 4: 52, 8: 50, 16: 96, 24: 116, 32: 114},
    (128, 0, 2): {1: 68, 4: 68, 8: 72, 16: 72, 24: 118, 32: 114},
    (64, 16, 1): {1: 52, 4: 54, 8: 58, 16: 83, 24: 128, 32: 146},
    (64, 16, 2): {1: 56, 4: 56, 8: 56, 16: 84, 24: 117, 32: 146},
    (64, 0, 1): {1: 40, 4: 47, 8: 48, 16: 70, 24: 92, 32: 119},
    (64, 0, 2): {1: 42, 4: 48, 8: 48, 16: 64, 24: 91, 32: 118},
}
# The deep cap squeezes a build to 80 registers, six CTAs per SM, and is worth
# it up to G = 8 on the 8-bit arms; the shallow squeeze buys one more CTA where
# the bf16 arm is at most this many registers over.
_CAP_REGS = 80
_CAP_G = (0, 8, 8)
_CAP_SHALLOW = (16, 0, 0)
_CAP_FLOOR = 3
# The most registers a build behind the front gives up to hold one more CTA
# per SM. Squeezes of 2-6 registers are 8-23% faster on the tail builds; the
# 8-bit V's 16-register deep cap is 2-6% slower.
_FRONT_SQUEEZE = 6


def _arm(
    D: int,
    G: int,
    v8: bool,
    v_regs: bool,
    truncate: bool,
    tail: int = -1,
    front: bool = False,
    weight_terms: int | None = None,
    one: bool = False,
) -> tuple[int, int, int]:
    """(uncapped registers, widest G the deep cap fits, shallow slack).

    A width the table skips is priced at the higher of its neighbours: the
    count is not monotone in G, and a group with no `wgmma` N is padded up.
    `tail` is the tail's rank, -1 for none; `front` selects the tables of the
    builds serving runs. `weight_terms` None takes the larger of the two: a
    one-term build holds its value matmul's descriptors across the loop, a
    two-term one rebuilds them. `one` is a build of a single split, whose
    counts `_REGS_ONE` overrides.
    """
    col = 1 if (v8 and v_regs) else (2 if v8 else 0)
    dk = 128 if D >= 128 else 64
    if tail >= 0:
        name = "_REGS_TAIL_FRONT" if front else "_REGS_TAIL"
        tt = _REGS_TAIL_FRONT if front else _REGS_TAIL
        ranks = sorted({r for d, r, _ in tt if d == dk})
        second = next((r for r in ranks if r >= tail), ranks[-1])
    else:
        name = "_REGS_FRONT" if front else "_REGS"
        tt = _REGS_FRONT if front else _REGS
        second = 1 if truncate else 0
    over = _REGS_ONE.get(name, {}) if one else {}
    keys = [k for k in tt if k[:2] == (dk, second)]
    if weight_terms is not None and (dk, second, weight_terms) in tt:
        keys = [(dk, second, weight_terms)]
    regs = 0
    for key in keys:
        tab = {**tt[key], **over.get(key, {})}
        up = [k for k in tab if k >= G]
        if not up:
            return _SPILLS, _CAP_G[col], _CAP_SHALLOW[col]
        dn = [k for k in tab if k <= G]
        pick = [tab[min(up)]] + ([tab[max(dn)]] if dn else [])
        # the tail's tables hold the bf16 arm alone
        regs = max(regs, *(x if isinstance(x, int) else x[col] for x in pick))
    return regs, _CAP_G[col], _CAP_SHALLOW[col]


def _resident_by_regs(regs: int) -> int:
    return 65536 // (NT * (-(-regs // 8) * 8))


def capped_footprint(smem: int, n_ctas: int, sms: int) -> int:
    """A decode CTA's shared memory padded so an SM holds at most the grid's
    share of CTAs, `ceil(n_ctas / sms)`. Under early release the front's CTAs
    still hold some SMs when the decode launches, and a build that fits more
    than its share packs the others unevenly: at D = 64 with the rank sums
    in the combine, 9 on 55 SMs and 7 on 67, and the step waits on the full
    ones (4-10% slower)."""
    per = max(1, -(-n_ctas // sms))
    cap = 228 * 1024
    if cap // (smem + 1024) <= per:
        return smem
    # midway between the sizes at which `per` and `per + 1` CTAs just fit, so
    # neither edge's rounding moves the count
    mid = (cap // per + cap // (per + 1)) // 2 - 1024
    return max(smem, mid // 16 * 16)


def _resident_by_smem(D, G, v8, v_regs, tail, weight_terms) -> int:
    return (228 * 1024) // (smem_bytes(D, G, v8, v_regs, tail, weight_terms) + 1024)


def _shallow_cap(
    D: int,
    G: int,
    v8: bool,
    v_regs: bool,
    truncate: bool,
    tail: int = -1,
    front: bool = False,
    weight_terms: int = 1,
    one: bool = False,
) -> int:
    """The residency one shallow register squeeze buys, or 0. Asking costs
    1.5-6% on a build that already holds it, so it is asked only where it
    moves the answer."""
    regs, _, slack = _arm(D, G, v8, v_regs, truncate, tail, front, weight_terms, one)
    if regs >= _SPILLS or not slack:
        return 0
    n = _resident_by_smem(D, G, v8, v_regs, tail, weight_terms)
    free = min(n, _resident_by_regs(regs))
    if free > _CAP_FLOOR or free >= n:
        return 0
    need = (65536 // (NT * (free + 1))) // 8 * 8
    return free + 1 if regs - need <= slack else 0


def cap_wanted(
    D: int,
    G: int,
    v8: bool,
    v_regs: bool,
    truncate: bool,
    tail: int = -1,
    weight_terms: int = 1,
    one: bool = False,
) -> bool:
    """Whether a build outside the front asks ptxas for a register cap; the
    front's builds ask `front_min_blocks`.

    Uncapped, ptxas spends whatever registers shared memory leaves it, which
    loses the register-V arm a CTA per SM at the widths the deep cap reaches.
    Everywhere else a cap costs more than the residency it holds: the widened
    8-bit arm is 0-7% faster uncapped at every width, head dim and depth.
    """
    return bool(
        (v8 and v_regs and G <= _CAP_G[1])
        or _shallow_cap(D, G, v8, v_regs, truncate, tail, False, weight_terms, one)
    )


# Uncapped registers of the packed build (`decode.packed`, D = 64) by (G,
# weight terms), with the rank-16 tail and without it.
_REGS_PACKED = {
    (1, 1): 72,
    (1, 2): 81,
    (2, 1): 72,
    (2, 2): 82,
    (3, 1): 88,
    (3, 2): 99,
    (4, 1): 91,
    (4, 2): 92,
}
_REGS_PACKED_PLAIN = {
    (1, 1): 48,
    (1, 2): 48,
    (2, 1): 56,
    (2, 2): 56,
    (3, 1): 56,
    (3, 2): 58,
    (4, 1): 56,
    (4, 2): 56,
}


def packed_min_blocks(G: int, tail: int, weight_terms: int, smem: int) -> int:
    """The residency the packed build asks ptxas for, or 0: its shared
    memory's, where its own count would cost a CTA a SM by at least two
    registers. Asked where the build already fits, the cap costs ~6%: ptxas
    spends up to it and schedules the loop longer. A squeeze of one register
    loses too (G = 1 with two weight terms, 81 to 80, is up to 5% slower),
    where two gains 15-22% (G = 2, 82 to 80)."""
    res = (228 * 1024) // (smem + 1024)
    tab = _REGS_PACKED if tail >= 0 else _REGS_PACKED_PLAIN
    regs = tab.get((G, weight_terms), max(tab.values()))
    return res if regs - 65536 // (NT * res) // 8 * 8 >= 2 else 0


def front_min_blocks(
    D: int,
    G: int,
    v8: bool,
    v_regs: bool,
    truncate: bool,
    tail: int,
    weight_terms: int,
    one: bool = False,
) -> int:
    """The residency a build behind the front asks ptxas for, or 0 for none:
    one CTA per SM more than its uncapped registers hold, where shared memory
    has room for it and it costs at most `_FRONT_SQUEEZE` registers. A D = 128
    register-V build up to the deep cap's width always asks the deep cap's.

    D = 64 never squeezes: below a wave of the uncapped residency the extra
    resident CTAs spin on the front's words and starve it, and past one a
    squeeze is 2-10% slower (the bf16 G = 8 tail build at 256 row groups,
    ragged, and the 8-bit V G = 8 build)."""
    regs = _arm(D, G, v8, v_regs, truncate, tail, True, weight_terms, one)[0]
    if regs >= _SPILLS or D < 128:
        return 0
    n = _resident_by_smem(D, G, v8, v_regs, tail, weight_terms)
    if v8 and v_regs and G <= _CAP_G[1]:
        # Uncapped, ptxas spends 21-25 registers on running every slice's V
        # conversion at once and holds 4 CTAs, 6-10% slower; under the deep
        # cap it holds 6 without spilling.
        return min(n, _resident_by_regs(_CAP_REGS))
    have = _resident_by_regs(regs)
    if have >= n:
        return 0
    need = (65536 // (NT * (have + 1))) // 8 * 8
    return have + 1 if regs - need <= _FRONT_SQUEEZE else 0


def resident(
    v8: bool,
    v_regs: bool,
    D: int = 128,
    G: int = 8,
    truncate: bool = True,
    front: bool = False,
    cap: bool | None = None,
    tail: int = -1,
    weight_terms: int = 1,
    one: bool = False,
) -> int:
    """CTAs per SM a build holds on an H100 (228 KB shared, 64K registers),
    the smaller of what shared memory and registers allow. `cap` defaults to
    `cap_wanted`, the condition the build asks under; `tail` is the tail's
    rank, -1 for none; `one` a build of a single split."""
    n = _resident_by_smem(D, G, v8, v_regs, tail, weight_terms)
    regs, cap_g, _ = _arm(D, G, v8, v_regs, truncate, tail, front, weight_terms, one)
    if regs >= _SPILLS:
        return 1
    if front:
        sq = front_min_blocks(D, G, v8, v_regs, truncate, tail, weight_terms, one=one)
        return sq or max(1, min(n, _resident_by_regs(regs)))
    sh = _shallow_cap(D, G, v8, v_regs, truncate, tail, front, weight_terms, one)
    if cap is None:
        cap = cap_wanted(D, G, v8, v_regs, truncate, tail, weight_terms, one)
    if cap and G <= cap_g:
        regs = min(regs, _CAP_REGS)
    return max(1, (sh if cap else 0) or min(n, _resident_by_regs(regs)))


# The split values the rules return. A fitted real number is snapped to one
# of these rather than rounded: the measured curves are flat near their
# minimum but have sharp local spikes, and a neighbouring candidate costs far
# less than a spike.
_SPLIT_CANDIDATES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256)

# The bf16 wave model's constants, by (head dim, dense): a CTA's fixed cost in
# tiles, the power its fill charges the last partial wave, and the most CTAs
# per SM worth filling (None for all that fit); and each split's own cost in
# tiles, the combine's extra partial and the CTA's latency, which stops a
# small batch splitting without bound. Fitted on 1110 swept curves (D = 128 at
# G = 1, 8, 16 and D = 64 at G = 1, 8; 4-512 row groups; 4K-128K keys; uniform
# and ragged; dense and truncated), it gives up 1.3% against the best swept
# split in the mean and 20% at worst; on 1400 curves at the current register
# tables (every family, 4-256 row groups, 1K-32K, dense and depths 13, 16,
# 18) it gives up 0.4-1.8% per family in the mean and 10.5% at worst, within
# 0.15% of the best constants there. The dense D = 128 kernel saturates at
# four CTAs, so it gains nothing from filling every slot.
_WAVE = {
    (128, False): (6, 0.0, None),
    (128, True): (10, 0.0, 4),
    (64, False): (6, 0.25, None),
    (64, True): (4, 0.0, 8),
}
# The 8-bit V's constants, one set for both head dims and both regimes: a
# CTA's fixed cost is nothing, the partial wave is charged its fill's square
# root, and every resident CTA is worth filling. Fitted on 700 swept curves
# (D = 128 at G = 1, 4, 8, 16 and D = 64 at G = 1, 4, 8; 4-256 row groups;
# 1K-32K keys; uniform and ragged; dense and depth 16), it gives up 1.2-1.9% in
# the mean and 19% at worst, where the rules it replaced gave up 6-14%; on
# 1400 curves at the current register tables and depths 13, 16 and 18 as
# well, 0.8-2.8% per family in the mean and 19% at worst, the worst a batch of
# 4-8 row groups at 32K that wants 128 splits.
_WAVE_V8 = (0, 0.5, None)
_SPLIT_COST = 0.1


def _wave_split(nbh: int, seq_len: int, max_len: int, slots, c0: float, beta: float) -> int:
    """The candidate split with the least waves times a CTA's tiles plus its
    fixed cost, `slots(sp)` the resident CTAs its grid runs at. A ragged
    batch's longest request bounds the time from below: the longest-first
    order starts its CTAs first, so they end last."""
    tm = -(-seq_len // BN)
    tx = -(-max_len // BN)
    best: tuple[float, int] | None = None
    for sp in _SPLIT_CANDIDATES:
        sl = slots(sp)
        full, part = divmod(nbh * sp, sl)
        waves = full + ((part / sl) ** beta if part else 0.0)
        t = max(waves * (-(-tm // sp) + c0), -(-tx // sp) + c0) + _SPLIT_COST * sp
        if best is None or t < best[0] - 1e-9:
            best = (t, sp)
    assert best is not None
    return best[1]


def pick_split(
    nbh: int,
    seq_len: int,
    v8: bool,
    sms: int | None = None,
    v_regs: bool | None = None,
    max_len: int | None = None,
    D: int = 128,
    G: int = 8,
    truncate: bool = True,
    front: bool = False,
    tail: int = -1,
    weight_terms: int = 1,
) -> int:
    """CTAs per row group for `nbh` row groups over `seq_len` keys, or for a
    ragged batch whose mean length is `seq_len` and longest request `max_len`.

    Under a static reference a split changes no weight, so it decides only
    how the machine is filled: `_wave_split` over the build's resident CTAs,
    with each V format's own constants.
    """
    if max_len is not None:
        max_len = int(max_len)
    if sms is None:
        sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if v_regs is None:
        v_regs = v_regs_default(v8, D, G, truncate)
    res = [
        resident(
            v8,
            v_regs,
            D=D,
            G=G,
            truncate=truncate,
            front=front,
            tail=tail,
            weight_terms=weight_terms,
            one=one,
        )
        for one in (False, True)
    ]
    c0, beta, reff = _WAVE_V8 if v8 else _WAVE[(128 if D >= 128 else 64, not truncate)]

    def slots(sp):
        one = sp == 1
        n = res[one]
        if front:
            n = front_min_blocks(D, G, v8, v_regs, truncate, tail, weight_terms, one=one) or n
        return sms * (n if reff is None else min(n, reff))

    return _wave_split(nbh, seq_len, max_len or seq_len, slots, c0, beta)


def shared_runs_wide(rows: int, D: int, image_length: int | None) -> bool:
    """Whether a shared level runs on the wide kernel: past 64 stacked rows
    the decode kernel cannot hold the rows, and at 64 the wide kernel wins
    with an image when the rows are narrow (D = 64) or the prefix is long
    (16K keys up). Below 64 the decode kernel is faster at every shape
    measured (6-18% at 16 and 32 rows)."""
    if rows > 64:
        return True
    return rows == 64 and image_length is not None and (D == 64 or image_length >= 16384)


def wide_split(n_groups: int, keys: int, rows: int, device) -> int:
    """Splits for a wide level: whole waves of one CTA per SM, each split at
    least four tiles, and the fewest splits within 5% of the best."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    tiles = -(-keys // BN)
    ctas = n_groups * -(-rows // (MR * NWG))
    cost = {}
    for sp in range(1, 129):
        if sp > 1 and tiles < 4 * sp:
            break
        cost[sp] = -(-ctas * sp // sms) * (-(-tiles // sp) + 2)
    best = min(cost.values())
    return min(sp for sp, c in cost.items() if c <= best * 1.05)
