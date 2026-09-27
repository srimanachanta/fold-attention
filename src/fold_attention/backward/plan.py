"""How one backward call's work is cut up, and the dense work list's order.

A work tile is `(n_block, head, batch, record)`: one key block, `subgroup`
query heads walked in turn, and the `record`-th run of `record_width` query
blocks, or the whole query range when records are off. Its cost is counted in
query blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Keeping a whole GQA group's dK/dV private to one CTA divides the grid by the
# group. It pays once the whole-group tiles number this many: at 128 it is
# 1.6-1.8x slower than canonical records, from 256 up 2-7% faster.
WHOLE_GROUP_TILES = 256
# From this group size, counted at head_dim 128, a whole-group tile is too
# long to balance: the last tiles to start run past the rest of the machine by
# up to a third at G = 16.
SPLIT_GROUP = 8
# Heads whose Q, dO and dQ tiles the dense order keeps live at once.
SECTION_BUDGET_BYTES = 64 * 2**20


@dataclass(frozen=True)
class TileConfig:
    """FA-3's SM90 backward tiles. dQ is split between the two MMA warpgroups,
    which ping-pong needs, along the query rows (`atom_layout_m_dq` 2) or
    along the head dim (1)."""

    tile_m: int
    tile_n: int
    atom_layout_m_dq: int


def tile_config(head_dim: int) -> TileConfig:
    if head_dim == 64:
        return TileConfig(128, 128, 2)
    if head_dim in (96, 128):
        return TileConfig(64, 128, 1)
    raise NotImplementedError(f"head_dim {head_dim}: 64, 96 and 128 are supported")


@dataclass(frozen=True)
class Plan:
    """Every field that changes bits is a function of the request's shape: a
    dense call's whole `(B, H, S, D)` tensor is the request, and a packed
    batch decides from head counts alone, so a request's gradients do not
    depend on what it was packed with."""

    subgroup: int  # query heads per work tile; the whole group keeps dK/dV private
    record: int  # canonical dK/dV record width in query blocks, 0 for the whole range


def split_subgroup(G, cfg: TileConfig) -> int:
    """The subgroup a group too long for one tile splits into, 0 if it is not.

    A tile walks `S / tile_m` query blocks per head, and a block costs about
    the same at every head dim, so at head_dim 64 (128-row blocks) a group is
    as long as half as many heads at 128. A long group splits into subgroups
    as long as two heads at 128: two heads there, four at 64. Those combine
    in fp32 and are 5-35% faster than the whole group, dense and packed, and
    faster than the other subgroup sizes."""
    scale = cfg.tile_m // 64
    sub = 2 * scale
    if G < SPLIT_GROUP * scale or G % sub or sub >= G:
        return 0
    return sub


def plan_dense(B, H, H_KV, S, D, causal, cfg: TileConfig) -> Plan:
    G = H // H_KV
    nm, nn = -(-S // cfg.tile_m), -(-S // cfg.tile_n)
    subgroup, record = G, 0
    sub = split_subgroup(G, cfg)
    if sub and B * (H // sub) * nn >= WHOLE_GROUP_TILES:
        return Plan(subgroup=sub, record=0)
    if G > 1 and B * H_KV * nn < WHOLE_GROUP_TILES:
        # Too few whole-group tiles to fill the machine: cut heads into
        # subgroups and long query ranges into canonical records.
        sub, joint_width = joint_head_record_schedule(
            B, H, H_KV, S, S, D, cfg.tile_m, cfg.tile_n, causal
        )
        width = joint_width if sub > 1 else canonical_record_width(H, S, S, cfg.tile_m, cfg.tile_n)
        if not (H_KV >= 8 and sub == 1 and width >= nm):
            subgroup, record = sub, (width if width < nm else 0)
    return Plan(subgroup=subgroup, record=record)


def plan_varlen(H, H_KV, cfg: TileConfig) -> Plan:
    G = H // H_KV
    sub = split_subgroup(G, cfg)
    if sub:
        return Plan(subgroup=sub, record=0)
    # A whole-group tile is G heads long and there are H_KV of them per key
    # block. When the group is long against the number of tile columns, a long
    # request's tiles make the tail, and halving the group wins by up to 1.6x.
    subgroup = 2 if G % 2 == 0 and G >= 2 * H_KV else G
    return Plan(subgroup=subgroup, record=0)


def canonical_record_width(heads, S_Q, S_K, tile_m, tile_n, target_tiles=256):
    nm, nn = -(-S_Q // tile_m), -(-S_K // tile_n)
    if nm <= 16:
        return nm
    records = max(1, -(-target_tiles // (heads * nn)))
    return -(-nm // records)


def canonical_record_count(B, heads, S_Q, S_K, tile_m, tile_n, causal, width):
    nm, nn = -(-S_Q // tile_m), -(-S_K // tile_n)
    total = 0
    for n in range(nn):
        blocks = max(nm - (n * tile_n + S_Q - S_K) // tile_m, 0) if causal else nm
        total += max(-(-blocks // width), 1)
    return B * heads * total


def joint_head_record_schedule(B, H, H_KV, S_Q, S_K, D, tile_m, tile_n, causal):
    """`(subgroup, record width)` for a grouped causal call too small for
    whole-group tiles: two- or four-head subgroups where they still leave
    enough records, single heads otherwise."""
    group = H // H_KV
    width = canonical_record_width(H, S_Q, S_K, tile_m, tile_n)
    if not causal or group <= 2 or S_K < (4096 if D <= 64 else 8192):
        return 1, width
    subgroup = next((g for g in (4, 2) if g < group and group % g == 0), 1)
    if subgroup == 1:
        return 1, width
    subgroup_width = canonical_record_width(H // subgroup, S_Q, S_K, tile_m, tile_n)
    tiles = canonical_record_count(
        B, H // subgroup, S_Q, S_K, tile_m, tile_n, causal, subgroup_width
    )
    minimum = 192 if D <= 64 else 256
    return (subgroup, subgroup_width) if tiles >= minimum else (1, width)


def tiles(B, heads, S_Q, S_K, tile_m, tile_n, causal, G=1):
    """Every `(n_block, head, batch, 0)` tile and its cost in query blocks."""
    nm, nn = -(-S_Q // tile_m), -(-S_K // tile_n)
    n = np.arange(nn)
    if causal:
        cnt = np.maximum(nm - (n * tile_n + S_Q - S_K) // tile_m, 0)
    else:
        cnt = np.full(nn, nm)
    b, h, nb = np.meshgrid(np.arange(B), np.arange(heads), n, indexing="ij")
    b, h, nb = (x.reshape(-1) for x in (b, h, nb))
    return (
        np.stack([nb, h, b, np.zeros_like(nb)], 1).astype(np.int32),
        (cnt[nb] * G).astype(np.float64),
    )


def canonical_record_tiles(B, heads, S_Q, S_K, tile_m, tile_n, causal, width, G=1):
    """Every tile cut into records of `width` query blocks, the record index
    in the last field."""
    coords, cost = tiles(B, heads, S_Q, S_K, tile_m, tile_n, causal, G=G)
    blocks = (cost / G).astype(np.int64)
    count = np.maximum(-(-blocks // width), 1)
    rep = np.repeat(np.arange(len(blocks)), count)
    record = np.arange(len(rep)) - np.repeat(np.cumsum(count) - count, count)
    assert record.max(initial=0) < 256, "canonical record index overflows the split field"
    out = coords[rep].copy()
    out[:, 3] = record
    record_cost = np.minimum(blocks[rep] - record * width, width)
    return out, (record_cost * G).astype(np.float64)


def pair_bytes(S, D, G):
    """One `(batch, head)` pair's Q and dO (bf16) and dQ accumulator (int32)
    for a tile of `G` query heads."""
    return S * 2 * D * 2 * G + S * D * 4 * G


def section_heads(S, D, G):
    """`(batch, head)` pairs per section of the work order: as many as keep
    their Q, dO and dQ inside `SECTION_BUDGET_BYTES`, rounded down to a power
    of two. dQ is visited a tile at a time, but the budget charges all of it,
    which is the measured concurrency boundary rather than the L2 size."""
    one = pair_bytes(S, D, G)
    return max(1, 1 << math.floor(math.log2(max(1, SECTION_BUDGET_BYTES // one))))


def work_order(coords, cost, heads, section, one, tail_cap, sms):
    """Cost-descending tiles inside sections of `section` `(batch, head)`
    pairs, so the heads live at once stay few enough for L2. The last
    sections merge into one cost-descending run holding a wave of the
    largest tile over `sms` CTAs, within `tail_cap` bytes, so the list ends
    on short tiles whatever the section size. `one` is `pair_bytes`."""
    bh = coords[:, 2].astype(np.int64) * heads + coords[:, 1]
    sec = bh // section
    nsec = int(sec.max()) + 1
    work = np.bincount(sec, weights=cost, minlength=nsec)
    need = cost.max() * sms
    cap = max(1, tail_cap // (section * one))
    first, acc = nsec, 0.0
    while first > 0 and acc < need and nsec - first < cap:
        first -= 1
        acc += work[first]
    return np.lexsort((bh, -cost, np.minimum(sec, first)))


def work_list(B, H, H_KV, S, D, causal, cfg: TileConfig, plan: Plan, sms: int):
    """The dense work list in dispatch order, `(tiles, 4)` int32.

    A tile that walks a whole GQA group is long, and its sections are as
    large as L2 allows, which balances them; every other kind runs in
    sections of two pairs. There each pair's key blocks run side by side and
    share its Q, dO and dQ lines, which at S <= 2048 is 10-30% faster than
    sections filling the budget, and level at longer S."""
    heads = H // plan.subgroup
    if plan.record:
        coords, cost = canonical_record_tiles(
            B, heads, S, S, cfg.tile_m, cfg.tile_n, causal, plan.record, G=plan.subgroup
        )
    else:
        coords, cost = tiles(B, heads, S, S, cfg.tile_m, cfg.tile_n, causal, G=plan.subgroup)
    G = plan.subgroup
    section = section_heads(S, D, G)
    if not (G > 1 and G == H // H_KV):
        section = min(section, 2)
    one = pair_bytes(S, D, G)
    idx = work_order(coords, cost, heads, section, one, SECTION_BUDGET_BYTES, sms)
    return coords[idx].copy()
