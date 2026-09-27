"""Query-row layouts: draft trees and a cascade's stacked rows.

A row group is `G` query rows against one KV head. Under a draft they are
`G // q_len` query heads at `q_len` positions, row `g * q_len + t`. Under a
cascade's shared level they are several requests' rows stacked, named by a row
map.
"""

from __future__ import annotations

import itertools

import torch

# The stacked widths `wgmma` can issue: N is 8i for i in 1..4 and 16i above,
# and a row group holds at most 64 rows.
CASCADE_WIDTHS = (8, 16, 24, 32, 48, 64)


def draft_mask(parents) -> torch.Tensor:
    """The `(q_len, q_len)` ancestor mask of a draft tree from its parents.

    `parents[i]` is node `i`'s parent, or -1 for a root. Node `i` attends `j`
    exactly when `j` is `i` or an ancestor; the chain `[-1, 0, 1, 2]` is
    `causal=True`. A parent must precede its child in the cache.
    """
    par = [int(x) for x in parents]
    n = len(par)
    m = torch.zeros((n, n), dtype=torch.bool)
    for i, pi in enumerate(par):
        if pi >= i:
            raise ValueError(
                f"node {i} has parent {pi}: a draft node's parent is written "
                "to the cache before it, so it must have a lower index"
            )
        m[i, i] = True
        j = pi
        while j >= 0:
            m[i, j] = True
            j = par[j]
    return m


def _as_int32(x: torch.Tensor) -> torch.Tensor:
    """32 mask bits held in int64 as the int32 with the same bits: bit 31, a
    32-node draft's last, is the sign."""
    return (((x + 2**31) % 2**32) - 2**31).to(torch.int32)


def pack_tree(mask, causal, q_len: int, G: int, NBH: int, device) -> torch.Tensor:
    """`(NBH, G)` int32: bit `j` of row `g * q_len + t` says draft node `t`
    attends node `j`."""
    if causal:
        bits = _as_int32(
            torch.tensor([(1 << (t + 1)) - 1 for t in range(q_len)], dtype=torch.int64)
        ).to(device)
        return bits.repeat(G // q_len)[None].expand(NBH, G).contiguous()
    m = mask if torch.is_tensor(mask) else torch.as_tensor(mask)
    m = (m if m.dtype == torch.bool else m != 0).to(device)
    if m.dim() == 2:
        m = m[None]
    if m.dim() != 3 or tuple(m.shape[-2:]) != (q_len, q_len):
        raise ValueError(
            f"mask is (q_len, q_len) or (NBH, q_len, q_len) bool; expected "
            f"the last two axes to be {(q_len, q_len)}, got {tuple(m.shape)}"
        )
    if m.shape[0] not in (1, NBH):
        raise ValueError(
            f"mask's leading axis is the row group: expected 1 or NBH={NBH}, "
            f"got {m.shape[0]}. A per-request tree is "
            "`mask.repeat_interleave(n_kv_heads, 0)`"
        )
    d = torch.arange(q_len, device=device)
    if not bool(m[:, d, d].all()):
        raise ValueError(
            "a draft node attends its own key, so the mask's diagonal must be "
            "set; row(s) without it would divide by an empty softmax"
        )
    w = 1 << torch.arange(q_len, device=device, dtype=torch.int64)
    bits = _as_int32((m.to(torch.int64) * w).sum(-1))
    return bits.repeat(1, G // q_len).expand(NBH, G).contiguous()


def pack_rows(x, n_kv_heads: int, q_len: int = 1) -> torch.Tensor:
    """`(B, q_len, H_q, ...)` to the kernel's `(NBH, G, ...)`.

    `NBH = B * H_KV` indexes `b * H_KV + h`, and row `g * q_len + t` is the
    `g`-th query head of KV head `h` at draft position `t`. It takes `q`, `z`
    and `cut` alike; `unpack_rows` is the inverse.
    """
    B, T, H = x.shape[:3]
    tail = x.shape[3:]
    if T != q_len:
        raise ValueError(f"axis 1 is the draft length: expected {q_len}, got {T}")
    if H % n_kv_heads:
        raise ValueError(f"H_q={H} is not a multiple of n_kv_heads={n_kv_heads}")
    g0 = H // n_kv_heads
    return (
        x.reshape(B, T, n_kv_heads, g0, *tail)
        .permute(0, 2, 3, 1, *range(4, 4 + len(tail)))
        .reshape(B * n_kv_heads, g0 * T, *tail)
        .contiguous()
    )


def unpack_rows(x, n_kv_heads: int, q_len: int = 1) -> torch.Tensor:
    """`(NBH, G, ...)` to `(B, q_len, H_q, ...)`, the inverse of `pack_rows`."""
    NBH, G = x.shape[:2]
    tail = x.shape[2:]
    if G % q_len:
        raise ValueError(f"G={G} is not a multiple of q_len={q_len}")
    if NBH % n_kv_heads:
        raise ValueError(f"NBH={NBH} is not a multiple of n_kv_heads={n_kv_heads}")
    return (
        x.reshape(NBH // n_kv_heads, n_kv_heads, G // q_len, q_len, *tail)
        .permute(0, 3, 1, 2, *range(4, 4 + len(tail)))
        .reshape(NBH // n_kv_heads, q_len, n_kv_heads * (G // q_len), *tail)
        .contiguous()
    )


def cascade_width(rows: int) -> int:
    """The stacked width that holds `rows` query rows: a width the decode
    kernel's matmul can issue up to 64, and past 64 a multiple of eight, which
    the wide kernel reads in tiles of 64 rows."""
    for w in CASCADE_WIDTHS:
        if rows <= w:
            return w
    return -(-rows // 8) * 8


def cascade_degree(G: int) -> int:
    """Requests to share one prefix with on the decode kernel, at a row group
    of `G`.

    The budget is 32 stacked rows, not the 64 a row group holds: a 64-row
    group runs out of issue slots, so reading the prefix twice as often at
    width 32 wins at every G. Past 64 rows a level runs on the wide kernel
    instead, which takes the whole batch (`cascade_rows`).
    """
    return max(1, 32 // int(G))


def cascade_rows(n_req: int, G: int, n_kv_heads: int = 1, groups=None, device=None):
    """The shared level's row map: which (row group, row) each stacked row is.

    `groups` is an indptr over requests: `[0, n_req]` is one prefix for the
    whole batch, `[2, 4]` a prefix only requests 2 and 3 hold. Requests sharing
    a prefix must be contiguous. `None` stacks the whole batch once it is past
    64 rows, which the wide kernel reads in one pass, and otherwise chunks it at
    `cascade_degree`.

    Returns `(rows, G_s)`: `rows` is `(n_ranges * n_kv_heads, G_s)` int32, each
    word `((bh * 64 + g) * 2) | 1`. A padding row points at its range's first
    row with bit 0 clear, so its loads stay in bounds and it moves no bytes.
    """
    if groups is None:
        if n_req * G > 64:
            groups = [0, n_req]
        else:
            d = cascade_degree(G)
            groups = list(range(0, n_req, d)) + [n_req]
    groups = [int(x) for x in (groups.tolist() if torch.is_tensor(groups) else groups)]
    if len(groups) < 2 or groups[0] < 0 or groups[-1] > n_req:
        raise ValueError(
            f"groups={groups} is an indptr over the {n_req} requests: it must "
            "lie inside [0, n_req]. A piece need not cover the batch, but it "
            "cannot name a request the batch does not have"
        )
    if any(b <= a for a, b in itertools.pairwise(groups)):
        raise ValueError(f"groups={groups} must increase; a prefix with no requests has no level")
    G_s = cascade_width(max(b - a for a, b in itertools.pairwise(groups)) * G)
    ng = len(groups) - 1
    start = torch.tensor(groups[:-1])[:, None, None]
    stop = torch.tensor(groups[1:])[:, None, None]
    hkv = torch.arange(n_kv_heads)[None, :, None]
    r = torch.arange(G_s)[None, None, :]
    b = start + r // G
    g = r % G
    ok = b < stop
    b = torch.where(ok, b, start)
    g = torch.where(ok, g, 0)
    rows = (((b * n_kv_heads + hkv) * 64 + g) * 2) | ok.to(torch.int64)
    return rows.reshape(ng * n_kv_heads, G_s).to(torch.int32).to(device), G_s
