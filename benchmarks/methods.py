"""Decode methods of every class at one error metric and one byte model.

On the final state of each generation case (`generate.CASES`, the same
prompts, steps and seed), every method is scored against FP32 softmax
attention over the bf16 cache, and charged the bytes a decode reads from the
cache per key and KV head, metadata included, averaged over the batch:

- FoldAttention members, run through `FoldKVCache` as `generate` runs them;
  error and bytes come from the kernel's own output and key counts
  (`fold.bytes_read`). An 8-bit V's second plane is charged from the
  emulated V gate, since the kernel does not count it.
- Page selection (Quest): 16-token pages with per-channel bf16 min and max,
  the upper-bound score `sum_d max(q_d min_d, q_d max_d)`, top pages at a
  token budget with the sink page and the last page always kept, and exact
  attention over the selected pages only. Selection shared by the group
  (max over its query rows) and per query head.
- Top-delta block dropping (FFD, `qluoluo/faster-flash-decoding`): keys in
  symmetric 2-bit codes with one scale per channel per 128-key block and an
  e4m3 residual, the last partial block kept in bf16. Each row's threshold
  is the largest 2-bit screen score of the first and the last full block
  less `delta` (base 2); a sub-block is read when any row of the group
  reaches its threshold on the screen, and its logits are then
  screen + residual. Dropped mass leaves the denominator.
- Quantised caches: KIVI 2- and 4-bit (K per channel over 32-token groups, V
  per token over 32-channel groups, the newest 128-159 tokens in bf16), INT8
  per token, and e4m3 with one scale per tensor or per (request, KV head).
  These are the format's error, with exact arithmetic on the dequantised
  cache; the FP8 kernels themselves run too, on the same state.
- The BF16 kernels (FA-3, FA-4, FlashInfer, XQA, cuDNN), tuned as
  `generate` tunes them, give each case's BF16 error band.

`gate` compares Fold's refine and cut decisions, which test the plane-A
logit against the step's reference Z, with the same rules tested against an
online softmax's running maximum: per row, the largest plane-A logit of the
keys the split has seen through the current 64-key tile, with the member's
split count as contiguous chunks, as a split-KV online-softmax kernel scans
them, and over one unsplit scan. That test is sound (a weight is at most `2^(s - m)`),
so it needs no reference, and the comparison is what it costs in bytes. The
plane-A logits are the member's own (`FoldKVCache.coarse_logits`), and the
Z-gated counts they give are checked against the kernel's.

`cut_mass` (per case, and pooled over the cases in the file's meta) is, per
query row of the truncated bf16-V members, the share of the row's mass on
the keys the kernel cuts, weighed by the FP32 logits and by the kernel's own
plane-A weights, and their ratio: whether the mass the kernel sums for the
cut keys is a faithful measure of what it drops.

    python -m benchmarks.methods --out methods
"""

from __future__ import annotations

import argparse
import math
import traceback
from typing import Any

import torch

from benchmarks.generate import CASES
from benchmarks.harness import baselines as B
from benchmarks.harness import captures as C
from benchmarks.harness import fold as F
from benchmarks.harness.report import Report
from benchmarks.harness.timing import L2Flush
from fold_attention import FoldKVCache
from fold_attention.decode.config import BN

LOG2E = 1.0 / math.log(2.0)
DEPTHS = (20, 18, 16, 14, 13, 12)
QUEST_BUDGETS = (256, 512, 1024, 2048, 4096, 8192, 12288, 16384)
QUEST_PAGE = 16
FFD_BLOCK = 128
FFD_SUB = (16, 128)
FFD_DELTAS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)
KIVI_GROUP = 32
KIVI_RESIDUAL = 128
BF16_FAMILIES = ("FA-3", "FA-4", "FlashInfer", "XQA", "cuDNN")
FP8_FAMILIES = ("FA-3 FP8", "FlashInfer FP8", "XQA FP8")
NEG = float("-inf")
# a refine depth no logit reaches
CLOSED = -1e4


def v_scale_of(v):
    """An 8-bit V's fixed scale for a layer, from all of the layer's V: the
    power of two that leaves a factor two of headroom."""
    return 2.0 ** math.ceil(math.log2(float(v.abs().max()) * 2 / B.E4M3_MAX))


def members(args, v_scale=None):
    """`{name: FoldKVCache kwargs}`. An 8-bit V takes the layer's fixed scale,
    never the batch's. The capacity members never read a second plane: their
    refine gates can never fire."""
    out = {}
    v8kw = {} if v_scale is None else dict(v_scale=v_scale)
    for v8 in (False, True):
        for depth in (None, *args.depths):
            out[F.name_of(depth, v8)] = dict(depth=depth, v8=v8, **(v8kw if v8 else {}))
    out["Fold capacity"] = dict(depth=None, v8=False, refine_k=CLOSED, refine_v=CLOSED)
    out["Fold capacity v8"] = dict(depth=None, v8=True, refine_k=CLOSED, refine_v=CLOSED, **v8kw)
    return out


def attend(q, k, v, keep, D):
    """FP32 attention of `q` `(B, HKV, G, D)` over `k`, `v` `(B, HKV, S, D)`
    restricted to `keep` (broadcast to `(B, HKV, G, S)`)."""
    s = torch.einsum("bhgd,bhsd->bhgs", q.float(), k.float()) / math.sqrt(D)
    s = s.masked_fill(~keep, NEG)
    return torch.einsum("bhgs,bhsd->bhgd", torch.softmax(s, -1), v.float())


def err(o, ref):
    return B.rel_l2(o, ref)


# ---------------------------------------------------------------- page selection


def quest(q, k, v, n, ref, D, budgets, per_head):
    """Quest's selection and exact attention over it, per token budget."""
    Bz, HKV, S, _ = k.shape
    P = QUEST_PAGE
    NP = -(-S // P)
    pad = NP * P - S
    kp = torch.nn.functional.pad(k.float(), (0, 0, 0, pad)).view(Bz, HKV, NP, P, D)
    pos = torch.arange(NP * P, device=k.device).view(NP, P)
    valid = pos[None] < n[:, None, None]
    big = torch.finfo(torch.float32).max
    vm = valid[:, None, :, :, None]
    kmin = torch.where(vm, kp, big).amin(3)
    kmax = torch.where(vm, kp, -big).amax(3)
    qf = q.float()
    score = torch.maximum(
        qf[:, :, :, None, :] * kmin[:, :, None], qf[:, :, :, None, :] * kmax[:, :, None]
    ).sum(-1)
    npg = (n + P - 1) // P
    pidx = torch.arange(NP, device=k.device)
    real = pidx[None, :] < npg[:, None]
    score = score.masked_fill(~real[:, None, None], NEG)
    if not per_head:
        score = score.amax(2, keepdim=True)
    last = npg - 1
    forced = (pidx[None] == 0) | (pidx[None] == last[:, None])
    score = score.masked_fill(forced[:, None, None], float("inf"))
    tok_valid = torch.arange(S, device=k.device)[None] < n[:, None]
    out = []
    for bud in budgets:
        want = torch.clamp(torch.full_like(npg, bud // P), max=npg)
        order = score.argsort(-1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, pidx.expand_as(order).contiguous())
        sel = rank < want[:, None, None, None]
        sel = sel & real[:, None, None]
        keep = sel.repeat_interleave(P, -1)[..., :S] & tok_valid[:, None, None]
        o = attend(q, k, v, keep, D)
        # the KV head reads the union of its rows' pages, and every page's
        # min and max
        union = sel.any(2)
        tokens = (union.repeat_interleave(P, -1)[..., :S] & tok_valid[:, None]).sum()
        meta = float(npg.sum()) * HKV * 4 * D
        keys = float(n.sum()) * HKV
        out.append(
            dict(
                budget=bud,
                err=err(o, ref),
                bytes_per_key=(meta + float(tokens) * 4 * D) / keys,
                read_fraction=float(tokens) / keys,
            )
        )
    return out


# ---------------------------------------------------------------- top-delta


def ffd_quantise(k, n):
    """FFD's key format per request: symmetric 2-bit codes, one bf16 scale per
    channel per `FFD_BLOCK` keys, an e4m3 residual; `(screen, refined, nq)`
    with keys past each request's last full block left as they are."""
    Bz, HKV, S, D = k.shape
    BS = FFD_BLOCK
    NB = -(-S // BS)
    kp = torch.nn.functional.pad(k.float(), (0, 0, 0, NB * BS - S)).view(Bz, HKV, NB, BS, D)
    scale = (kp.abs().amax(3, keepdim=True) / 1.5).to(torch.bfloat16).float().clamp_min(1e-8)
    code = (kp / scale + 1.5).round().clamp(0, 3)
    deq = (code - 1.5) * scale
    res = (kp - deq).to(torch.float8_e4m3fn).float()
    screen = deq.view(Bz, HKV, NB * BS, D)[:, :, :S]
    refined = (deq + res).view(Bz, HKV, NB * BS, D)[:, :, :S]
    nq = n // BS * BS
    return screen, refined, nq


def ffd(q, k, v, n, ref, D, deltas, subs):
    _, HKV, S, _ = k.shape
    screen, refined, nq = ffd_quantise(k, n)
    BS = FFD_BLOCK
    t = torch.arange(S, device=k.device)
    quant = t[None] < nq[:, None]
    cur = (t[None] >= nq[:, None]) & (t[None] < n[:, None])
    kk = torch.where(quant[:, None, :, None], refined, k.float())
    s_scr = torch.einsum("bhgd,bhsd->bhgs", q.float(), screen) / math.sqrt(D) * LOG2E
    s_scr = s_scr.masked_fill(~quant[:, None, None], NEG)
    first = t[None] < torch.clamp(nq, max=BS)[:, None]
    lastb = (t[None] >= (nq - BS)[:, None]) & quant
    m0 = s_scr.masked_fill(~first[:, None, None], NEG).amax(-1)
    m1 = s_scr.masked_fill(~lastb[:, None, None], NEG).amax(-1)
    peak = torch.maximum(m0, m1)
    keys = float(n.sum()) * HKV
    nqs = float(nq.sum())
    ncur = float((n - nq).sum())
    out = []
    for sb in subs:
        NS = -(-S // sb)
        spad = torch.nn.functional.pad(s_scr, (0, NS * sb - S), value=NEG).view(
            *s_scr.shape[:3], NS, sb
        )
        smax = spad.amax(-1)
        for delta in deltas:
            hit = (smax >= (peak - delta)[..., None]).any(2)
            keep_q = hit.repeat_interleave(sb, -1)[..., :S] & quant[:, None]
            keep = (keep_q | cur[:, None])[:, :, None]
            o = attend(q, kk, v, keep, D)
            # the same selection over exact keys: what the drop alone costs
            o_sel = attend(q, k, v, keep, D)
            kept = float(keep_q.sum())
            # screen and scales for every quantised key, plus the two blocks
            # the threshold rereads; residual and V for the kept keys; the
            # partial block in bf16
            nblk = float((nq // BS).sum())
            by = (
                HKV * (nqs * D / 4 + nblk * D * 2 + float((nq > 0).sum()) * 2 * BS * D / 4)
                + kept * 3 * D
                + HKV * ncur * 4 * D
            )
            out.append(
                dict(
                    sub_block=sb,
                    delta=delta,
                    err=err(o, ref),
                    err_selection_only=err(o_sel, ref),
                    bytes_per_key=by / keys,
                    read_fraction=(kept + HKV * ncur) / keys,
                )
            )
    return out


# ---------------------------------------------------------------- quantised caches


def _asym(x, bits, dim):
    """Asymmetric quantisation over `dim` with fp16 scale and minimum."""
    mn = x.amin(dim, keepdim=True).half().float()
    mx = x.amax(dim, keepdim=True)
    sc = ((mx - mn) / (2**bits - 1)).half().float().clamp_min(1e-8)
    return ((x - mn) / sc).round().clamp(0, 2**bits - 1) * sc + mn


def kivi(k, n, v, bits):
    """KIVI's cache per request: K per channel over `KIVI_GROUP`-token groups,
    V per token over 32-channel groups, the newest tokens in bf16."""
    Bz, HKV, _, D = k.shape
    kq, vq = k.float().clone(), v.float().clone()
    nqs = []
    for b in range(Bz):
        nb = int(n[b])
        nqb = max(0, nb - KIVI_RESIDUAL) // KIVI_GROUP * KIVI_GROUP
        nqs.append(nqb)
        if nqb:
            kb = k[b, :, :nqb].float().view(HKV, nqb // KIVI_GROUP, KIVI_GROUP, D)
            kq[b, :, :nqb] = _asym(kb, bits, 2).view(HKV, nqb, D)
            vb = v[b, :, :nqb].float().view(HKV, nqb, D // 32, 32)
            vq[b, :, :nqb] = _asym(vb, bits, 3).view(HKV, nqb, D)
    nq = sum(nqs)
    # codes for K and V, fp16 scale and minimum per channel per group for K
    # and per 32 channels per token for V; the rest in bf16
    by = HKV * (
        nq * (2 * bits * D / 8 + 4 * D / KIVI_GROUP + 4 * D / 32) + (float(n.sum()) - nq) * 4 * D
    )
    return kq, vq, by


def int8_token(x):
    s = (x.float().abs().amax(-1, keepdim=True) / 127).half().float().clamp_min(1e-30)
    return (x.float() / s).round().clamp(-127, 127) * s


def e4m3(x, per_head):
    dims = (2, 3) if per_head else (0, 1, 2, 3)
    s = (x.float().abs().amax(dim=dims, keepdim=True) / B.E4M3_MAX).clamp_min(1e-30)
    return (x.float() / s).clamp(-B.E4M3_MAX, B.E4M3_MAX).to(torch.float8_e4m3fn).float() * s


def quantised(q, k, v, n, ref, D):
    _, HKV, S, _ = k.shape
    keep = (torch.arange(S, device=k.device)[None] < n[:, None])[:, None, None]
    keys = float(n.sum()) * HKV
    out = []
    for bits in (2, 4):
        kq, vq, by = kivi(k, n, v, bits)
        out.append(
            dict(
                method=f"KIVI-{bits}",
                err=err(attend(q, kq, vq, keep, D), ref),
                bytes_per_key=by / keys,
            )
        )
    out.append(
        dict(
            method="INT8 per token",
            err=err(attend(q, int8_token(k), int8_token(v), keep, D), ref),
            bytes_per_key=2 * D + 4,
        )
    )
    for ph in (False, True):
        kq, vq = e4m3(k, ph), e4m3(v, ph)
        tag = "per head" if ph else "per tensor"
        out.append(
            dict(
                method=f"FP8 e4m3 {tag}",
                err=err(attend(q, kq, vq, keep, D), ref),
                bytes_per_key=2 * D,
            )
        )
        # the FP8 kernels also take Q in e4m3, one scale per (request, KV head)
        qs = q.float().abs().amax(dim=(2, 3), keepdim=True) / B.E4M3_MAX
        q8 = (q.float() / qs).to(torch.float8_e4m3fn).float() * qs
        out.append(
            dict(
                method=f"FP8 e4m3 {tag}, Q e4m3",
                err=err(attend(q8, kq, vq, keep, D), ref),
                bytes_per_key=2 * D,
            )
        )
    return out


# ---------------------------------------------------------------- Fold's gates


def running_max(s, n, split):
    """Per row, the largest logit its split has seen through each key's tile,
    where a split is a contiguous chunk of whole tiles scanned in order, as a
    split-KV online-softmax kernel runs it; `s` `(B, HKV, G, S)`."""
    Bz, HKV, G, S = s.shape
    NT = -(-S // BN)
    tiles = (
        torch.nn.functional.pad(s, (0, NT * BN - S), value=NEG).view(Bz, HKV, G, NT, BN).amax(-1)
    )
    m = torch.full_like(tiles, NEG)
    for b in range(Bz):
        nb = int(n[b])
        per = -(-nb // split)
        chunk = -(-per // BN) * BN
        for lo in range(0, nb, chunk):
            t0, t1 = lo // BN, -(-min(lo + chunk, nb) // BN)
            m[b, :, :, t0:t1] = torch.cummax(tiles[b, :, :, t0:t1], -1).values
    return m.repeat_interleave(BN, -1)[..., :S]


def gate_counts(rel, n, rk, rv, depth):
    """Kernel verdicts from logits relative to a row reference: live (some
    row clears the cut), refine K and refine V (the group's largest relative
    logit clears the gate), counted over live keys."""
    S = rel.shape[-1]
    dg = rel.amax(2)
    valid = (torch.arange(S, device=rel.device)[None] < n[:, None])[:, None].expand_as(dg)
    live = valid if depth is None else (rel >= -depth).any(2) & valid
    refk = (dg >= -rk[:, None, None]) & live
    refv = (dg >= -rv[:, None, None]) & live
    keys = float(valid.sum())
    return dict(
        live=float(live.sum()) / keys,
        refined=float(refk.sum()) / keys,
        refined_v=float(refv.sum()) / keys,
    )


CUT_MEMBERS = ("Fold T=16", "Fold T=14", "Fold T=12")


def cut_mass(s, z, sref, n, depth):
    """Per query row, the share of its mass on the keys the kernel cuts: keys
    no row of the group keeps (`s - Z < -depth` in every row, the group
    truncation the tail build uses), whose V the block rows stand in for.
    `true` weighs keys by their FP32 logits `sref`; `coarse` by plane-A
    logits `s`, the weights the kernel sums for a cut key; `coarse_over_true`
    divides the cut keys' plane-A mass by the row's FP32 mass, which is the
    cut's share of the kernel's own denominator up to the kept keys' logit
    error. Each `(rows,)`, base-2 logits in, fractions out."""
    S = s.shape[-1]
    valid = (torch.arange(S, device=s.device)[None] < n[:, None])[:, None, None]
    cut = ~((s - z >= -depth).any(2, keepdim=True)) & valid
    ln2 = math.log(2.0)

    def lse(x, m):
        return torch.logsumexp(x.masked_fill(~m, NEG) * ln2, -1)

    t_all, t_cut = lse(sref, valid.expand_as(sref)), lse(sref, cut.expand_as(sref))
    c_all, c_cut = lse(s, valid.expand_as(s)), lse(s, cut.expand_as(s))
    return dict(
        true=torch.exp(t_cut - t_all).flatten(),
        coarse=torch.exp(c_cut - c_all).flatten(),
        coarse_over_true=torch.exp(c_cut - t_all).flatten(),
        cut_fraction=float(cut.sum()) / float(valid.expand_as(cut).sum()),
    )


def cut_stats(rows):
    """Pooled statistics of per-row cut masses, `rows` as `cut_mass` returns."""
    t = torch.cat([r["true"] for r in rows]).double()
    c = torch.cat([r["coarse"] for r in rows]).double()
    ct = torch.cat([r["coarse_over_true"] for r in rows]).double()

    def q(x):
        return dict(
            median=float(x.median()),
            p90=float(x.quantile(0.9)),
            p99=float(x.quantile(0.99)),
            max=float(x.max()),
        )

    ok = t > 0
    ratio = c[ok] / t[ok]
    return dict(
        rows=int(t.numel()),
        rows_with_cut_mass=int(ok.sum()),
        true=q(t),
        coarse=q(c),
        coarse_over_true=q(ct),
        ratio_coarse_to_true=dict(
            median=float(ratio.median()),
            min=float(ratio.min()),
            max=float(ratio.max()),
            p01=float(ratio.quantile(0.01)),
            p99=float(ratio.quantile(0.99)),
        )
        if ok.any()
        else None,
    )


def fold_bytes(frac, D, v8):
    v = D if v8 else 2 * D
    by = D + 2 + frac["refined"] * D + frac["live"] * v
    if v8:
        by += frac.get("refined_v", 0.0) * D
    return by


# ---------------------------------------------------------------- one case


def run_case(name, args, rep, flusher, cut_rows):
    cap, Bz, lo, hi = CASES[name]
    qc, kc, vc = C.raw(cap)
    H, _, D = qc.shape
    HKV = kc.shape[0]
    G = H // HKV
    steps = args.steps
    g = torch.Generator().manual_seed(args.seed)
    lens = torch.randint(lo, hi + 1, (Bz,), generator=g).tolist()
    need = max(lens) + steps
    dev = "cuda"
    q = qc[:, :need].transpose(0, 1).contiguous().to(dev).bfloat16()
    k = kc[:, :need].transpose(0, 1).contiguous().to(dev).bfloat16()
    v = vc[:, :need].transpose(0, 1).contiguous().to(dev).bfloat16()
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device=dev, dtype=torch.int32)
    pk = torch.cat([k[:n] for n in lens])
    pv = torch.cat([v[:n] for n in lens])
    fin = [n + steps for n in lens]
    n = torch.tensor(fin, device=dev)
    idx = [x - 1 for x in fin]
    qn = q[idx]
    Smax = -(-need // B.PAGE) * B.PAGE
    bk = torch.zeros(Bz, HKV, Smax, D, device=dev, dtype=torch.bfloat16)
    bv = torch.zeros_like(bk)
    for b, x in enumerate(fin):
        bk[b, :, :x] = k[:x].transpose(0, 1)
        bv[b, :, :x] = v[:x].transpose(0, 1)
    q4 = qn.view(Bz, HKV, G, D)
    ref = B.reference(qn, bk, bv, n, 1.0 / math.sqrt(D))
    keys = float(n.sum()) * HKV
    print(
        f"\n=== {name}: {cap} H={H} H_KV={HKV} D={D} B={Bz} final {min(fin)}..{max(fin)} ===",
        flush=True,
    )

    specs = B.build(qn, bk, bv, n.int(), families=[*BF16_FAMILIES, *FP8_FAMILIES])
    won = B.tune(specs, ref, rounds=args.tune_rounds, flusher=flusher)
    kernels = {
        f: dict(
            arm=w.spec.name, err=w.err, fp8=w.spec.fp8, bytes_per_key=(2 if w.spec.fp8 else 4) * D
        )
        for f, w in won.items()
    }
    del specs, won
    torch.cuda.empty_cache()
    band = [kernels[f]["err"] for f in kernels if not kernels[f]["fp8"]]
    band = dict(lo=min(band), hi=max(band)) if band else None

    points = []
    for f, x in kernels.items():
        points.append(
            dict(
                method=f"{f} kernel",
                family="FP8 kernel" if x["fp8"] else "BF16 kernel",
                err=x["err"],
                bytes_per_key=x["bytes_per_key"],
            )
        )

    Sk = int(n.max())
    kq, vq = bk[:, :, :Sk], bv[:, :, :Sk]
    for per_head in (False, True):
        for r in quest(q4, kq, vq, n, ref, D, args.quest_budgets, per_head):
            points.append(
                dict(
                    method="Quest" + (" per head" if per_head else ""),
                    family="Quest",
                    param=r["budget"],
                    **{x: r[x] for x in ("err", "bytes_per_key", "read_fraction")},
                )
            )
    for r in ffd(q4, kq, vq, n, ref, D, args.ffd_deltas, FFD_SUB):
        points.append(
            dict(
                method=f"FFD sub-block {r['sub_block']}",
                family="FFD",
                param=r["delta"],
                **{
                    x: r[x] for x in ("err", "err_selection_only", "bytes_per_key", "read_fraction")
                },
            )
        )
    for r in quantised(q4, kq, vq, n, ref, D):
        points.append(dict(family="quantised cache", **r))
    torch.cuda.empty_cache()

    gate = {}
    cm = {}
    sref = torch.einsum("bhgd,bhsd->bhgs", q4.float(), kq.float()) / math.sqrt(D) * LOG2E
    vs = v_scale_of(vc)
    for mname, kw in members(args, vs).items():
        try:
            att = FoldKVCache(Bz, H, HKV, D, need + 64, page_size=B.PAGE, **kw)
            att.write_prompt(pk, pv, cu, lens)
            for t in range(steps):
                ii = [x + t for x in lens]
                att.decode(q[ii], k[ii], v[ii])
            run = att.prepare_replay_decode(out_dtype=torch.bfloat16)
            out, _, counts = run()
            e = err(out.reshape(Bz, HKV, G, D).float(), ref)
            fr = F.fractions(counts, fin, HKV)
            s = att.coarse_logits()
            rk, rv = (x.to(dev) for x in att.refine_gates())
            z = att.z.view(Bz, HKV, G)[..., None]
            emu = gate_counts(s - z, n, rk, rv, kw["depth"])
            fr["refined_v"] = emu["refined_v"] if kw["v8"] else 0.0
            split = run.config.split
            by = fold_bytes(fr, D, kw["v8"])
            points.append(
                dict(
                    method=mname,
                    family="FoldAttention",
                    err=e,
                    bytes_per_key=by,
                    live=fr["live"],
                    refined=fr["refined"],
                    refined_v=fr["refined_v"],
                    split=split,
                    bytes_counted=F.bytes_read(dict(fr, v8=kw["v8"]), keys, D) / keys,
                )
            )
            row: dict[str, Any] = dict(
                split=split,
                kernel=dict(live=fr["live"], refined=fr["refined"]),
                z_emulated=emu,
                rk=rk.tolist(),
                rv=rv.tolist(),
            )
            if not mname.startswith("Fold capacity"):
                for sp, tag in ((split, "running_max"), (1, "running_max_split1")):
                    m = running_max(s, n, sp)
                    rm = gate_counts(s - m, n, rk, rv, kw["depth"])
                    rm["bytes_per_key"] = fold_bytes(
                        dict(rm, refined_v=rm["refined_v"] if kw["v8"] else 0.0), D, kw["v8"]
                    )
                    row[tag] = rm
                row["z_bytes_per_key"] = by
                # the plane-A logit against the FP32 one, to check the read-back
                ok = torch.isfinite(s)
                row["plane_a_vs_fp32_max_abs"] = float((s[ok] - sref[ok]).abs().max())
            gate[mname] = row
            if mname in CUT_MEMBERS:
                x = cut_mass(s, z, sref, n, kw["depth"])
                cut_rows.setdefault(mname, []).append(x)
                cm[mname] = dict(
                    cut_stats([x]), cut_fraction=x["cut_fraction"], live_kernel=fr["live"]
                )
                print(
                    f"  {mname:18s} cut mass true median {cm[mname]['true']['median']:.2e} "
                    f"max {cm[mname]['true']['max']:.2e}; coarse/true median "
                    f"{cm[mname]['ratio_coarse_to_true']['median']:.3f}",
                    flush=True,
                )
            print(
                f"  {mname:18s} err {e:.3e}  {by:7.1f} B/key  live {fr['live']:.3f}  "
                f"refined {fr['refined']:.3f} (emulated {emu['refined']:.3f})"
                + (
                    f"  running-max refined {row['running_max']['refined']:.3f} "
                    f"live {row['running_max']['live']:.3f}"
                    if "running_max" in row
                    else ""
                ),
                flush=True,
            )
            del att, run, s
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            gate[mname] = dict(error=repr(ex)[:300])
        torch.cuda.empty_cache()

    for p in points:
        if p["family"] not in ("FoldAttention",):
            print(
                f"  {p['method']:30s} {p.get('param', '')!s:>6s} err {p['err']:.3e}  "
                f"{p['bytes_per_key']:7.1f} B/key",
                flush=True,
            )
    rep.add(
        case=name,
        capture=cap,
        fingerprint=C.fingerprint(cap),
        v_scale=vs,
        H=H,
        HKV=HKV,
        D=D,
        B=Bz,
        lens=lens,
        steps=steps,
        bf16_band=band,
        kernels=kernels,
        points=points,
        gate=gate,
        cut_mass=cm,
        pareto=pareto(points, band),
    )
    del bk, bv, kq, vq
    torch.cuda.empty_cache()


def pareto(points, band):
    """Per method, the fewest bytes per key at which its error is within the
    BF16 band (at or below its least accurate kernel) and at or below its
    most accurate kernel, or None where no setting gets there."""
    if band is None:
        return None
    by_m = {}
    for p in points:
        by_m.setdefault(p["method"], []).append(p)
    out = {}
    for m, ps in by_m.items():

        def least(limit, ps=ps):
            ok = [p for p in ps if p["err"] <= limit]
            if not ok:
                return None
            best = min(ok, key=lambda p: p["bytes_per_key"])
            return dict(
                bytes_per_key=best["bytes_per_key"], err=best["err"], param=best.get("param")
            )

        out[m] = dict(
            band=least(band["hi"]), strict=least(band["lo"]), best_err=min(p["err"] for p in ps)
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--depths", type=float, nargs="+", default=list(DEPTHS))
    p.add_argument("--quest-budgets", type=int, nargs="+", default=list(QUEST_BUDGETS))
    p.add_argument("--ffd-deltas", type=float, nargs="+", default=list(FFD_DELTAS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tune-rounds", type=int, default=9)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rep = Report(
        args.out,
        cases={c: CASES[c] for c in args.cases},
        steps=args.steps,
        depths=list(args.depths),
        quest=dict(page=QUEST_PAGE, budgets=list(args.quest_budgets), metadata="bf16 min+max"),
        ffd=dict(
            block=FFD_BLOCK,
            sub_blocks=list(FFD_SUB),
            deltas=list(args.ffd_deltas),
            source="qluoluo/faster-flash-decoding ffd_core (paged_decode_kernel, "
            "quantized_cache.quantize_symmetric_blocks)",
        ),
        kivi=dict(group=KIVI_GROUP, residual=KIVI_RESIDUAL),
        byte_model="bytes read from the cache per key and KV head, metadata included",
        members=members(args),
    )
    flusher = L2Flush()
    cut_rows = {}
    for c in args.cases:
        try:
            run_case(c, args, rep, flusher, cut_rows)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rep.add(case=c, error=repr(e)[:400])
            torch.cuda.empty_cache()
    rep.meta["cut_mass"] = {m: cut_stats(r) for m, r in cut_rows.items()}
    rep.write()


if __name__ == "__main__":
    main()
