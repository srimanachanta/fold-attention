"""How much a perfect reference would buy.

The mass reference estimates each row's log-sum-exp from 192 keys scored
with plane A. The oracle is the exact log-sum-exp of the row's FP32 scores,
which no decode can know before it has read every key. Both drive the same
kernel-level call with the same refine gates, weight terms, cut model and
depths; only `Z` differs. The oracle's time excludes computing its `Z`, the
mass arm's includes its prepass.

Every cell sweeps depth, so each reference traces bytes and time against
error. For each mass point, `oracle_bytes` and `oracle_us` interpolate the
oracle's curve at that point's error (linear in bytes and time against log
error, between adjacent depths): what the oracle needs for the same accuracy.

    python -m benchmarks.oracle --out oracle
"""

from __future__ import annotations

import argparse
import itertools
import math
import traceback

import torch

from benchmarks.harness import baselines as B
from benchmarks.harness import captures as C
from benchmarks.harness.report import Report
from benchmarks.harness.timing import L2Flush, measure
from fold_attention.decode import prepare_fold_decode, quantize_k, quantize_q
from fold_attention.decode.heuristics import refine_for, weight_terms_for

LOG2E = 1.4426950408889634
# (D, G, capture, H_KV) at 256 row groups
CELLS = {
    "qwen D128 G8": (128, 8, "qwen3-30b-l24-d128", 4),
    "qwen D128 G4": (128, 4, "qwen3-30b-l24-d128", 4),
    "glm D128 G16": (128, 16, "glm4-9b-l28-d128", 2),
    "gptoss D64 G8": (64, 8, "gptoss-20b-l9-d64", 8),
}
DEPTHS = (None, 20, 18, 16, 15, 14, 13, 12, 11, 10)
# a depth past any logit's range: nothing is cut
NONE = 1e4


def exact_lse(q, k, lens, D):
    """Each row's log2-sum-exp2 of its FP32 scores in the kernel's units,
    `(NBH, G)`."""
    Bz, H, _ = q.shape
    HKV = k.shape[1]
    G = H // HKV
    z = torch.empty(Bz, HKV, G, dtype=torch.float32, device=q.device)
    qf = q.reshape(Bz, HKV, G, D).float() * (LOG2E / math.sqrt(D))
    for b in range(Bz):
        n = int(lens[b])
        s = qf[b] @ k[b, :, :n].float().transpose(1, 2)
        z[b] = torch.logsumexp(s * math.log(2.0), -1) / math.log(2.0)
    return z.reshape(Bz * HKV, G)


def interp(err, pts):
    """The oracle's (bytes, us) at `err`, linear in log error between the two
    oracle points that bracket it; None outside the oracle's range."""
    pts = sorted(pts, key=lambda p: p[0])
    x = math.log(err)
    for (e0, b0, t0), (e1, b1, t1) in itertools.pairwise(pts):
        l0, l1 = math.log(e0), math.log(e1)
        if l0 <= x <= l1:
            w = 0.0 if l1 == l0 else (x - l0) / (l1 - l0)
            return b0 + w * (b1 - b0), t0 + w * (t1 - t0)
    return None


def run_cell(name, S, args, rep, flusher):
    D, G, cap, HKV = CELLS[name]
    Bz = 256 // HKV
    NBH = Bz * HKV
    lens = torch.full((Bz,), S, dtype=torch.int32, device="cuda")
    tag = f"{name} S{S}"
    print(f"\n=== {tag} ===", flush=True)
    shape = C.decode_shape(cap, B=Bz, S=S, G=G, HKV=HKV)
    q, k, v = shape["q"], shape["k"], shape["v"]
    ref = B.reference(q, k, v, lens, 1.0 / math.sqrt(D))
    qa, qb, eq = quantize_q((q.float() * (LOG2E / math.sqrt(D))).reshape(NBH, G, D).contiguous())
    ka, kb, ek = quantize_k(k.reshape(NBH, S, D).float())
    vf = v.reshape(NBH, S, D).contiguous()
    vmean = vf.float().mean(1)
    z_or = exact_lse(q, k, lens, D)
    keys = NBH * S

    arms, meta, held = {}, {}, []
    for depth in DEPTHS:
        rk, rv = refine_for(depth, seq_len=S, head_dim=D, group=G)
        knobs = dict(
            truncate=depth is not None,
            refine_k=rk,
            refine_v=rv,
            weight_terms=weight_terms_for(depth),
            group_cut="auto",
            vmean=vmean if depth is not None else None,
            out_dtype=torch.bfloat16,
        )
        dd = NONE if depth is None else float(depth)
        z_m = torch.empty((NBH, G), device="cuda", dtype=torch.float32)
        cut_or = z_or - dd
        runs = {
            "mass": prepare_fold_decode(
                qa,
                qb,
                eq,
                ka,
                kb,
                ek,
                vf,
                z_m,
                torch.full_like(z_m, dd),
                reference="mass",
                **knobs,
            ),
            "oracle": prepare_fold_decode(
                qa,
                qb,
                eq,
                ka,
                kb,
                ek,
                vf,
                z_or,
                cut_or,
                reference="given",
                **knobs,
            ),
        }
        for rname, run in runs.items():
            out, _, counts = run()
            c = counts.float()
            live = float(c[:, 0].sum()) / keys
            refined = float(c[:, 1].sum()) / keys
            label = f"{rname} {'dense' if depth is None else f'T={depth}'}"
            arms[label] = run
            meta[label] = dict(
                reference=rname,
                depth=depth,
                err=B.rel_l2(out.view(Bz, HKV, G, D), ref),
                live=live,
                refined=refined,
                bytes_per_key=D + 2 + refined * D + live * 2 * D,
            )
            print(
                f"  {label:14s} err {meta[label]['err']:.3e} live {live:.3f} refined {refined:.3f}",
                flush=True,
            )
            held.append(run)
        held += [z_m, cut_or]

    res = measure(arms, rounds=args.rounds, cold=True, flusher=flusher)
    for n in arms:
        meta[n]["cold"] = res[n]
        meta[n]["us"] = res[n]["us"]
    pts = [
        (meta[n]["err"], meta[n]["bytes_per_key"], meta[n]["us"])
        for n in arms
        if meta[n]["reference"] == "oracle"
    ]
    rows = []
    for n in arms:
        row = dict(arm=n, **meta[n])
        if meta[n]["reference"] == "mass":
            got = interp(meta[n]["err"], pts)
            if got is not None:
                row["oracle_bytes"], row["oracle_us"] = got
                row["bytes_ratio"] = got[0] / meta[n]["bytes_per_key"]
                row["us_ratio"] = got[1] / meta[n]["us"]
        rows.append(row)
        print(
            f"  {n:14s} {meta[n]['us']:8.1f} us  {meta[n]['bytes_per_key']:6.1f} B/key"
            + (
                f"  oracle at same error {row['bytes_ratio']:.3f}x bytes "
                f"{row['us_ratio']:.3f}x time"
                if "bytes_ratio" in row
                else ""
            ),
            flush=True,
        )
    rep.add(
        cell=dict(name=name, D=D, G=G, S=S, B=Bz, HKV=HKV),
        tag=tag,
        capture=cap,
        fingerprint=shape["fingerprint"],
        keys=keys,
        arms=rows,
    )
    del arms, held
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="+", default=list(CELLS), choices=list(CELLS))
    p.add_argument("--contexts", type=int, nargs="+", default=[4096, 16384])
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rep = Report(args.out, cells=args.cells, contexts=args.contexts, depths=list(DEPTHS))
    flusher = L2Flush()
    for name in args.cells:
        for S in args.contexts:
            try:
                run_cell(name, S, args, rep, flusher)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rep.add(cell=dict(name=name, S=S), error=repr(e)[:400])
                torch.cuda.empty_cache()
    rep.write()


if __name__ == "__main__":
    main()
