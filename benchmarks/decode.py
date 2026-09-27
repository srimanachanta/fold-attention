"""Decode latency and accuracy against every baseline, per shape.

A cell is a (head dim, query group, context, row groups, uniform or ragged)
point over a post-RoPE capture. In each cell:

1. The capture is cut to the shape and one FP32 reference is computed, which
   every arm is scored against.
2. Every baseline family is tuned at the shape (`baselines.tune`), BF16 and
   FP8 alike.
3. FoldAttention is built dense and at each truncation depth, with bf16 V or
   (`--v8`) 8-bit V. A member whose error is over every accuracy budget is
   recorded untimed and freed.
4. The tuned winners and our members are timed together, cold L2 then warm,
   at the decode boundary and at the step boundary: the new token's write
   plus the attention. A baseline's step is its fastest append (its own
   fused one where it has it) ahead of its tuned decode; ours is the front
   kernel ahead of the decode and combine.

Accuracy budgets, all recorded: `band` is the largest BF16 baseline error,
the headline rule (no more error than a production BF16 kernel on the same
workload); `strict` the most accurate BF16 baseline's; `ref` FA-4's (FA-3's
if FA-4 is absent); `fastest` the fastest BF16 baseline's. A speed tie between
two baselines of different error would move `fastest`, which is why no claim
rests on it.

    python -m benchmarks.decode --out decode_context_bf16
    python -m benchmarks.decode --v8 --out decode_context_v8
    python -m benchmarks.decode --row-groups 32 64 128 512 --contexts 4096 16384 \\
        --out decode_batch_bf16
"""

from __future__ import annotations

import argparse
import math
import traceback

import torch

from benchmarks.harness import baselines as B
from benchmarks.harness import captures as C
from benchmarks.harness import fold as F
from benchmarks.harness.ceiling import read_ceiling
from benchmarks.harness.report import Report
from benchmarks.harness.timing import L2Flush, measure, paired_ratio

DEPTHS = (10, 12, 13, 14, 16, 18, 20)

# (head dim, query group) -> (capture, H_KV). Each takes the capture's own
# heads, so no query row repeats; G=4 and G=1 take the first rows of each
# captured group.
SOURCES = {
    (128, 16): ("glm4-9b-l28-d128", 2),
    (128, 8): ("qwen3-30b-l24-d128", 4),
    (128, 4): ("qwen3-30b-l24-d128", 4),
    (128, 1): ("qwen3-30b-l24-d128", 4),
    (64, 8): ("gptoss-20b-l9-d64", 8),
    (64, 4): ("gptoss-20b-l9-d64", 8),
    (64, 1): ("gptoss-20b-l9-d64", 8),
}


def _excess(err, floor):
    """The arm's error beyond the bf16 rounding of the exact answer, which
    every arm pays; errors add in quadrature."""
    if err is None or math.isnan(err):
        return None
    return math.sqrt(max(0.0, err * err - floor * floor))


FIXED_DEPTHS = (14.0, 16.0)


def cells(args):
    out = []
    for D in args.dims:
        for G in args.groups:
            if (D, G) not in SOURCES:
                continue
            cap, HKV = SOURCES[(D, G)]
            for rg in args.row_groups:
                for S in args.contexts:
                    for ragged in args.ragged:
                        out.append(
                            dict(
                                D=D,
                                G=G,
                                S=S,
                                ragged=bool(ragged),
                                row_groups=rg,
                                B=max(1, rg // HKV),
                                HKV=HKV,
                                capture=cap,
                                v8=args.v8,
                            )
                        )
    return out


def run_cell(cell, args, rep, flusher, ceiling):
    D, G, S, Bz, HKV = cell["D"], cell["G"], cell["S"], cell["B"], cell["HKV"]
    tag = (
        f"D{D} G{G} S{S} B{Bz} HKV{HKV} {'ragged' if cell['ragged'] else 'uniform'}"
        f" {'v8' if cell['v8'] else 'bf16'}"
    )
    print(f"\n=== {tag} :: {cell['capture']} ===", flush=True)
    lens = (
        C.ragged_lens(Bz, S, page=B.PAGE)
        if cell["ragged"]
        else torch.full((Bz,), S, dtype=torch.int32, device="cuda")
    )
    shape = C.decode_shape(cell["capture"], B=Bz, S=S, G=G, HKV=HKV, lens=lens.tolist())
    keys = int(lens.sum()) * HKV
    ref = B.reference(shape["q"], shape["k"], shape["v"], lens, 1.0 / math.sqrt(D))
    floor = B.rel_l2(ref.to(torch.bfloat16).float(), ref)
    print(f"  bf16 output floor {floor:.3e}", flush=True)

    pr = F.prompt(shape, lens)
    lay = B.paged_layout(shape["k"], shape["v"], lens)
    specs = B.build(
        shape["q"], shape["k"], shape["v"], lens, layout=lay, new_kv=(pr["kn"], pr["vn"])
    )
    won = B.tune(specs, ref, rounds=args.tune_rounds, flusher=flusher)
    step_fns = B.step_arms(won, rounds=args.tune_rounds, flusher=flusher)
    bf16 = {f: w for f, w in won.items() if not w.spec.fp8}
    if not bf16:
        print("  no BF16 baseline live; skipping", flush=True)
        return
    base_err = {f: w.err for f, w in bf16.items()}
    fastest_family = min(bf16, key=lambda f: bf16[f].us)
    ref_family = next((f for f in ("FA-4", "FA-3") if f in base_err), fastest_family)
    budget = dict(
        ref=base_err[ref_family],
        strict=min(base_err.values()),
        fastest=base_err[fastest_family],
        band=max(base_err.values()),
    )
    widest = max(budget.values())
    print(
        "  budgets "
        + "  ".join(f"{k} {v:.3e}" for k, v in budget.items())
        + f"  (ref = {ref_family})",
        flush=True,
    )

    members, untimed = {}, []
    builds = [dict(depth=depth) for depth in (None, *args.depths)]
    builds.append(dict(depth=None, refine_k=-1e4, refine_v=-1e4, name="Fold capacity"))
    for kw in builds:
        depth = kw["depth"]
        m = F.member(shape, lens, pr, v8=cell["v8"], page=B.PAGE, **kw)
        m["err"] = B.rel_l2(m["out"], ref)
        # the fixed members are timed in every cell, so their range covers
        # the cells where they miss a budget too
        if depth is None or depth in FIXED_DEPTHS or m["err"] <= widest:
            members[m["name"]] = m
        else:
            untimed.append(m)
            for k in ("fa", "decode", "step"):
                del m[k]
            torch.cuda.empty_cache()
        print(
            f"  built {m['name']:14s} err {m['err']:.3e} live {m['live']:.3f} "
            f"refined {m['refined']:.3f} split {m['split']}"
            + ("" if m["name"] in members else "  (over budget: untimed)"),
            flush=True,
        )

    arms, graphable, meta = {}, {}, {}
    for w in won.values():
        arms[w.spec.name] = w.spec.fn
        graphable[w.spec.name] = w.spec.graphable
        meta[w.spec.name] = dict(
            family=w.spec.family,
            provenance=w.spec.provenance,
            paged=w.spec.paged,
            fp8=w.spec.fp8,
            err=w.err,
            boundary="decode",
            bytes=keys * (2 if w.spec.fp8 else 4) * D,
        )
    for family, (sname, fn, prov) in step_fns.items():
        w = won[family]
        arms[sname] = fn
        graphable[sname] = w.spec.graphable
        meta[sname] = dict(
            family=family,
            provenance=prov,
            paged=w.spec.paged,
            fp8=w.spec.fp8,
            err=w.err,
            boundary="step",
        )
    for n, m in members.items():
        for boundary in ("decode", "step"):
            key = n if boundary == "decode" else f"{n} step"
            arms[key] = m[boundary]
            meta[key] = dict(
                family="FoldAttention",
                ours=True,
                provenance=m["provenance"],
                paged=True,
                fp8=False,
                err=m["err"],
                boundary=boundary,
                depth=m["depth"],
                v8=m["v8"],
                live=m["live"],
                refined=m["refined"],
                split=m["split"],
                kernel=m["kernel"],
                bytes=F.bytes_read(m, keys, D) if boundary == "decode" else None,
            )

    cold = measure(arms, graphable=graphable, rounds=args.rounds, cold=True, flusher=flusher)
    warm = (
        {} if args.cold_only else measure(arms, graphable=graphable, rounds=args.rounds, cold=False)
    )

    fb = bf16[fastest_family].spec.name
    fa4 = bf16["FA-4"].spec.name if "FA-4" in bf16 else None
    print(f"  fastest BF16: {fb} {cold[fb]['us']:.1f} us", flush=True)
    bsteps = [step_fns[f][0] for f in step_fns if f in bf16]
    fbs = min(bsteps, key=lambda n: cold[n]["us"]) if bsteps else None
    rows = []
    for n in arms:
        mt = meta[n]
        err = mt["err"]
        passes = {k: err <= v for k, v in budget.items()}
        vs_fastest = paired_ratio(cold, fb, n)
        vs_step = paired_ratio(cold, fbs, n) if fbs and mt["boundary"] == "step" else None
        row = dict(
            arm=n,
            ours=mt.get("ours", False),
            **{k: v for k, v in mt.items() if k != "ours"},
            excess_over_floor=_excess(err, floor),
            passes=passes,
            cold=cold[n],
            warm=warm.get(n),
            vs_fastest=vs_fastest,
            vs_fastest_step=vs_step,
            vs_fa4=paired_ratio(cold, fa4, n) if fa4 else None,
        )
        if mt.get("bytes"):
            gbs = mt["bytes"] / (cold[n]["us"] * 1e-6) / 1e9
            row.update(gbs=gbs, pct_ceiling=100 * gbs / ceiling["gbs"])
        rows.append(row)
        if mt.get("ours"):
            gate = "".join(k[0].upper() for k, ok in passes.items() if ok) or "over"
            print(
                f"  {n:22s} {cold[n]['us']:8.1f} us  err {err:.3e}  "
                f"{vs_fastest['ratio']:.3f}x vs {fb}  [{gate}]",
                flush=True,
            )
    for m in untimed:
        rows.append(
            dict(
                arm=m["name"],
                ours=True,
                family="FoldAttention",
                provenance=m["provenance"],
                boundary="decode",
                err=m["err"],
                excess_over_floor=_excess(m["err"], floor),
                passes={k: False for k in budget},
                depth=m["depth"],
                v8=m["v8"],
                live=m["live"],
                refined=m["refined"],
                split=m["split"],
                kernel=m["kernel"],
                cold=None,
                warm=None,
            )
        )
    rep.add(
        cell=cell,
        tag=tag,
        capture=cell["capture"],
        fingerprint=shape["fingerprint"],
        proxy=shape["proxy"],
        lens=[int(x) for x in lens],
        keys=keys,
        floor_bf16=floor,
        budgets=budget,
        budget_reference=ref_family,
        fastest_baseline=fb,
        fastest_baseline_step=fbs,
        fastest_family=fastest_family,
        tuned={
            f: dict(
                arm=w.spec.name, us=w.us, err=w.err, fp8=w.spec.fp8, alternatives=w.alternatives
            )
            for f, w in won.items()
        },
        arms=rows,
    )
    del members, specs, won, arms, step_fns
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=int, nargs="+", default=[128, 64])
    p.add_argument("--groups", type=int, nargs="+", default=[16, 8, 4, 1])
    p.add_argument(
        "--contexts", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768]
    )
    p.add_argument("--ragged", type=int, nargs="+", default=[0, 1])
    p.add_argument(
        "--row-groups",
        type=int,
        nargs="+",
        default=[256],
        help="KV row groups (batch x H_KV) per cell",
    )
    p.add_argument("--depths", type=float, nargs="+", default=list(DEPTHS))
    p.add_argument("--v8", action="store_true", help="8-bit V members instead of bf16 V")
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--tune-rounds", type=int, default=9)
    p.add_argument("--cold-only", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cs = cells(args)
    if args.dry_run:
        print(f"{len(cs)} cells, depths {args.depths}, v8={args.v8}")
        for c in cs:
            print("  ", c)
        return
    ceiling = read_ceiling()
    print(f"read ceiling {ceiling['gbs']:.0f} GB/s ({ceiling['method']})", flush=True)
    rep = Report(
        args.out,
        cells=cs,
        page=B.PAGE,
        depths=list(args.depths),
        v8=args.v8,
        rounds=args.rounds,
        ceiling=ceiling,
        controls=dict(
            cold_l2=True,
            warm_pair=not args.cold_only,
            rotation="coprime",
            per_family_tuning=True,
            one_fp32_reference=True,
        ),
    )
    flusher = L2Flush()
    for c in cs:
        try:
            run_cell(c, args, rep, flusher, ceiling)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rep.add(cell=c, error=repr(e)[:400])
            torch.cuda.empty_cache()
    rep.write()


if __name__ == "__main__":
    main()
