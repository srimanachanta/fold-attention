"""The training backward against FA-3, FA-4, cuDNN and DASH.

For each shape every arm reads the same inputs (FA-4's forward supplies O and
the LSE) and is scored three ways:

- latency, all arms timed together in one rotation;
- accuracy, dQ, dK and dV against FP32 gradients;
- repeatability, whether `--repeats` calls give the same bits.

FA-3 and DASH are one module name, so they run in separate processes
(`--impl fa3`, the default, or `--impl dash`). FA-4, cuDNN and FoldAttention
run in both, so DASH's time is joined to the FA-3 run through its own
same-process FoldAttention and FA-4.

`--mode train` times forward plus backward through each library's autograd
entry point, eagerly, as a training step calls it.

    python -m benchmarks.backward --sets headline small gqa dash-paper varlen --out backward_fa3
    python -m benchmarks.backward --impl dash --sets headline small dash-paper varlen --out backward_dash
    python -m benchmarks.backward --mode train --sets headline small --out train_fa3
"""

from __future__ import annotations

import argparse
import traceback

import torch

from benchmarks.harness import training as T
from benchmarks.harness.report import Report
from benchmarks.harness.timing import L2Flush, measure, paired_ratio

# the error multiple of the best arm past which an arm's gradients are wrong
SANITY = 20.0

# (B, H, H_KV, S, D, causal)
SETS = {
    # model shapes at FlashAttention's benchmark scale, 16K tokens or more
    "headline": [
        (2, 32, 8, 8192, 128, 1),
        (4, 32, 8, 4096, 128, 1),
        (8, 32, 8, 2048, 128, 1),
        (16, 32, 8, 1024, 128, 1),
        (2, 32, 32, 8192, 128, 1),
        (2, 32, 8, 8192, 128, 0),
        (2, 64, 8, 8192, 64, 1),
        (8, 64, 8, 2048, 64, 1),
    ],
    # single requests below that scale, where launch overhead is a large
    # share of the step
    "small": [
        (1, 8, 2, 8192, 128, 1),
        (1, 8, 2, 2048, 128, 1),
        (1, 16, 2, 4096, 64, 1),
    ],
    # the query group at a fixed token count and head count
    "gqa": [(2, 32, hkv, 8192, 128, 1) for hkv in (32, 16, 4, 2, 1)]
    + [(2, 64, hkv, 8192, 64, 1) for hkv in (64, 16, 4, 1)],
    # the DASH paper's sweep: 16K tokens, 2048 model width, MHA
    "dash-paper": [
        (16384 // s, 2048 // d, 2048 // d, s, d, causal)
        for d in (64, 128)
        for causal in (0, 1)
        for s in (512, 1024, 2048, 4096, 8192, 16384)
    ],
    # the same grid with groups of eight query heads per KV head
    "dash-paper-gqa": [
        (16384 // s, 2048 // d, 2048 // d // 8, s, d, causal)
        for d in (64, 128)
        for causal in (0, 1)
        for s in (512, 1024, 2048, 4096, 8192, 16384)
    ],
    # packed causal batches: (name, H, H_KV, D)
    "varlen": [
        ("mixed16", 8, 2, 128),
        ("mixed16", 16, 2, 64),
        ("skew16", 8, 2, 128),
        ("short64", 32, 8, 128),
        ("mixed16", 32, 32, 128),
    ],
}


def varlen_lens(tag):
    g = torch.Generator().manual_seed(0)
    if tag == "mixed16":
        return [int(x) for x in torch.randint(256, 4096, (16,), generator=g)]
    if tag == "short64":
        return [int(x) for x in torch.randint(128, 1024, (64,), generator=g)]
    if tag == "skew16":
        return [8192] + [int(x) for x in torch.randint(64, 512, (15,), generator=g)]
    raise ValueError(tag)


def arm_names(impl):
    common = ["FoldAttention", "FA-4", "FA-4 det", "cuDNN"]
    if impl == "dash":
        return ["DASH", *common]
    return ["FA-3", "FA-3 det", *common]


def run_shape(shape, set_name, args, rep, flusher, fa3, dash):
    if isinstance(shape[0], str):
        tag, H, HKV, D = shape
        lens = varlen_lens(tag)
        inp = T.make_varlen(lens, H, HKV, D)
        desc = dict(varlen=tag, lens=lens, H=H, HKV=HKV, D=D, causal=1)
        label = f"varlen {tag} H{H}/{HKV} D{D} ({len(lens)} seqs, {sum(lens)} tokens)"
    else:
        Bz, H, HKV, S, D, causal = shape
        inp = T.make(Bz, H, HKV, S, D, bool(causal))
        desc = dict(B=Bz, H=H, HKV=HKV, S=S, D=D, causal=causal)
        label = f"B{Bz} H{H}/{HKV} S{S} D{D} {'causal' if causal else 'full'}"
    print(f"\n=== {label} ===", flush=True)

    names = arm_names(args.impl)
    build = T.train_arms if args.mode == "train" else T.backward_arms
    arms = build(inp, names, fa3=fa3, dash=dash)

    accuracy, repeat = {}, {}
    if args.mode == "backward":
        ref = T.reference_grads(inp)
        for n, a in arms.items():
            a.run()
            g0 = [x.clone() for x in a.grads()]
            accuracy[n] = {
                w: T.rel_l2(x, r) for w, x, r in zip(("dq", "dk", "dv"), g0, ref, strict=True)
            }
            same = True
            for _ in range(args.repeats - 1):
                a.run()
                same &= all(torch.equal(x, y) for x, y in zip(a.grads(), g0, strict=True))
            repeat[n] = same
        del ref
    worst = {n: max(v.values()) for n, v in accuracy.items()}
    floor = min(worst.values()) if worst else None

    res = measure(
        {n: a.run for n, a in arms.items()},
        graphable={n: a.graphable for n, a in arms.items()},
        rounds=args.rounds,
        cold=True,
        flusher=flusher,
    )
    ours = "FoldAttention"
    rows = []
    for n, a in arms.items():
        rows.append(
            dict(
                arm=n,
                family=a.family,
                provenance=a.provenance,
                deterministic_by_design=a.deterministic,
                repeatable=repeat.get(n),
                accuracy=accuracy.get(n),
                # an arm this far from the best computes a different function,
                # and its time is not comparable
                correct=None if floor is None else worst[n] <= SANITY * floor,
                cold=res[n],
                over_ours=paired_ratio(res, n, ours) if ours in res else None,
            )
        )
        acc = accuracy.get(n)
        print(
            f"  {n:14s} {res[n]['us']:10.1f} us"
            + (f"  {res[n]['us'] / res[ours]['us']:.3f}x ours" if ours in res else "")
            + (f"  err dq {acc['dq']:.2e} dk {acc['dk']:.2e} dv {acc['dv']:.2e}" if acc else "")
            + (f"  bits {'same' if repeat[n] else 'DIFFER'}" if n in repeat else ""),
            flush=True,
        )
    rep.add(set=set_name, shape=desc, label=label, mode=args.mode, impl=args.impl, arms=rows)
    del arms, inp
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["fa3", "dash"], default="fa3")
    p.add_argument("--mode", choices=["backward", "train"], default="backward")
    p.add_argument("--sets", nargs="+", default=["headline"], choices=list(SETS))
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-shapes", type=int, default=None, help="the first N of each set")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    fa3 = T.load_interface("fa3") if args.impl == "fa3" else None
    dash = T.load_interface("dash") if args.impl == "dash" else None
    rep = Report(
        args.out,
        impl=args.impl,
        mode=args.mode,
        sets={s: SETS[s] for s in args.sets},
        repeats=args.repeats,
        controls=dict(cold_l2=True, rotation="coprime", same_inputs="FA-4 forward"),
    )
    flusher = L2Flush()
    for s in args.sets:
        for shape in SETS[s][: args.max_shapes]:
            try:
                run_shape(shape, s, args, rep, flusher, fa3, dash)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rep.add(shape=list(shape), set=s, error=repr(e)[:400])
                torch.cuda.empty_cache()
    rep.write()


if __name__ == "__main__":
    main()
