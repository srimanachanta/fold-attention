"""What stays bit-identical, measured: one request's result under each change
of how it is run, against the same request's first result.

Every entry is the largest absolute difference over the request's outputs
(decode) or its dQ, dK and dV (backward); zero means every bit matched. The
request's inputs are the same bits in every run, including the forward's O
and LSE, so a difference is the kernel's alone.

Backward (`--impl fa3` or `--impl dash`, as in `backward.py`):

- `repeat`: ten calls on the same inputs.
- `batch`: the request alone, then first in a batch of four.
- `packing`: the request alone, then packed with three others (varlen).

Decode, over a capture:

- `repeat`: two calls on the same cache.
- `batch`: the request in a batch of 32, then alone.
- `split`: FoldAttention at two split counts; FA-3 at two `num_splits`.
  FoldAttention also runs with a fixed chunk of keys per split, whose split
  count follows the batch while each request's partials do not.

    python -m benchmarks.invariance --impl fa3 --out invariance_fa3
    python -m benchmarks.invariance --impl dash --no-decode --out invariance_dash
"""

from __future__ import annotations

import argparse
import traceback

import torch

from benchmarks.harness import baselines as B
from benchmarks.harness import captures as C
from benchmarks.harness import fold as F
from benchmarks.harness import training as T
from benchmarks.harness.report import Report

REPEATS = 10


def maxdiff(a, b) -> float:
    return max(float((x.float() - y.float()).abs().max()) for x, y in zip(a, b, strict=True))


def _cat_dense(a: T.Inputs, b: T.Inputs) -> T.Inputs:
    return T.Inputs(
        *(
            torch.cat([x, y])
            for x, y in zip(
                (a.q, a.k, a.v, a.o, a.do, a.lse), (b.q, b.k, b.v, b.o, b.do, b.lse), strict=True
            )
        ),
        a.causal,
        a.scale,
        a.lens + b.lens,
    )


def _cat_varlen(a: T.Inputs, b: T.Inputs) -> T.Inputs:
    assert a.cu is not None and b.cu is not None
    cu = torch.cat([a.cu, b.cu[1:] + a.cu[-1]])
    return T.Inputs(
        *(
            torch.cat([x, y])
            for x, y in zip((a.q, a.k, a.v, a.o, a.do), (b.q, b.k, b.v, b.o, b.do), strict=True)
        ),
        torch.cat([a.lse, b.lse], dim=1),
        True,
        a.scale,
        a.lens + b.lens,
        cu,
    )


def _grads(arm, first_of=None):
    arm.run()
    torch.cuda.synchronize()
    g = tuple(x.clone() for x in arm.grads())
    if first_of is None:
        return g
    return tuple(x[first_of] for x in g)


def backward_rows(args, rep, fa3, dash):
    names = ["FA-3", "FA-3 det", "FA-4", "FA-4 det", "cuDNN", "FoldAttention"]
    if args.impl == "dash":
        names = ["DASH", "FA-4", "FA-4 det", "FoldAttention"]
    for B_, H, HKV, S, D, causal in args.shapes:
        label = f"B{B_} H{H}/{HKV} S{S} D{D} {'causal' if causal else 'full'}"
        print(f"\n=== backward {label} ===", flush=True)
        one = T.make(1, H, HKV, S, D, bool(causal), seed=0)
        rest = T.make(3, H, HKV, S, D, bool(causal), seed=1)
        four = _cat_dense(one, rest)
        arms1 = T.backward_arms(one, names, fa3, dash)
        arms4 = T.backward_arms(four, names, fa3, dash)
        for n, arm in arms1.items():
            base = _grads(arm)
            rep_dev = max(maxdiff(base, _grads(arm)) for _ in range(REPEATS - 1))
            batch_dev = (
                maxdiff(base, _grads(arms4[n], first_of=slice(0, 1))) if n in arms4 else None
            )
            rep.add(
                kind="backward",
                shape=label,
                arm=n,
                repeat=rep_dev,
                batch=batch_dev,
                provenance=arm.provenance,
            )
            print(
                f"  {n:16s} repeat {rep_dev:.3e}  batch "
                f"{'-' if batch_dev is None else f'{batch_dev:.3e}'}",
                flush=True,
            )
        del arms1, arms4, one, rest, four
        torch.cuda.empty_cache()

    if args.impl == "dash":
        return
    for H, HKV, D in ((8, 2, 128), (16, 2, 64)):
        label = f"varlen H{H}/{HKV} D{D}"
        print(f"\n=== backward {label} ===", flush=True)
        L = 4096
        one = T.make_varlen([L], H, HKV, D, seed=0)
        others = T.make_varlen([2048, 6144, 1024], H, HKV, D, seed=1)
        packed = _cat_varlen(one, others)
        vnames = ["FA-3", "FA-3 det", "FA-4", "FA-4 det", "FoldAttention"]
        arms1 = T.backward_arms(one, vnames, fa3, None)
        armsp = T.backward_arms(packed, vnames, fa3, None)
        for n, arm in arms1.items():
            base = _grads(arm)
            rep_dev = max(maxdiff(base, _grads(arm)) for _ in range(REPEATS - 1))
            pack_dev = maxdiff(base, _grads(armsp[n], first_of=slice(0, L))) if n in armsp else None
            rep.add(
                kind="backward",
                shape=label,
                arm=n,
                repeat=rep_dev,
                packing=pack_dev,
                provenance=arm.provenance,
            )
            print(
                f"  {n:16s} repeat {rep_dev:.3e}  packing "
                f"{'-' if pack_dev is None else f'{pack_dev:.3e}'}",
                flush=True,
            )
        del arms1, armsp
        torch.cuda.empty_cache()


# the baseline variant whose configuration the library chooses itself
AUTO = {
    "FA-3": "FA-3 paged",
    "FA-4": "FA-4 paged",
    "FlashInfer": "FlashInfer fa2-tc",
    "XQA": "XQA xqa",
    "cuDNN": "cuDNN paged",
}


def _baseline_outs(shape, lens):
    specs = {s.name: s for s in B.build(shape["q"], shape["k"], shape["v"], lens)}
    out = {}
    for fam, name in AUTO.items():
        if name in specs:
            out[fam] = specs[name].post(specs[name].fn()).clone()
    for ns in (2, 8):
        name = f"FA-3 paged s{ns}"
        if name in specs:
            out[f"FA-3 s{ns}"] = specs[name].post(specs[name].fn()).clone()
    torch.cuda.synchronize()
    return out, specs


def _slice_shape(shape, idx):
    return dict(shape, q=shape["q"][idx], k=shape["k"][idx], v=shape["v"][idx], B=1)


# keys per split for the fixed-chunk member
CHUNK = 2048


def decode_rows(args, rep):
    for cap, G, HKV, D in (("qwen3-30b-l24-d128", 8, 4, 128), ("gptoss-20b-l9-d64", 8, 8, 64)):
        S = 16384
        Bz = 32
        lens = C.ragged_lens(Bz, S, page=B.PAGE)
        shape = C.decode_shape(cap, B=Bz, S=S, G=G, HKV=HKV, lens=lens.tolist())
        label = f"{cap} B{Bz} S{S} ragged"
        print(f"\n=== decode {label} ===", flush=True)
        alone = _slice_shape(shape, slice(0, 1))
        lens1 = lens[:1].contiguous()

        many, _ = _baseline_outs(shape, lens)
        again, _ = _baseline_outs(shape, lens)
        one, _ = _baseline_outs(alone, lens1)
        for fam in AUTO:
            if fam not in many:
                continue
            row = dict(
                kind="decode",
                shape=label,
                arm=fam,
                repeat=maxdiff([many[fam]], [again[fam]]),
                batch=maxdiff([many[fam][:1]], [one[fam]]) if fam in one else None,
            )
            if fam == "FA-3" and "FA-3 s2" in many and "FA-3 s8" in many:
                row["split"] = maxdiff([many["FA-3 s2"]], [many["FA-3 s8"]])
            rep.add(**row)
            print(
                f"  {fam:12s} "
                + "  ".join(f"{k} {v:.3e}" for k, v in row.items() if isinstance(v, float)),
                flush=True,
            )

        for label_split, split, chunk in (
            ("default split", None, None),
            ("fixed split", 4, None),
            ("fixed chunk", None, CHUNK),
        ):
            pr = F.prompt(shape, lens)
            m = F.member(shape, lens, pr, depth=None, split=split, page=B.PAGE, chunk=chunk)
            again_m = m["decode"]()[0].reshape(Bz, HKV, G, D).float()
            pr1 = F.prompt(alone, lens1)
            m1 = F.member(alone, lens1, pr1, depth=None, split=split, page=B.PAGE, chunk=chunk)
            row = dict(
                kind="decode",
                shape=label,
                arm=f"FoldAttention ({label_split})",
                split_used=[m["split"], m1["split"]],
                repeat=maxdiff([m["out"]], [again_m]),
                batch=maxdiff([m["out"][:1]], [m1["out"]]),
            )
            if split is not None:
                m2 = F.member(shape, lens, pr, depth=None, split=2 * split, page=B.PAGE)
                row["split"] = maxdiff([m["out"]], [m2["out"]])
            rep.add(**row)
            print(
                f"  {row['arm']:30s} "
                + "  ".join(f"{k} {v:.3e}" for k, v in row.items() if isinstance(v, float))
                + f"  splits {row['split_used']}",
                flush=True,
            )
            del m, m1
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["fa3", "dash"], default="fa3")
    p.add_argument("--no-decode", action="store_true")
    p.add_argument("--no-backward", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    # a GQA and an MHA shape at FA's benchmark scale, and a D = 64 one
    args.shapes = [(4, 32, 8, 4096, 128, 1), (4, 16, 16, 4096, 128, 0), (4, 64, 8, 4096, 64, 1)]
    rep = Report(args.out, impl=args.impl, repeats=REPEATS, shapes=args.shapes)
    fa3 = T.load_interface("fa3") if args.impl == "fa3" else None
    dash = T.load_interface("dash") if args.impl == "dash" else None
    try:
        if not args.no_backward:
            backward_rows(args, rep, fa3, dash)
        if not args.no_decode:
            decode_rows(args, rep)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        rep.add(error=repr(e)[:400])
    rep.write()


if __name__ == "__main__":
    main()
