"""Where dense decode's speed comes from, one mechanism at a time.

Each cell times the tuned BF16 baselines beside FoldAttention members that
add one mechanism each, all over one capture and one FP32 reference:

- `refine all, 1 term`: every key reads both planes, so K costs 256 bytes a
  key at D = 128 as in BF16, and each weight is one bf16 term. What is left
  is the fixed reference: keys on tensor-core rows, no running maximum, no
  rescale.
- `refine all`: two weight terms, the dense default's accuracy.
- `gated, 1 term` and `dense` (gated, two terms): the second plane is read
  only for keys whose final weight needs it.
- `dense v8`: values in two e4m3 planes.
- `T=16`, `T=14`: keys below the depth are cut before V is read.

Bytes are the kernel's own counts of what it read.

    python -m benchmarks.ablation --out ablation
"""

from __future__ import annotations

import argparse
import math
import traceback

import torch

from benchmarks.harness import baselines as B
from benchmarks.harness import captures as C
from benchmarks.harness import fold as F
from benchmarks.harness.report import Report
from benchmarks.harness.timing import L2Flush, measure, paired_ratio

# (D, G, capture, H_KV) at 256 row groups
CELLS = {
    "qwen D128 G8": (128, 8, "qwen3-30b-l24-d128", 4),
    "qwen D128 G4": (128, 4, "qwen3-30b-l24-d128", 4),
    "glm D128 G16": (128, 16, "glm4-9b-l28-d128", 2),
    "gptoss D64 G8": (64, 8, "gptoss-20b-l9-d64", 8),
}
# refine_k past any logit's depth below Z, so every key reads plane B
ALL = 1e4
VARIANTS = (
    ("refine all, 1 term", dict(depth=None, refine_k=ALL, weight_terms=1)),
    ("refine all", dict(depth=None, refine_k=ALL)),
    ("gated, 1 term", dict(depth=None, weight_terms=1)),
    ("dense", dict(depth=None)),
    ("dense v8", dict(depth=None, v8=True)),
    ("T=16", dict(depth=16.0)),
    ("T=14", dict(depth=14.0)),
)


def run_cell(name, S, ragged, args, rep, flusher):
    D, G, cap, HKV = CELLS[name]
    Bz = 256 // HKV
    lens = (
        C.ragged_lens(Bz, S, page=B.PAGE)
        if ragged
        else torch.full((Bz,), S, dtype=torch.int32, device="cuda")
    )
    tag = f"{name} S{S} {'ragged' if ragged else 'uniform'}"
    print(f"\n=== {tag} ===", flush=True)
    shape = C.decode_shape(cap, B=Bz, S=S, G=G, HKV=HKV, lens=lens.tolist())
    keys = int(lens.sum()) * HKV
    ref = B.reference(shape["q"], shape["k"], shape["v"], lens, 1.0 / math.sqrt(D))

    lay = B.paged_layout(shape["k"], shape["v"], lens)
    specs = B.build(shape["q"], shape["k"], shape["v"], lens, layout=lay)
    won = {
        f: w
        for f, w in B.tune(specs, ref, rounds=args.tune_rounds, flusher=flusher).items()
        if not w.spec.fp8
    }
    arms = {w.spec.name: w.spec.fn for w in won.values()}
    meta = {
        w.spec.name: dict(
            family=w.spec.family, provenance=w.spec.provenance, err=w.err, bytes=keys * 4 * D
        )
        for w in won.values()
    }
    pr = F.prompt(shape, lens)
    for label, kw in VARIANTS:
        m = F.member(shape, lens, pr, page=B.PAGE, **kw)
        err = B.rel_l2(m["out"], ref)
        arms[label] = m["decode"]
        meta[label] = dict(
            family="FoldAttention",
            ours=True,
            provenance=m["provenance"],
            err=err,
            live=m["live"],
            refined=m["refined"],
            split=m["split"],
            bytes=F.bytes_read(m, keys, D),
        )
        print(
            f"  built {label:20s} err {err:.3e} refined {m['refined']:.3f} live {m['live']:.3f}",
            flush=True,
        )

    res = measure(arms, rounds=args.rounds, cold=True, flusher=flusher)
    fb = min((w.spec.name for w in won.values()), key=lambda n: res[n]["us"])
    rows = []
    for n in arms:
        rows.append(
            dict(
                arm=n,
                **{"ours": False, **meta[n]},
                cold=res[n],
                vs_fastest=paired_ratio(res, fb, n),
                bytes_per_key=meta[n]["bytes"] / keys,
            )
        )
        print(
            f"  {n:22s} {res[n]['us']:8.1f} us  err {meta[n]['err']:.3e}  "
            f"{meta[n]['bytes'] / keys:6.0f} B/key  {res[fb]['us'] / res[n]['us']:.3f}x",
            flush=True,
        )
    rep.add(
        cell=dict(name=name, D=D, G=G, S=S, ragged=ragged, B=Bz, HKV=HKV),
        tag=tag,
        capture=cap,
        fingerprint=shape["fingerprint"],
        keys=keys,
        fastest_baseline=fb,
        arms=rows,
    )
    del arms, specs, won
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="+", default=list(CELLS), choices=list(CELLS))
    p.add_argument("--contexts", type=int, nargs="+", default=[4096, 16384])
    p.add_argument("--ragged", type=int, nargs="+", default=[1])
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--tune-rounds", type=int, default=9)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rep = Report(
        args.out, cells=args.cells, contexts=args.contexts, variants=[v for v, _ in VARIANTS]
    )
    flusher = L2Flush()
    for name in args.cells:
        for S in args.contexts:
            for ragged in args.ragged:
                try:
                    run_cell(name, S, bool(ragged), args, rep, flusher)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    rep.add(cell=dict(name=name, S=S, ragged=bool(ragged)), error=repr(e)[:400])
                    torch.cuda.empty_cache()
    rep.write()


if __name__ == "__main__":
    main()
