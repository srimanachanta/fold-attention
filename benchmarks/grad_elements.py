"""Per-element gradient error on attention operands captured from training.

`train_curve.py --capture` saves the q, k, v and dO one attention layer saw in
one micro-batch of a real training step. For each capture every backward arm
reads the same inputs (FA-4's forward supplies O and the LSE, as in
`backward.py`) and its dQ, dK and dV are scored element by element against
FP64 gradients of the same q, k, v and dO:

- `|err| / |ref|` percentiles over the elements whose reference is nonzero;
- the error in units of each reference element's bf16 ulp, and the fraction
  of elements off by more than half an ulp and by more than one;
- the relative error binned by the element's binade below the largest
  reference magnitude of its (request, KV head), 0 down to -20 (the last bin
  holds everything smaller). This is where an integer grid would lose
  relative precision on small entries against an fp32 accumulator.

`bf16(ref)`, the reference rounded once to bf16, is the floor any bf16 output
can reach. Beside the libraries, FoldAttention's `backward.ablation` variant
with dQ summed as fp32 atomics (`Fold fp32 dQ`).

    python -m benchmarks.grad_elements --out grad_elements
"""

from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path

import torch

from benchmarks.bwd_ablation import variant_arm
from benchmarks.harness import training as T
from benchmarks.harness.report import Report

ARMS = ("FoldAttention", "FA-3", "FA-3 det", "FA-4", "FA-4 det", "cuDNN")
PCTS = (50, 90, 99, 99.9, 99.99, 100)
BINADES = 20
FLOOR = 2.0**-126


def reference64(inp: T.Inputs):
    """FP64 dQ, dK, dV of causal softmax attention from the bf16 operands,
    one (request, query head) at a time."""
    dq = torch.zeros_like(inp.q, dtype=torch.float64)
    dk = torch.zeros_like(inp.k, dtype=torch.float64)
    dv = torch.zeros_like(inp.v, dtype=torch.float64)
    B, S, H, _ = inp.q.shape
    G = H // inp.k.shape[2]
    mask = torch.ones(S, S, device="cuda", dtype=torch.bool).tril() if inp.causal else None
    for b in range(B):
        for h in range(H):
            hk = h // G
            q, k, v, do = (
                x[b, :, i].double() for x, i in ((inp.q, h), (inp.k, hk), (inp.v, hk), (inp.do, h))
            )
            s = (q @ k.T) * inp.scale
            if mask is not None:
                s.masked_fill_(~mask, float("-inf"))
            p = s.softmax(-1)
            del s
            o = p @ v
            ds = p * (do @ v.T - (do * o).sum(-1, keepdim=True))
            dq[b, :, h] = (ds @ k) * inp.scale
            dk[b, :, hk] += (ds.T @ q) * inp.scale
            dv[b, :, hk] += p.T @ do
            del p, ds
    return dq, dk, dv


def group_max(ref, G):
    """The largest |ref| of each (request, KV head), broadcast to `ref`'s
    `(B, S, heads, D)` shape; query heads come in groups of `G`."""
    B, S, Hn, D = ref.shape
    a = ref.abs().view(B, S, Hn // G, G, D)
    m = a.amax(dim=(1, 3, 4), keepdim=True)
    return m.expand_as(a).reshape(B, S, Hn, D)


def _pct(sorted_x, p):
    n = sorted_x.numel()
    return float(sorted_x[min(n - 1, max(0, math.ceil(p / 100 * n) - 1))])


def score(x, ref, G):
    """Every statistic of one gradient tensor against its reference."""
    ref = ref.double()
    err = (x.double() - ref).abs()
    aref = ref.abs()
    keep = aref > FLOOR
    rel = err[keep] / aref[keep]
    ulp = torch.exp2(torch.floor(torch.log2(aref[keep])) - 7)
    in_ulp = err[keep] / ulp
    srt = rel.sort().values
    out = dict(
        n=int(keep.sum()),
        n_zero_ref=int((~keep).sum()),
        rel_l2=float(err.norm() / ref.norm()),
        rel=dict(zip((f"p{p}" for p in PCTS), (_pct(srt, p) for p in PCTS), strict=True)),
        rel_mean=float(rel.mean()),
        ulp_mean=float(in_ulp.mean()),
        frac_gt_half_ulp=float((in_ulp > 0.5).double().mean()),
        frac_gt_1_ulp=float((in_ulp > 1.0).double().mean()),
    )
    b = torch.floor(torch.log2(aref[keep] / group_max(ref, G)[keep])).clamp(min=-BINADES)
    bins = []
    for k in range(0, -BINADES - 1, -1):
        m = b == k
        n = int(m.sum())
        if n == 0:
            bins.append(dict(binade=k, n=0))
            continue
        rs = rel[m].sort().values
        bins.append(
            dict(
                binade=k,
                n=n,
                frac=n / out["n"],
                rel_p50=_pct(rs, 50),
                rel_p90=_pct(rs, 90),
                rel_p99=_pct(rs, 99),
                rel_max=float(rs[-1]),
                ulp_mean=float(in_ulp[m].mean()),
                frac_gt_1_ulp=float((in_ulp[m] > 1.0).double().mean()),
            )
        )
    out["by_binade"] = bins
    return out


# the `bwd_ablation` variants scored beside the libraries, by the name they
# get here
VARIANT_ARMS = {"Fold fp32 dQ": "fp32 / persistent"}
# (a, b): the grid against fp32 atomics in one kernel, the grid against
# deterministic FA-3, and the two accumulation orders' spread inside FA-3 and
# FA-4
PAIRS = (
    ("FoldAttention", "Fold fp32 dQ"),
    ("FoldAttention", "FA-3 det"),
    ("FA-3 det", "FA-3"),
    ("FA-4 det", "FA-4"),
)


def pair_score(a, b, ref, G):
    """How far two arms' dQ are apart, relative to the reference, overall and
    by binade: the part of the error one accumulation scheme adds over the
    other, which the shared bf16 rounding of P and dS hides in `score`."""
    ref = ref.double()
    aref = ref.abs()
    keep = aref > FLOOR
    d = (a.double() - b.double()).abs()[keep]
    rel = d / aref[keep]
    ulp = torch.exp2(torch.floor(torch.log2(aref[keep])) - 7)
    bins = torch.floor(torch.log2(aref[keep] / group_max(ref, G)[keep])).clamp(min=-BINADES)
    out = dict(
        frac_differ=float((d > 0).double().mean()),
        rel_p50=_pct(rel.sort().values, 50),
        rel_p99=_pct(rel.sort().values, 99),
        frac_gt_half_ulp=float((d > 0.5 * ulp).double().mean()),
        by_binade=[],
    )
    for k in range(0, -BINADES - 1, -1):
        m = bins == k
        n = int(m.sum())
        if n == 0:
            out["by_binade"].append(dict(binade=k, n=0))
            continue
        rs = rel[m].sort().values
        out["by_binade"].append(
            dict(
                binade=k,
                n=n,
                frac_differ=float((d[m] > 0).double().mean()),
                rel_p50=_pct(rs, 50),
                rel_p99=_pct(rs, 99),
                frac_gt_half_ulp=float((d[m] > 0.5 * ulp[m]).double().mean()),
            )
        )
    return out


def load_capture(path):
    from flash_attn.cute.interface import _flash_attn_fwd

    c = torch.load(path, map_location="cuda")
    q, k, v, do = (c[n].contiguous() for n in ("q", "k", "v", "do"))
    o, lse = _flash_attn_fwd(q, k, v, softmax_scale=c["scale"], causal=True, return_lse=True)[:2]
    assert lse is not None
    inp = T.Inputs(q, k, v, o, do, lse.contiguous(), True, c["scale"], [q.shape[1]] * q.shape[0])
    return inp, c.get("tag"), c.get("layer")


def run_capture(path, args, rep, fa3):
    inp, tag, layer = load_capture(path)
    G = inp.q.shape[2] // inp.k.shape[2]
    label = f"{tag} layer {layer}"
    print(f"\n=== {label} ({path.name}, B{inp.q.shape[0]} S{inp.q.shape[1]}) ===", flush=True)
    arms = T.backward_arms(inp, ARMS, fa3=fa3)
    for n in args.variants:
        a = variant_arm(VARIANT_ARMS[n], inp)
        a.run()
        arms[n] = a
    ref = reference64(inp)
    grads = {"bf16(ref)": tuple(r.to(torch.bfloat16) for r in ref)}
    for n, a in arms.items():
        a.run()
        torch.cuda.synchronize()
        grads[n] = tuple(x.clone() for x in a.grads())
    rows = {}
    for n, g in grads.items():
        rows[n] = {
            w: score(x, r, G if w == "dq" else 1)
            for w, x, r in zip(("dq", "dk", "dv"), g, ref, strict=True)
        }
        s = rows[n]
        print(
            f"  {n:14s} "
            + "  ".join(
                f"{w} p50 {s[w]['rel']['p50']:.1e} p99 {s[w]['rel']['p99']:.1e} "
                f"max {s[w]['rel']['p100']:.1e} >1ulp {s[w]['frac_gt_1_ulp']:.3f}"
                for w in ("dq", "dk", "dv")
            ),
            flush=True,
        )
    print("  dQ median |err|/|ref| by binade below the (request, KV head) max:", flush=True)
    names = list(rows)
    print("    binade " + " ".join(f"{n[:12]:>12s}" for n in names), flush=True)
    for i in range(BINADES + 1):
        cells = []
        for n in names:
            b = rows[n]["dq"]["by_binade"][i]
            cells.append(f"{b['rel_p50']:12.2e}" if b["n"] else f"{'-':>12s}")
        frac = rows[names[0]]["dq"]["by_binade"][i].get("frac", 0.0)
        print(f"    {-i:6d} " + " ".join(cells) + f"   ({frac:.2e} of elements)", flush=True)
    pairs = {}
    for a, b in PAIRS:
        if a in grads and b in grads:
            pairs[f"{a} - {b}"] = pair_score(grads[a][0], grads[b][0], ref[0], G)
    print("  dQ pairs, p99 |a-b|/|ref| by binade (differing fraction overall):", flush=True)
    for k, v in pairs.items():
        cells = " ".join(f"{x['rel_p99']:.0e}" if x["n"] else "-" for x in v["by_binade"][::2])
        print(f"    {k:28s} {v['frac_differ']:.3f}  {cells}", flush=True)
    rep.add(
        capture=path.name,
        tag=tag,
        layer=layer,
        shape=list(inp.q.shape),
        kv_heads=inp.k.shape[2],
        provenance={n: a.provenance for n, a in arms.items()},
        arms=rows,
        dq_pairs=pairs,
    )
    del arms, grads, ref, inp
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--states",
        default=str(T.CACHE / "grad_states"),
        help="train_curve.py --capture's directory",
    )
    p.add_argument("--variants", nargs="*", default=list(VARIANT_ARMS), choices=list(VARIANT_ARMS))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    fa3 = T.load_interface("fa3")
    paths = sorted(Path(args.states).glob("step*_layer*.pt"))
    rep = Report(
        args.out,
        captures=[x.name for x in paths],
        reference="FP64 from the bf16 q, k, v, dO; exact O",
        floor=FLOOR,
        binades=BINADES,
        variants={n: VARIANT_ARMS[n] for n in args.variants},
    )
    for path in paths:
        try:
            run_capture(path, args, rep, fa3)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rep.add(capture=path.name, error=repr(e)[:400])
            torch.cuda.empty_cache()
    rep.write()


if __name__ == "__main__":
    main()
