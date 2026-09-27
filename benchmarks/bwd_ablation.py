"""What the backward's integer dQ fold and its persistent scheduler each cost
or buy.

The shipped kernel is timed beside `backward.ablation` variants that change
one mechanism each (`prepare_backward_variant`):

- `fp32 / persistent` and `fp32 / single tile`: dQ's partials summed as fp32
  atomics in arrival order, as FA-3 and FA-4 sum them, not deterministic;
- `fold / single tile`: one CTA per work tile instead of the persistent work
  list.

`fold / persistent` is the shipped kernel, checked bit for bit against
`prepare_backward`. Each shape also times FA-3 and FA-4, deterministic and
not, over the inputs and FP32 reference of `backward.py`. Every arm's
repeatability is checked and the largest run-to-run difference recorded.

    python -m benchmarks.bwd_ablation --sets headline dash-paper-gqa --out bwd_ablation
"""

from __future__ import annotations

import argparse
import traceback

import torch
from exact_fold_attn.backward.ablation import prepare_backward_variant

from benchmarks.backward import SANITY, SETS
from benchmarks.harness import training as T
from benchmarks.harness.report import Report
from benchmarks.harness.timing import L2Flush, measure, paired_ratio

# name -> `prepare_backward_variant`'s switches
VARIANTS = {
    "fold / persistent": dict(fp32_dq=False, persistent=True),
    "fold / single tile": dict(fp32_dq=False, persistent=False),
    "fp32 / persistent": dict(fp32_dq=True, persistent=True),
    "fp32 / single tile": dict(fp32_dq=True, persistent=False),
}
BASE = "fold / persistent"
BASELINES = ("FA-3", "FA-3 det", "FA-4", "FA-4 det")


def variant_arm(name, inp: T.Inputs) -> T.Arm:
    kw = VARIANTS[name]
    t = [x.transpose(1, 2) for x in (inp.q, inp.k, inp.v, inp.o, inp.do)]
    launch = prepare_backward_variant(*t, inp.lse, causal=inp.causal, softmax_scale=inp.scale, **kw)

    def grads():
        return tuple(x.transpose(1, 2) for x in launch.run())

    return T.Arm(
        name,
        "FoldAttention ablation",
        launch.run,
        grads,
        f"exact_fold_attn.backward.ablation.prepare_backward_variant("
        f"{', '.join(f'{k}={v!r}' for k, v in kw.items())}, plan={launch.plan})",
        not kw["fp32_dq"],
        held=(launch,),
    )


def run_shape(shape, set_name, args, rep, flusher, fa3):
    Bz, H, HKV, S, D, causal = shape
    inp = T.make(Bz, H, HKV, S, D, bool(causal))
    desc = dict(B=Bz, H=H, HKV=HKV, S=S, D=D, causal=causal)
    label = f"B{Bz} H{H}/{HKV} S{S} D{D} {'causal' if causal else 'full'}"
    print(f"\n=== {label} ===", flush=True)

    arms = {}
    for n in args.variants:
        try:
            a = variant_arm(n, inp)
            a.run()
            torch.cuda.synchronize()
            arms[n] = a
        except Exception as e:  # noqa: BLE001
            print(f"  dead  {n:24s} {repr(e)[:300]}", flush=True)
    arms.update(T.backward_arms(inp, args.baselines, fa3=fa3))

    shipped = T.backward_arms(inp, ["FoldAttention"])
    same_as_shipped = None
    if BASE in arms and shipped:
        g_ship = [x.clone() for x in shipped["FoldAttention"].grads()]
        arms[BASE].run()
        same_as_shipped = all(
            torch.equal(x, y) for x, y in zip(arms[BASE].grads(), g_ship, strict=True)
        )
        del g_ship
    del shipped

    ref = T.reference_grads(inp)
    accuracy, repeat, spread, dkv_as_base = {}, {}, {}, {}
    base_g = None
    if BASE in arms:
        arms[BASE].run()
        base_g = [x.clone() for x in arms[BASE].grads()]
    for n, a in arms.items():
        a.run()
        g0 = [x.clone() for x in a.grads()]
        if base_g is not None and n in VARIANTS:
            dkv_as_base[n] = all(torch.equal(x, y) for x, y in zip(g0[1:], base_g[1:], strict=True))
        accuracy[n] = {
            w: T.rel_l2(x, r) for w, x, r in zip(("dq", "dk", "dv"), g0, ref, strict=True)
        }
        same, diff = True, [0.0, 0.0, 0.0]
        for _ in range(args.repeats - 1):
            a.run()
            for i, (x, y) in enumerate(zip(a.grads(), g0, strict=True)):
                same &= torch.equal(x, y)
                diff[i] = max(diff[i], float((x.float() - y.float()).abs().max()))
        repeat[n] = same
        spread[n] = dict(zip(("dq", "dk", "dv"), diff, strict=True))
        del g0
    del ref
    worst = {n: max(v.values()) for n, v in accuracy.items()}
    floor = min(worst.values())

    res = measure(
        {n: a.run for n, a in arms.items()},
        graphable={n: a.graphable for n, a in arms.items()},
        rounds=args.rounds,
        cold=True,
        flusher=flusher,
    )
    rows = []
    for n, a in arms.items():
        rows.append(
            dict(
                arm=n,
                family=a.family,
                provenance=a.provenance,
                deterministic_by_design=a.deterministic,
                repeatable=repeat[n],
                max_repeat_diff=spread[n],
                dkv_same_as_base=dkv_as_base.get(n),
                accuracy=accuracy[n],
                correct=worst[n] <= SANITY * floor,
                cold=res[n],
                over_base=paired_ratio(res, n, BASE) if BASE in res else None,
            )
        )
        acc = accuracy[n]
        print(
            f"  {n:24s} {res[n]['us']:10.1f} us"
            + (f"  {rows[-1]['over_base']['ratio']:.3f}x base" if BASE in res else "")
            + f"  err dq {acc['dq']:.2e} dk {acc['dk']:.2e} dv {acc['dv']:.2e}"
            + f"  bits {'same' if repeat[n] else 'DIFFER'}"
            + ("" if repeat[n] else f" (max |d| dq {spread[n]['dq']:.1e})"),
            flush=True,
        )
    if same_as_shipped is not None:
        print(f"  {BASE} bits == prepare_backward: {same_as_shipped}", flush=True)
    rep.add(
        set=set_name,
        shape=desc,
        label=label,
        same_as_shipped=same_as_shipped,
        plans={n: a.provenance for n, a in arms.items() if n in VARIANTS},
        arms=rows,
    )
    del arms, inp
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=["headline", "dash-paper-gqa"], choices=list(SETS))
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max-shapes", type=int, default=None, help="the first N of each set")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--baselines", nargs="*", default=list(BASELINES), choices=list(BASELINES))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    fa3 = T.load_interface("fa3")
    rep = Report(
        args.out,
        variants={n: VARIANTS[n] for n in args.variants},
        base=BASE,
        sets={s: SETS[s] for s in args.sets},
        repeats=args.repeats,
        controls=dict(cold_l2=True, rotation="coprime", same_inputs="FA-4 forward"),
    )
    flusher = L2Flush()
    for s in args.sets:
        for shape in SETS[s][: args.max_shapes]:
            try:
                run_shape(shape, s, args, rep, flusher, fa3)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rep.add(shape=list(shape), set=s, error=repr(e)[:400])
                torch.cuda.empty_cache()
    rep.write()


if __name__ == "__main__":
    main()
