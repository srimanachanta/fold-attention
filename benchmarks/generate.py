"""Multi-step generation over real captures: accuracy through the steps, then
latency on the final state.

A case is a batch of prompts, each a prefix of one captured sequence (so
each keeps the sequence's attention sink), and `--steps` decode steps that
feed the capture's own next q, k and v. Every step of every FoldAttention
member is scored against FP32 attention over everything written so far, and
so is FA-4 over a bf16 cache. After the last step every baseline is built and
tuned on the final state and timed with our members, at the decode and step
boundaries.

This is the shipping path end to end: `FoldKVCache` with its per-step mass
reference, its tail model and its per-key scales. Each step also records,
per query row, how far the row's true largest logit and log-sum-exp sit
above the reference the step estimated (`headroom`), in binades.

    python -m benchmarks.generate --out generate
"""

from __future__ import annotations

import argparse
import math
import statistics
import traceback
from dataclasses import dataclass

import torch
from exact_fold_attn import FoldKVCache

from benchmarks.harness import baselines as B
from benchmarks.harness import captures as C
from benchmarks.harness import fold as F
from benchmarks.harness.report import Report
from benchmarks.harness.timing import L2Flush, measure, paired_ratio

CASES = {
    "qwen_b8_16k": ("qwen3-30b-l24-d128", 8, 16384, 16384),
    "qwen_b32_8k-16k": ("qwen3-30b-l24-d128", 32, 8192, 16384),
    "qwen_b16_16k-32k": ("qwen3-30b-l24-d128", 16, 16384, 32000),
    "gptoss-l9_b32_8k-16k": ("gptoss-20b-l9-d64", 32, 8192, 16384),
    "gptoss-l21_b32_8k-16k": ("gptoss-20b-l21-d64", 32, 8192, 16384),
    "glm-l28_b32_8k-16k": ("glm4-9b-l28-d128", 32, 8192, 16384),
    "glm-l8_b32_8k-16k": ("glm4-9b-l8-d128", 32, 8192, 16384),
}
DEPTHS = (10, 12, 13, 14, 16, 18, 20)
LOG2E = 1.0 / math.log(2.0)


@dataclass
class Member:
    att: FoldKVCache
    depth: float | None
    v8: bool
    chunk: int | None = None


def _reference(q, k, v, n):
    """FP32 attention of one request's query `(H, D)` over keys `[0, n)` of
    `k`, `v` `(S, H_KV, D)`, and each row's largest logit and log-sum-exp in
    the kernel's base-2 units, `(H,)` each."""
    H, D = q.shape
    HKV = k.shape[1]
    kk, vv = k[:n].float(), v[:n].float()
    s = torch.einsum("hgd,nhd->hgn", q.float().view(HKV, H // HKV, D), kk) / math.sqrt(D)
    o = torch.einsum("hgn,nhd->hgd", torch.softmax(s, -1), vv).reshape(H, D)
    peak = s.amax(-1).reshape(H) * LOG2E
    lse = torch.logsumexp(s, -1).reshape(H) * LOG2E
    return o, peak, lse


def run_case(name, args, rep, flusher):
    cap, Bz, lo, hi = CASES[name]
    qc, kc, vc = C.raw(cap)
    H, Sc, D = qc.shape
    HKV = kc.shape[0]
    G = H // HKV
    steps = args.steps
    g = torch.Generator().manual_seed(args.seed)
    lens = torch.randint(lo, hi + 1, (Bz,), generator=g).tolist()
    need = max(lens) + steps
    if need > Sc:
        raise ValueError(f"{cap} has {Sc} tokens; the case needs {need}")
    dev = "cuda"
    q = qc[:, :need].transpose(0, 1).contiguous().to(dev).bfloat16()
    k = kc[:, :need].transpose(0, 1).contiguous().to(dev).bfloat16()
    v = vc[:, :need].transpose(0, 1).contiguous().to(dev).bfloat16()
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device=dev, dtype=torch.int32)
    pk = torch.cat([k[:n] for n in lens])
    pv = torch.cat([v[:n] for n in lens])
    print(
        f"\n=== {name}: {cap} H={H} H_KV={HKV} D={D} B={Bz} prompts {min(lens)}..{max(lens)}, "
        f"{steps} steps ===",
        flush=True,
    )

    # an 8-bit V's scale is the layer's, from the whole capture rather than
    # the batch, so a request's bits do not depend on its batch
    vmax = float(vc.abs().amax())
    v_scale = 2.0 ** math.ceil(math.log2(vmax * 2.0 / 448.0))
    members = {}
    for v8 in (False, True):
        for depth in (None, *args.depths):
            att = FoldKVCache(
                Bz,
                H,
                HKV,
                D,
                need + 64,
                depth=depth,
                v8=v8,
                page_size=B.PAGE,
                v_scale=v_scale if v8 else None,
            )
            att.write_prompt(pk, pv, cu, lens)
            members[F.name_of(depth, v8)] = Member(att, depth, v8)
        # no second plane of K (or of an 8-bit V) is ever read, so the cache
        # needs only its first planes: 386 B a key at D = 128 with a bf16 V,
        # 258 B with an 8-bit one
        att = FoldKVCache(
            Bz,
            H,
            HKV,
            D,
            need + 64,
            v8=v8,
            page_size=B.PAGE,
            v_scale=v_scale if v8 else None,
            refine_k=-1e4,
            refine_v=-1e4,
        )
        att.write_prompt(pk, pv, cu, lens)
        members["Fold capacity" + (" v8" if v8 else "")] = Member(att, None, v8)
    # batch invariant with no fixed split: a fixed number of keys per split,
    # dense at every chunk and depth 16 at the largest
    for chunk in args.chunks:
        for depth in (None, 16.0) if chunk == max(args.chunks) else (None,):
            att = FoldKVCache(Bz, H, HKV, D, need + 64, depth=depth, page_size=B.PAGE, chunk=chunk)
            att.write_prompt(pk, pv, cu, lens)
            members[f"{F.name_of(depth, False)} chunk{chunk}"] = Member(att, depth, False, chunk)

    # the baselines' cache, bf16 (B, S, H_KV, D), pages of 128 keys being views
    Smax = -(-need // B.PAGE) * B.PAGE
    bk = torch.zeros(Bz, Smax, HKV, D, device=dev, dtype=torch.bfloat16)
    bv = torch.zeros_like(bk)
    for b, n in enumerate(lens):
        bk[b, :n] = k[:n]
        bv[b, :n] = v[:n]
    from flash_attn.cute import flash_attn_varlen_func as fa4

    trace = {n: [] for n in [*members, "FA-4"]}
    live = {n: [] for n in members}
    nan_rows = dict.fromkeys(members, 0)
    headroom = {"peak_minus_z": [], "lse_minus_z": []}
    ref = qn = kn = vn = None
    for t in range(steps):
        idx = [n + t for n in lens]
        qn, kn, vn = q[idx], k[idx], v[idx]
        refs = [_reference(q[i], k, v, i + 1) for i in idx]
        ref = torch.stack([r[0] for r in refs])
        peak = torch.stack([r[1] for r in refs])
        lse = torch.stack([r[2] for r in refs])
        for n, m in members.items():
            o = m.att.decode(qn, kn, vn)
            if n == "Fold dense":
                # the reference this step estimated, against the row's true
                # largest logit and log-sum-exp
                z = m.att.z.view(Bz, H)
                headroom["peak_minus_z"].append((peak - z).flatten().tolist())
                headroom["lse_minus_z"].append((lse - z).flatten().tolist())
            trace[n].append(B.rel_l2(o, ref))
            nan_rows[n] += int((~torch.isfinite(o)).any(-1).sum())
            live[n].append(F.fractions(m.att.counts, [i + 1 for i in idx], HKV)["live"])
        pos = torch.tensor(idx, device=dev)
        bk[torch.arange(Bz, device=dev), pos] = kn
        bv[torch.arange(Bz, device=dev), pos] = vn
        o4 = fa4(
            qn[:, None],
            bk,
            bv,
            seqused_k=(pos + 1).int(),
            softmax_scale=1.0 / math.sqrt(D),
        )
        trace["FA-4"].append(B.rel_l2(B.first(o4)[:, 0], ref))
        print(
            f"  step {t}: FA-4 {trace['FA-4'][-1]:.3e}  dense {trace['Fold dense'][-1]:.3e}  "
            f"T=14 {trace.get('Fold T=14', [float('nan')])[-1]:.3e}",
            flush=True,
        )
    assert ref is not None and qn is not None and kn is not None and vn is not None
    final_lens = torch.tensor([n + steps for n in lens], device=dev, dtype=torch.int32)
    ref4 = ref.view(Bz, HKV, G, D)

    specs = B.build(
        qn,
        bk.transpose(1, 2).contiguous(),
        bv.transpose(1, 2).contiguous(),
        final_lens,
        new_kv=(kn, vn),
    )
    won = B.tune(specs, ref4, rounds=args.tune_rounds, flusher=flusher)
    step_fns = B.step_arms(won, rounds=args.tune_rounds, flusher=flusher)
    bf16 = {f: w for f, w in won.items() if not w.spec.fp8}

    arms, meta = {}, {}
    for w in won.values():
        arms[w.spec.name] = w.spec.fn
        meta[w.spec.name] = dict(
            family=w.spec.family,
            provenance=w.spec.provenance,
            fp8=w.spec.fp8,
            boundary="decode",
            err_final=w.err,
        )
    for n, m in members.items():
        att = m.att
        run = att.prepare_replay_decode(out_dtype=torch.bfloat16)
        out = run()[0].reshape(Bz, HKV, G, D).float()
        err_final = B.rel_l2(out, ref4)
        assert run.config is not None
        common = dict(
            family="FoldAttention",
            ours=True,
            depth=m.depth,
            v8=m.v8,
            split=run.config.split,
            kernel=F.kernel_of(run.config),
            chunk=m.chunk,
            err_final=err_final,
            err_trace=trace[n],
            err_median=statistics.median(trace[n]),
            err_worst=max(trace[n]),
            live=statistics.mean(live[n]),
            nan_rows=nan_rows[n],
            provenance=f"exact_fold_attn.FoldKVCache(page_size={B.PAGE}, depth={m.depth}, "
            f"v8={m.v8}"
            + (", refine_k=-1e4, refine_v=-1e4" if n.startswith("Fold capacity") else "")
            + (f", chunk={m.chunk}" if m.chunk else "")
            + f") split={run.config.split}, {F.kernel_of(run.config)}",
        )
        arms[n] = run
        meta[n] = dict(common, boundary="decode")
        arms[f"{n} step"] = att.prepare_replay_step(qn, kn, vn, at=att.seq_lens - 1)
        meta[f"{n} step"] = dict(common, boundary="step")
    for family, (sname, fn, prov) in step_fns.items():
        arms[sname] = fn
        meta[sname] = dict(
            family=family,
            provenance=prov,
            fp8=won[family].spec.fp8,
            boundary="step",
            err_final=won[family].err,
        )

    res = measure(arms, rounds=args.rounds, cold=True, flusher=flusher)
    fb = min((w.spec.name for w in bf16.values()), key=lambda n: res[n]["us"])
    fa4_name = bf16["FA-4"].spec.name if "FA-4" in bf16 else None
    bsteps = [step_fns[f][0] for f in step_fns if f in bf16]
    fbs = min(bsteps, key=lambda n: res[n]["us"]) if bsteps else None
    rows = []
    for n in sorted(res, key=lambda n: res[n]["us"]):
        mt = meta[n]
        rows.append(
            dict(
                arm=n,
                **mt,
                cold=res[n],
                vs_fastest=paired_ratio(res, fb, n),
                vs_fastest_step=(
                    paired_ratio(res, fbs, n) if fbs and mt["boundary"] == "step" else None
                ),
                vs_fa4=paired_ratio(res, fa4_name, n) if fa4_name else None,
                passes_fa4=(mt["err_final"] <= won["FA-4"].err) if "FA-4" in won else None,
            )
        )
        print(
            f"  {n:22s} {res[n]['us']:8.1f} us  err {mt['err_final']:.3e}  "
            f"{res[fb]['us'] / res[n]['us']:5.2f}x of {fb}",
            flush=True,
        )
    rep.add(
        case=name,
        capture=cap,
        fingerprint=C.fingerprint(cap),
        H=H,
        HKV=HKV,
        D=D,
        B=Bz,
        lens=lens,
        steps=steps,
        fa4_trace=trace["FA-4"],
        headroom={k: [[round(x, 3) for x in step] for step in v] for k, v in headroom.items()},
        fastest_baseline=fb,
        fastest_baseline_step=fbs,
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
    p.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--depths", type=float, nargs="+", default=list(DEPTHS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--tune-rounds", type=int, default=9)
    p.add_argument(
        "--chunks",
        type=int,
        nargs="*",
        default=[512, 1024, 2048],
        help="keys per split, chunk members",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rep = Report(
        args.out,
        cases={n: CASES[n] for n in args.cases},
        steps=args.steps,
        depths=list(args.depths),
        chunks=list(args.chunks),
        page=B.PAGE,
        controls=dict(cold_l2=True, rotation="coprime", per_family_tuning=True),
    )
    flusher = L2Flush()
    for n in args.cases:
        try:
            run_case(n, args, rep, flusher)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rep.add(case=n, error=repr(e)[:400])
            torch.cuda.empty_cache()
    rep.write()


if __name__ == "__main__":
    main()
