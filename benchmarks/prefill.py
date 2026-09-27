"""The prompt: its attention and what writing the decode cache costs.

`FoldKVCache.prefill` is FA-4's causal varlen attention followed by
`write_prompt`, which quantises K into two int8 planes and V into bf16 or two
e4m3 planes and, at a finite depth on a bf16 V, fits the tail model. This
times the attention beside FA-3's and FlashInfer's, and the cache write
beside FlashInfer's bf16 page append, which is what a bf16 serving cache pays
at the same point.

The operands are random: the prefill attention is FA-4's to the bit and the
writes' costs do not depend on the data. Cache writes are timed eagerly,
host work included, since `write_prompt` plans on the host.

    python -m benchmarks.prefill --out prefill
"""

from __future__ import annotations

import argparse
import math
import traceback

import torch
from exact_fold_attn import FoldKVCache

from benchmarks.harness.report import Report
from benchmarks.harness.timing import L2Flush, measure, paired_ratio

# (H, H_KV, D) of each captured model
MODELS = {"qwen3-30b": (32, 4, 128), "glm4-9b": (32, 2, 128), "gptoss-20b": (64, 8, 64)}
# (requests, prompt length)
SHAPES = [(32, 2048), (8, 8192), (2, 32768)]
PAGE = 128


def run_shape(model, Bz, P, args, rep, flusher):
    H, HKV, D = MODELS[model]
    print(f"\n=== {model} H={H}/{HKV} D={D} B={Bz} P={P} ===", flush=True)
    g = torch.Generator(device="cuda").manual_seed(0)
    N = Bz * P
    q = torch.randn(N, H, D, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(N, HKV, D, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(N, HKV, D, generator=g, device="cuda", dtype=torch.bfloat16)
    cu = torch.arange(0, N + 1, P, device="cuda", dtype=torch.int32)
    lens = [P] * Bz
    sc = 1.0 / math.sqrt(D)
    fns, graphable, prov, held = {}, {}, {}, []

    from flash_attn.cute import flash_attn_varlen_func as fa4

    fns["FA-4 attention"] = lambda: fa4(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=P,
        max_seqlen_k=P,
        causal=True,
        softmax_scale=sc,
    )
    prov["FA-4 attention"] = "flash_attn.cute.flash_attn_varlen_func(causal=True)"
    try:
        import flash_attn_interface as fa3

        fns["FA-3 attention"] = lambda: fa3.flash_attn_varlen_func(
            q, k, v, cu, cu, P, P, causal=True, softmax_scale=sc
        )
        prov["FA-3 attention"] = "flash_attn_interface.flash_attn_varlen_func(causal=True)"
    except Exception as e:  # noqa: BLE001
        print(f"  FA-3 unavailable: {repr(e)[:140]}", flush=True)
    try:
        import flashinfer as fi

        ws = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
        w = fi.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD")
        w.plan(cu, cu, H, HKV, D, causal=True, q_data_type=torch.bfloat16, sm_scale=sc)
        held += [ws, w]
        fns["FlashInfer attention"] = lambda: w.run(q, k, v)
        prov["FlashInfer attention"] = (
            "flashinfer.BatchPrefillWithRaggedKVCacheWrapper(causal=True)"
        )

        npg = -(-P // PAGE)
        kc = torch.empty(Bz * npg, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16)
        vc = torch.empty_like(kc)
        indptr = torch.arange(Bz + 1, device="cuda", dtype=torch.int32) * npg
        indices = torch.arange(Bz * npg, device="cuda", dtype=torch.int32)
        last = torch.full((Bz,), P - (npg - 1) * PAGE, device="cuda", dtype=torch.int32)
        seq = torch.full((Bz,), P, device="cuda", dtype=torch.int32)
        bi, pos = fi.get_batch_indices_positions(cu, seq, N)
        held += [kc, vc, indptr, indices, last, bi, pos]
        fns["FlashInfer bf16 page append"] = lambda: fi.append_paged_kv_cache(
            k, v, bi, pos, (kc, vc), indices, indptr, last, kv_layout="NHD"
        )
        graphable["FlashInfer bf16 page append"] = False
        prov["FlashInfer bf16 page append"] = "flashinfer.append_paged_kv_cache(bf16, NHD)"
    except Exception as e:  # noqa: BLE001
        print(f"  FlashInfer unavailable: {repr(e)[:140]}", flush=True)

    for label, kw in (
        ("bf16 V", dict(v8=False)),
        ("8-bit V", dict(v8=True)),
        ("bf16 V + tail (T=14)", dict(v8=False, depth=14.0)),
    ):
        att = FoldKVCache(Bz, H, HKV, D, P + 64, page_size=PAGE, **kw)
        held.append(att)
        n = f"Fold write_prompt, {label}"
        fns[n] = lambda att=att: att.write_prompt(k, v, cu, lens)
        graphable[n] = False
        prov[n] = f"FoldKVCache({kw}).write_prompt"
    att = FoldKVCache(Bz, H, HKV, D, P + 64, page_size=PAGE, depth=14.0)
    held.append(att)
    fns["Fold prefill (T=14)"] = lambda: att.prefill(q, k, v, cu, P)
    graphable["Fold prefill (T=14)"] = False
    prov["Fold prefill (T=14)"] = "FoldKVCache(depth=14).prefill: FA-4 attention + write_prompt"

    res = measure(fns, graphable=graphable, rounds=args.rounds, cold=True, flusher=flusher)
    flops = 4 * Bz * P * P / 2 * H * D
    rows = []
    for n, r in res.items():
        attn = n.endswith("attention")
        rows.append(
            dict(
                arm=n,
                provenance=prov[n],
                cold=r,
                tflops=flops / r["us"] / 1e6 if attn else None,
                over_fa4=paired_ratio(res, n, "FA-4 attention"),
            )
        )
        print(
            f"  {n:32s} {r['us']:10.1f} us  {r['mode']:5s}"
            + (f"  {flops / r['us'] / 1e6:6.0f} TFLOP/s" if attn else "")
            + f"  {r['us'] / res['FA-4 attention']['us']:6.1%} of FA-4",
            flush=True,
        )
    rep.add(model=model, H=H, HKV=HKV, D=D, B=Bz, P=P, arms=rows)
    del fns, held
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rep = Report(args.out, models=MODELS, shapes=SHAPES, page=PAGE)
    flusher = L2Flush()
    for m in args.models:
        for Bz, P in SHAPES:
            try:
                run_shape(m, Bz, P, args, rep, flusher)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rep.add(model=m, B=Bz, P=P, error=repr(e)[:400])
                torch.cuda.empty_cache()
    rep.write()


if __name__ == "__main__":
    main()
