"""Serving compositions: shared-prefix cascades, prefix trees and draft
verification.

Cascade: `n_req` requests hold one prefix of `P` keys and a suffix of `U`
keys of their own. Prefix tree: every request holds a system prompt of `P0`
keys shared by the whole batch, then one of `--tree-groups` documents of
`P1` keys shared by its group of requests, then its own `U` keys. Arms:

- every decode baseline over the flat batch, its paged arms reading each
  shared prefix's pages once through the page table (what prefix caching
  gives a paged server) and its contiguous arms over per-request copies;
- FlashInfer's `MultiLevelCascadeAttentionWrapper` with a level per prefix;
- the SM90 prefix-sharing prior art in `harness.prior`: PAT, vLLM's cascade
  path over the prefix the whole batch holds, and FastTree;
- FoldAttention flat, through `FoldKVCache` (the shipping step) and through
  the kernel-level call with its mass reference;
- FoldAttention's cascade (`SharedPrefix` per level), reading each prefix
  once for the requests that hold it: on the wide kernel past 64 stacked
  rows, from a `prefix_image` built once per prefix, and on the decode kernel
  once per `cascade_degree` requests below. Its reference is supplied, taken
  from the flat kernel-level call's mass estimate, because the mass prepass
  sees only the level it runs on; its time excludes that estimate and it is
  labelled "Z given". The flat member is timed at the decode and step
  boundaries, so the front's cost (Q's planes, the append and the reference:
  step minus decode) can be charged to the cascade too.

Draft: each request verifies `q_len` draft positions at the end of an `S`-key
cache, as a chain (causal) or a tree (ancestor mask). Arms: FA-3 and FA-4
(chains only), XQA with its draft mask, FlashInfer's paged prefill (both) and
multi-token decode (chains), SGLang's FA-3 tree verification (trees), and
FoldAttention's draft decode with its mass
reference: the decode kernel up to 64 rows (query heads x positions) and the
wide kernel past them.

Every arm is scored against FP32 attention with the same mask.

    python -m benchmarks.compose --out compose
"""

from __future__ import annotations

import argparse
import math
import traceback
from typing import Any

import torch

from benchmarks.harness import baselines as B
from benchmarks.harness import captures as C
from benchmarks.harness import fold as F
from benchmarks.harness import prior
from benchmarks.harness.report import Report
from benchmarks.harness.timing import L2Flush, measure, paired_ratio
from fold_attention.decode import (
    SharedPrefix,
    cascade_rows,
    draft_mask,
    pack_rows,
    paged_cache,
    paged_scale,
    prefix_image,
    prepare_fold_decode,
    quantize_k,
    quantize_q,
    unpack_rows,
)
from fold_attention.decode.heuristics import refine_for, shared_runs_wide, weight_terms_for

LOG2E = 1.4426950408889634
CAPTURES = ("qwen3-30b-l24-d128", "gptoss-20b-l9-d64")
DEPTHS = (12, 14, 16)
# binary draft trees: node i's parent is (i - 1) // 2
DRAFTS = {
    "chain2": [-1, 0],
    "chain4": [-1, 0, 1, 2],
    "chain8": list(range(-1, 7)),
    "tree8": [-1] + [i // 2 for i in range(7)],
    "chain16": list(range(-1, 15)),
    "tree16": [-1] + [i // 2 for i in range(15)],
}


def _is_chain(parents):
    return parents == list(range(-1, len(parents) - 1))


def _q_planes(q, D):
    return quantize_q((q.float() * (LOG2E / math.sqrt(D))).contiguous())


def _kernel_knobs(depth, D, G, S):
    rk, rv = refine_for(depth, seq_len=S, head_dim=D, group=G)
    return dict(
        truncate=depth is not None,
        refine_k=rk,
        refine_v=rv,
        weight_terms=weight_terms_for(depth),
    )


def _cut(depth, like):
    """The mass reference reads `cut` as each row's depth below Z; a dense
    call cuts nothing, which any depth past the logits' range says."""
    return torch.full_like(like, 1e4 if depth is None else float(depth))


def _rows_ref(q, k, v, lens, D):
    return B.reference(q, k, v, lens, 1.0 / math.sqrt(D))


def _level_groups(n_req, groups):
    """A level's request ranges as an indptr: `groups` contiguous groups of
    equal size, or None for `cascade_rows`'s own chunking of one prefix."""
    if groups is None:
        return None
    if n_req % groups:
        raise ValueError(f"{n_req} requests do not split into {groups} groups")
    return list(range(0, n_req + 1, n_req // groups))


def cascade_case(cap, n_req, levels, U, args, rep, flusher, kind="cascade"):
    """`n_req` requests over a tree of shared prefixes, then `U` keys each of
    their own. `levels` is `[(keys, groups)]` from the root: every request
    holds one prefix of each level, the `groups`-th share of the batch one
    document (None: one prefix for the whole batch, stacked as `cascade_rows`
    chunks it). A single level is a cascade, two a prefix tree."""
    qc, kc_all, vc_all = C.raw(cap)
    H, Sc, D = qc.shape
    HKV = kc_all.shape[0]
    G = H // HKV
    page = B.PAGE
    P = sum(n for n, _ in levels)
    S = P + U
    assert all(n % page == 0 for n, _ in levels) and U % page == 0
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    shape_tag = " + ".join(f"{n}x{1 if gr is None else gr}" for n, gr in levels)
    tag = f"{kind} {cap} n={n_req} P={shape_tag} U={U}"
    print(f"\n=== {tag} ===", flush=True)

    # each level's documents: the root starts at the capture's first key, so it
    # keeps the sink; deeper documents and the suffixes come from elsewhere
    docs = []
    for li, (n, gr) in enumerate(levels):
        nd = 1 if gr is None else gr
        root = levels[0][0]
        offs = [0] if li == 0 else torch.randint(root, Sc - n, (nd,), generator=g).tolist()
        docs.append(
            [
                (
                    kc_all[:, o : o + n].to(dev, torch.bfloat16),
                    vc_all[:, o : o + n].to(dev, torch.bfloat16),
                )
                for o in offs
            ]
        )
    offs = torch.randint(P, Sc - U, (n_req,), generator=g).tolist()

    def doc_of(li, r):
        gr = levels[li][1]
        return 0 if gr is None else r // (n_req // gr)

    suf_k = torch.stack([kc_all[:, o : o + U] for o in offs]).to(dev, torch.bfloat16)
    suf_v = torch.stack([vc_all[:, o : o + U] for o in offs]).to(dev, torch.bfloat16)
    q = torch.stack([qc[:, o + U - 1] for o in offs]).to(dev, torch.bfloat16)  # (B, H, D)
    k = torch.stack(
        [
            torch.cat([*(docs[li][doc_of(li, r)][0] for li in range(len(levels))), suf_k[r]], 1)
            for r in range(n_req)
        ]
    ).contiguous()
    v = torch.stack(
        [
            torch.cat([*(docs[li][doc_of(li, r)][1] for li in range(len(levels))), suf_v[r]], 1)
            for r in range(n_req)
        ]
    ).contiguous()
    lens = torch.full((n_req,), S, dtype=torch.int32, device=dev)
    ref = _rows_ref(q, k, v, lens, D)

    # one physical copy of each document's pages, then each request's own
    pages_k, pages_v, first = [], [], []
    at = 0
    for li, (n, _) in enumerate(levels):
        first.append([])
        for dk_, dv_ in docs[li]:
            first[li].append(at)
            pages_k.append(dk_.transpose(0, 1).reshape(n // page, page, HKV, D))
            pages_v.append(dv_.transpose(0, 1).reshape(n // page, page, HKV, D))
            at += n // page
    npu = U // page
    pages_k.append(suf_k.permute(0, 2, 1, 3).reshape(n_req * npu, page, HKV, D))
    pages_v.append(suf_v.permute(0, 2, 1, 3).reshape(n_req * npu, page, HKV, D))
    kc = torch.cat(pages_k).contiguous()
    vc = torch.cat(pages_v).contiguous()
    uniq_pages = at + torch.arange(n_req * npu, device=dev, dtype=torch.int32).view(n_req, npu)

    def level_pages(li, d):
        n = levels[li][0] // page
        return torch.arange(first[li][d], first[li][d] + n, device=dev, dtype=torch.int32)

    table = torch.stack(
        [
            torch.cat(
                [*(level_pages(li, doc_of(li, r)) for li in range(len(levels))), uniq_pages[r]]
            )
            for r in range(n_req)
        ]
    ).contiguous()
    npr = table.shape[1]
    lay = B.Layout(
        k_s=k.transpose(1, 2).contiguous(),
        v_s=v.transpose(1, 2).contiguous(),
        kc=kc,
        vc=vc,
        table=table,
        indptr=torch.arange(n_req + 1, device=dev, dtype=torch.int32) * npr,
        indices=table.reshape(-1).contiguous(),
        last=torch.full((n_req,), page, device=dev, dtype=torch.int32),
        page=page,
    )
    specs = B.build(q, k, v, lens, layout=lay)
    held = []

    try:
        import flashinfer as fi

        def i32(x):
            return torch.as_tensor(x, device=dev, dtype=torch.int32)

        qo, kv_indptr, kv_indices, last = [], [], [], []
        for li, (n, gr) in enumerate(levels):
            nd = 1 if gr is None else gr
            per = n_req // nd
            qo.append(i32(range(0, n_req + 1, per)))
            kv_indptr.append(i32(range(nd + 1)) * (n // page))
            kv_indices.append(torch.cat([level_pages(li, d) for d in range(nd)]).contiguous())
            last.append(i32([page] * nd))
        qo.append(i32(range(n_req + 1)))
        kv_indptr.append(i32(range(n_req + 1)) * npu)
        kv_indices.append(uniq_pages.reshape(-1).contiguous())
        last.append(i32([page] * n_req))
        ws = torch.zeros(B.WS_BYTES, dtype=torch.uint8, device=dev)
        w = fi.MultiLevelCascadeAttentionWrapper(len(levels) + 1, ws, "NHD")
        w.plan(
            qo,
            kv_indptr,
            kv_indices,
            last,
            H,
            HKV,
            D,
            page,
            q_data_type=torch.bfloat16,
            sm_scale=1.0 / math.sqrt(D),
        )
        qd = q.contiguous()

        def fi_cascade():
            return w.run(qd, (kc, vc))

        fi_cascade()
        torch.cuda.synchronize()
        held += [w, ws, qd, qo, kv_indptr, kv_indices, last]
        specs.append(
            B.ArmSpec(
                "FlashInfer cascade",
                "FlashInfer cascade",
                fi_cascade,
                lambda o: B.first(o).reshape(n_req, HKV, G, D).float(),
                f"flashinfer.MultiLevelCascadeAttentionWrapper({len(levels) + 1} levels, "
                f"page_size={page})",
                True,
            )
        )
    except Exception as e:  # noqa: BLE001
        print(f"  FlashInfer cascade unavailable: {repr(e)[:160]}", flush=True)

    def post(o):
        return B.first(o).reshape(n_req, HKV, G, D).float()

    scale = 1.0 / math.sqrt(D)
    nodes = [*docs, [(suf_k[r], suf_v[r]) for r in range(n_req)]]
    for family, make in (
        ("PAT", lambda: prior.pat(q, lay, lens, post, scale)),
        ("vLLM cascade", lambda: prior.vllm_cascade(q, lay, lens, levels[0][0], post, scale)),
        ("FastTree", lambda: prior.fasttree(q, nodes, post, scale)),
    ):
        _add_live(specs, family, make)

    won = B.tune(specs, ref, rounds=args.tune_rounds, flusher=flusher)

    arms, meta = {}, {}
    for wn in won.values():
        arms[wn.spec.name] = wn.spec.fn
        meta[wn.spec.name] = dict(
            family=wn.spec.family, provenance=wn.spec.provenance, fp8=wn.spec.fp8, err=wn.err
        )

    shape = dict(q=q, k=k, v=v, B=n_req, H=H, HKV=HKV, D=D, S=S)
    pr = F.prompt(shape, lens)
    NBH = n_req * HKV
    qa, qb, eq = _q_planes(q.view(n_req, HKV, G, D).reshape(NBH, G, D), D)
    kflat = quantize_k(k.reshape(NBH, S, D).float())
    vflat = v.reshape(NBH, S, D).contiguous()
    kuniq = quantize_k(suf_k.reshape(NBH, U, D).float())
    vuniq = suf_v.reshape(NBH, U, D).contiguous()

    # each level's pieces: its request ranges, their copies of the documents
    # they hold, paged over the documents' pages, and an image where the level
    # is stacked past 64 rows or at 64
    pieces: list[dict[str, Any]] = []
    for li, (n, gr) in enumerate(levels):
        rows, G_s = cascade_rows(n_req, G, HKV, _level_groups(n_req, gr), dev)
        ng = rows.shape[0] // HKV
        starts = [int(x) >> 7 for x in rows[::HKV, 0].tolist()]
        rdoc = [doc_of(li, b_ // HKV) for b_ in starts]
        lk = torch.cat([docs[li][d][0] for d in rdoc]).contiguous()
        lv = torch.cat([docs[li][d][1] for d in rdoc]).contiguous()
        planes = quantize_k(lk.float())
        ltab = torch.stack(
            [
                torch.arange(n // page, device=dev, dtype=torch.int32) + first[li][d] - first[li][0]
                for d in rdoc
            ]
        ).contiguous()
        pka, pkb, pv = paged_cache(planes[0], planes[1], lv, lv, ltab, page, HKV)[:3]
        pieces.append(
            dict(
                rows=rows,
                G_s=G_s,
                ng=ng,
                length=n,
                planes=planes,
                v=lv,
                pka=pka,
                pkb=pkb,
                pv=pv,
                pek=paged_scale(planes[2], ltab, page, HKV),
                table=ltab,
            )
        )

    for depth in (None, *args.depths):
        knobs = _kernel_knobs(depth, D, G, S)
        m = F.member(shape, lens, pr, depth=depth, page=page)
        name = m["name"]
        arms[f"{name} serving"] = m["decode"]
        meta[f"{name} serving"] = dict(
            family="FoldAttention",
            ours=True,
            depth=depth,
            err=B.rel_l2(m["out"], ref),
            live=m["live"],
            provenance=m["provenance"] + " (flat, FoldKVCache)",
        )
        arms[f"{name} serving step"] = m["step"]
        meta[f"{name} serving step"] = dict(meta[f"{name} serving"], boundary="step")
        held.append(m)

        z = torch.empty((NBH, G), device=dev, dtype=torch.float32)
        vm = vflat.float().mean(1) if depth is not None else None
        flat = prepare_fold_decode(
            qa,
            qb,
            eq,
            kflat[0],
            kflat[1],
            kflat[2],
            vflat,
            z,
            _cut(depth, z),
            reference="mass",
            group_cut="auto",
            vmean=vm,
            out_dtype=torch.bfloat16,
            **knobs,
        )
        o = flat()[0]
        arms[f"{name} kernel"] = flat
        meta[f"{name} kernel"] = dict(
            family="FoldAttention",
            ours=True,
            depth=depth,
            err=B.rel_l2(o.view(n_req, HKV, G, D), ref),
            provenance=f"prepare_fold_decode(reference='mass', depth={depth}) flat, contiguous",
        )

        zg = flat.z.clone()
        cut = zg - (1e4 if depth is None else float(depth))
        shared, how = [], []
        for pc in pieces:
            # a level stacked past 64 rows, or at 64, runs on the wide kernel,
            # which reads it as tiles built once per prefix, as a server would
            # when the prefix is cached
            image = prefix_image(*pc["planes"], v=pc["v"]) if pc["G_s"] >= 64 else None
            shared.append(
                SharedPrefix(
                    rows=pc["rows"],
                    ka=pc["pka"],
                    kb=pc["pkb"],
                    v=pc["pv"],
                    ek=pc["pek"],
                    vmean=pc["v"].float().mean(1).contiguous() if depth is not None else None,
                    page_table=pc["table"],
                    seq_lens=torch.full((pc["ng"],), pc["length"], device=dev, dtype=torch.int32),
                    length=pc["length"],
                    image=image,
                )
            )
            how.append(
                f"{pc['G_s']} rows x {pc['ng']} range(s)"
                + (
                    ", wide"
                    if shared_runs_wide(pc["G_s"], D, None if image is None else pc["length"])
                    else ""
                )
                + (", image" if image is not None else "")
            )
        casc = prepare_fold_decode(
            qa,
            qb,
            eq,
            kuniq[0],
            kuniq[1],
            kuniq[2],
            vuniq,
            zg,
            cut,
            shared=shared,
            n_kv_heads=HKV,
            page_size=page,
            vmean=vuniq.float().mean(1) if depth is not None else None,
            out_dtype=torch.bfloat16,
            **knobs,
        )
        o = casc()[0]
        arms[f"{name} cascade"] = casc
        meta[f"{name} cascade"] = dict(
            family="FoldAttention",
            ours=True,
            depth=depth,
            err=B.rel_l2(o.view(n_req, HKV, G, D), ref),
            reference="given (flat mass estimate, untimed)",
            provenance=f"prepare_fold_decode(shared=[{'; '.join(how)}], paged, "
            f"reference='given', depth={depth})",
        )
        held += [flat, casc, zg, cut, shared, z]
        print(
            f"  {name:12s} serving {meta[f'{name} serving']['err']:.3e}  kernel "
            f"{meta[f'{name} kernel']['err']:.3e}  cascade {meta[f'{name} cascade']['err']:.3e}",
            flush=True,
        )

    _time_and_add(
        rep,
        arms,
        meta,
        won,
        flusher,
        args,
        kind=kind,
        tag=tag,
        params=dict(
            capture=cap, n_req=n_req, P=P, U=U, levels=[list(x) for x in levels], H=H, HKV=HKV, D=D
        ),
    )
    del held, specs, won, arms, pieces
    torch.cuda.empty_cache()


def _add_live(specs, family, make):
    """Append `make()`'s arms that run, printing why the others do not."""
    try:
        made = make()
    except Exception as e:  # noqa: BLE001
        print(f"  {family} unavailable: {repr(e)[:160]}", flush=True)
        return
    for s in made:
        try:
            s.fn()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  dead  {s.name:30s} {repr(e)[:110]}", flush=True)
            continue
        specs.append(s)


def _draft_reference(q, k, v, mask, D):
    """`q` `(B, T, H, D)` over `k`, `v` `(B, H_KV, S, D)`; the last `T` keys
    are the draft's, visible per `mask` `(T, T)`."""
    Bz, T, H, _ = q.shape
    HKV, S = k.shape[1], k.shape[2]
    G = H // HKV
    out = torch.empty(Bz, T, H, D, device=q.device, dtype=torch.float32)
    vis = torch.ones(T, S, dtype=torch.bool, device=q.device)
    vis[:, S - T :] = mask.to(q.device)
    for b in range(Bz):
        for h in range(HKV):
            qh = q[b, :, h * G : (h + 1) * G].float()  # (T, G, D)
            s = torch.einsum("tgd,sd->tgs", qh, k[b, h].float()) / math.sqrt(D)
            s = s.masked_fill(~vis[:, None, :], float("-inf"))
            out[b, :, h * G : (h + 1) * G] = torch.einsum(
                "tgs,sd->tgd", s.softmax(-1), v[b, h].float()
            )
    return out


def draft_case(cap, Bz, S, dname, args, rep, flusher):
    parents = DRAFTS[dname]
    T = len(parents)
    chain = _is_chain(parents)
    qc, kc_all, vc_all = C.raw(cap)
    H, Sc, D = qc.shape
    HKV = kc_all.shape[0]
    G = H // HKV
    dev = torch.device("cuda")
    page = B.PAGE
    tag = f"draft {cap} B={Bz} S={S} {dname}"
    print(f"\n=== {tag} ===", flush=True)
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(0, Sc - S, (Bz,), generator=g).tolist()
    k = torch.stack([kc_all[:, s0 : s0 + S] for s0 in starts]).to(dev, torch.bfloat16)
    v = torch.stack([vc_all[:, s0 : s0 + S] for s0 in starts]).to(dev, torch.bfloat16)
    q = torch.stack([qc[:, s0 + S - T : s0 + S].transpose(0, 1) for s0 in starts]).to(
        dev, torch.bfloat16
    )  # (B, T, H, D)
    mask = draft_mask(parents)
    ref = _draft_reference(q, k, v, mask, D)
    lens = torch.full((Bz,), S, dtype=torch.int32, device=dev)
    lay = B.paged_layout(k, v, lens)
    scale = 1.0 / math.sqrt(D)

    def rows(o):
        return B.first(o).reshape(Bz, T, H, D).float()

    specs = []

    def add(name, family, fn, prov, paged):
        try:
            fn()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  dead  {name:30s} {repr(e)[:110]}", flush=True)
            return
        specs.append(B.ArmSpec(name, family, fn, rows, prov, paged))

    qc4 = q.contiguous()
    qflat = q.reshape(Bz * T, H, D).contiguous()
    if chain:
        try:
            import flash_attn_interface as fa3

            for paged in (True, False):
                add(
                    "FA-3" + (" paged" if paged else " contig"),
                    "FA-3",
                    lambda paged=paged: fa3.flash_attn_with_kvcache(
                        qc4,
                        lay.kc if paged else lay.k_s,
                        lay.vc if paged else lay.v_s,
                        cache_seqlens=lens,
                        page_table=lay.table if paged else None,
                        causal=True,
                        softmax_scale=scale,
                    ),
                    f"flash_attn_interface.flash_attn_with_kvcache(q_len={T}, causal=True, "
                    f"page_size={page if paged else None})",
                    paged,
                )
        except Exception as e:  # noqa: BLE001
            print(f"  FA-3 unavailable: {repr(e)[:140]}", flush=True)
        try:
            from flash_attn.cute import flash_attn_varlen_func as fa4

            for paged in (True, False):
                add(
                    "FA-4" + (" paged" if paged else " contig"),
                    "FA-4",
                    lambda paged=paged: fa4(
                        qc4,
                        lay.kc if paged else lay.k_s,
                        lay.vc if paged else lay.v_s,
                        seqused_k=lens,
                        page_table=lay.table if paged else None,
                        causal=True,
                        softmax_scale=scale,
                    ),
                    f"flash_attn.cute.flash_attn_varlen_func(q_len={T}, causal=True, "
                    f"page_table={'yes' if paged else 'no'})",
                    paged,
                )
        except Exception as e:  # noqa: BLE001
            print(f"  FA-4 unavailable: {repr(e)[:140]}", flush=True)
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache as xqa

        # row t's draft bits as one 32-bit word per row, read as two uint16
        bits = (mask.to(torch.int64) << torch.arange(T)).sum(-1).to(torch.int32)
        xmask = bits.view(T, 1).view(torch.uint16).to(dev)[None].expand(Bz, T, 2).contiguous()
        xws = torch.zeros(B.WS_BYTES, dtype=torch.uint8, device=dev)
        for be in B.XQA_BACKENDS:
            add(
                f"XQA {be}",
                "XQA",
                lambda be=be: xqa(
                    query=qflat,
                    kv_cache=(lay.kc, lay.vc),
                    workspace_buffer=xws,
                    block_tables=lay.table,
                    seq_lens=lens,
                    max_seq_len=S,
                    bmm1_scale=scale,
                    bmm2_scale=1.0,
                    kv_layout="NHD",
                    backend=be,
                    q_len_per_req=T,
                    mask=xmask,
                    out_dtype=torch.bfloat16,
                ),
                f"flashinfer.decode.trtllm_batch_decode_with_kv_cache(backend={be}, "
                f"q_len_per_req={T}, mask=draft bits, page_size={page})",
                True,
            )
    except Exception as e:  # noqa: BLE001
        print(f"  XQA unavailable: {repr(e)[:140]}", flush=True)
    try:
        import flashinfer as fi

        full = torch.ones(T, S, dtype=torch.bool, device=dev)
        full[:, S - T :] = mask.to(dev)
        kw = dict(causal=True) if chain else dict(custom_mask=full.reshape(-1).repeat(Bz))
        for be in ("fa2", "fa3"):
            try:
                ws = torch.zeros(B.WS_BYTES, dtype=torch.uint8, device=dev)
                w = fi.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend=be)
                w.plan(
                    torch.arange(Bz + 1, device=dev, dtype=torch.int32) * T,
                    lay.indptr,
                    lay.indices,
                    lay.last,
                    H,
                    HKV,
                    D,
                    page,
                    q_data_type=torch.bfloat16,
                    sm_scale=scale,
                    **kw,
                )
                add(
                    f"FlashInfer prefill {be}",
                    "FlashInfer",
                    lambda w=w: w.run(qflat, (lay.kc, lay.vc)),
                    f"flashinfer.BatchPrefillWithPagedKVCacheWrapper(backend={be}, "
                    + ("causal=True" if chain else "custom_mask=tree")
                    + f", page_size={page})",
                    True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  FlashInfer prefill {be}: {repr(e)[:140]}", flush=True)
        if chain:
            # the decode wrapper's multi-token path, causal over the draft
            ws = torch.zeros(B.WS_BYTES, dtype=torch.uint8, device=dev)
            w = fi.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
            w.plan(
                lay.indptr,
                lay.indices,
                lay.last,
                H,
                HKV,
                D,
                page,
                q_data_type=torch.bfloat16,
                sm_scale=scale,
                q_len_per_req=T,
            )
            add(
                "FlashInfer decode",
                "FlashInfer",
                lambda: w.run(qflat, (lay.kc, lay.vc)),
                f"flashinfer.BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True, "
                f"q_len_per_req={T}, page_size={page})",
                True,
            )
    except Exception as e:  # noqa: BLE001
        print(f"  FlashInfer unavailable: {repr(e)[:140]}", flush=True)

    if not chain:
        _add_live(
            specs,
            "SGLang",
            lambda: prior.sglang_tree_verify(qflat, lay, S, mask, rows, scale),
        )

    won = B.tune(specs, ref, rounds=args.tune_rounds, flusher=flusher)
    arms, meta = {}, {}
    for wn in won.values():
        arms[wn.spec.name] = wn.spec.fn
        meta[wn.spec.name] = dict(
            family=wn.spec.family, provenance=wn.spec.provenance, fp8=False, err=wn.err
        )

    NBH = Bz * HKV
    qa, qb, eq = _q_planes(pack_rows(q, HKV, T), D)
    kq = quantize_k(k.reshape(NBH, S, D).float())
    vf = v.reshape(NBH, S, D).contiguous()
    held = []
    for depth in (None, *args.depths):
        z = torch.empty((NBH, G * T), device=dev, dtype=torch.float32)
        kw = dict(causal=True) if chain else dict(mask=mask)
        run = prepare_fold_decode(
            qa,
            qb,
            eq,
            kq[0],
            kq[1],
            kq[2],
            vf,
            z,
            _cut(depth, z),
            reference="mass",
            draft_len=T,
            vmean=vf.float().mean(1) if depth is not None else None,
            out_dtype=torch.bfloat16,
            **kw,
            **_kernel_knobs(depth, D, G * T, S),
        )
        o = unpack_rows(run()[0], HKV, T)
        name = F.name_of(depth, False)
        arms[name] = run
        meta[name] = dict(
            family="FoldAttention",
            ours=True,
            depth=depth,
            err=B.rel_l2(o, ref),
            provenance=f"prepare_fold_decode(reference='mass', draft_len={T}, "
            + ("causal=True" if chain else "mask=tree")
            + f", depth={depth}), "
            + ("wide kernel" if G * T > 64 else F.kernel_of(run.config)),
        )
        held.append(run)
        print(f"  {name:12s} err {meta[name]['err']:.3e}", flush=True)

    _time_and_add(
        rep,
        arms,
        meta,
        won,
        flusher,
        args,
        kind="draft",
        tag=tag,
        params=dict(
            capture=cap, B=Bz, S=S, draft=dname, parents=parents, q_len=T, H=H, HKV=HKV, D=D
        ),
    )
    del held, won, arms
    torch.cuda.empty_cache()


def _time_and_add(rep, arms, meta, won, flusher, args, *, kind, tag, params):
    res = measure(
        arms,
        rounds=args.rounds,
        cold=True,
        graphable={w.spec.name: w.spec.graphable for w in won.values()},
        flusher=flusher,
    )
    bf16 = [w.spec.name for w in won.values() if not w.spec.fp8]
    fb = min(bf16, key=lambda n: res[n]["us"]) if bf16 else None
    rows = []
    for n in sorted(res, key=lambda n: res[n]["us"]):
        mt = meta[n]
        rows.append(
            dict(
                arm=n,
                boundary=mt.get("boundary", "decode"),
                **{k: v for k, v in mt.items() if k != "boundary"},
                cold=res[n],
                vs_fastest=paired_ratio(res, fb, n) if fb else None,
            )
        )
        print(
            f"  {n:30s} {res[n]['us']:8.1f} us  err {mt['err']:.3e}"
            + (f"  {res[fb]['us'] / res[n]['us']:5.2f}x of {fb}" if fb else ""),
            flush=True,
        )
    rep.add(
        kind=kind,
        tag=tag,
        **params,
        fastest_baseline=fb,
        tuned={
            f: dict(
                arm=w.spec.name, us=w.us, err=w.err, fp8=w.spec.fp8, alternatives=w.alternatives
            )
            for f, w in won.items()
        },
        arms=rows,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kinds", nargs="+", default=["cascade", "tree", "draft"])
    p.add_argument("--captures", nargs="+", default=list(CAPTURES))
    p.add_argument("--requests", type=int, nargs="+", default=[8, 32])
    p.add_argument("--prefixes", type=int, nargs="+", default=[4096, 16384])
    p.add_argument("--suffixes", type=int, nargs="+", default=[256, 2048])
    p.add_argument("--tree-system", type=int, default=2048, help="the prefix tree's root, P0")
    p.add_argument(
        "--tree-docs", type=int, nargs="+", default=[4096, 16384], help="its documents, P1"
    )
    p.add_argument("--tree-groups", type=int, default=4, help="documents per batch")
    p.add_argument("--tree-suffix", type=int, default=256)
    p.add_argument("--contexts", type=int, nargs="+", default=[4096, 16384])
    p.add_argument("--drafts", nargs="+", default=list(DRAFTS), choices=list(DRAFTS))
    p.add_argument("--depths", type=float, nargs="+", default=list(DEPTHS))
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--tune-rounds", type=int, default=9)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rep = Report(
        args.out,
        args=vars(args),
        page=B.PAGE,
        controls=dict(cold_l2=True, rotation="coprime", per_family_tuning=True),
    )
    flusher = L2Flush()
    jobs = []
    for cap in args.captures:
        if "cascade" in args.kinds:
            for n in args.requests:
                for P in args.prefixes:
                    for U in args.suffixes:
                        jobs.append(("cascade", (cap, n, [(P, None)], U)))
        if "tree" in args.kinds:
            for n in args.requests:
                for P1 in args.tree_docs:
                    levels = [(args.tree_system, None), (P1, args.tree_groups)]
                    jobs.append(("tree", (cap, n, levels, args.tree_suffix)))
        if "draft" in args.kinds:
            for n in args.requests:
                for S in args.contexts:
                    for d in args.drafts:
                        jobs.append(("draft", (cap, n, S, d)))
    for kind, a in jobs:
        try:
            if kind == "draft":
                draft_case(*a, args, rep, flusher)
            else:
                cascade_case(*a, args, rep, flusher, kind=kind)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rep.add(kind=kind, params=list(a), error=repr(e)[:400])
            torch.cuda.empty_cache()
    rep.write()


if __name__ == "__main__":
    main()
