"""Every decode baseline over one cache, each in its fastest configuration.

All paged arms read one page size, `PAGE = 128`: XQA accepts nothing larger,
and FoldAttention's output does not depend on its page size. FA-3 and FA-4
also run over a contiguous cache, so paging's cost to each is visible.

`tune` measures each family's variants at the shape and keeps the fastest,
so a comparison is against each library's best configuration and the result
names it. Every candidate is invoked once when built and dropped, with its
reason, if it fails; one whose error is far above the most accurate arm of
its precision is dropped as computing a different function (a length or mask
its entry point ignored).

FP8 families (`ArmSpec.fp8`) quantise K and V to e4m3 with one scale per
(request, KV head) where the library takes one, else per tensor. They are
reported beside the BF16 families and never chosen as the BF16 comparator.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

PAGE = 128
WS_BYTES = 256 << 20
E4M3_MAX = 448.0
# trtllm-gen, XQA's other backend, runs on SM100 only
XQA_BACKENDS = ("xqa",)

# An arm whose error is this many times its precision class's best is not a
# slower kernel for the same function. Real BF16 kernels land within 2x of
# each other; an ignored length or mask lands near 1.0.
SANITY = 20.0


@dataclass
class ArmSpec:
    name: str
    family: str
    fn: Callable
    post: Callable  # the arm's output to (B, H_KV, G, D) fp32
    provenance: str
    paged: bool
    graphable: bool = True
    fp8: bool = False
    keep: list = field(default_factory=list)
    # the step boundary: `append` writes the new token where this arm reads
    # it; `step` is an arm's own fused append-and-attend when it has one
    append: Callable | None = None
    step: Callable | None = None


@dataclass
class Won:
    spec: ArmSpec
    us: float
    err: float
    alternatives: dict


def first(o):
    return o[0] if isinstance(o, tuple) else o


def reference(q, k, v, lens, scale):
    """FP32 softmax attention of `q` `(B, H, D)` over each request's first
    `lens[b]` keys of `k`, `v` `(B, H_KV, S, D)`; `(B, H_KV, G, D)`."""
    B, H, D = q.shape
    HKV = k.shape[1]
    G = H // HKV
    out = torch.empty(B, HKV, G, D, dtype=torch.float32, device=q.device)
    qf = q.reshape(B, HKV, G, D).float()
    for b in range(B):
        n = int(lens[b])
        for h in range(HKV):
            s = (qf[b, h] @ k[b, h, :n].float().T) * scale
            out[b, h] = s.softmax(-1) @ v[b, h, :n].float()
    return out


def rel_l2(x, ref) -> float:
    return float((x.float() - ref).norm() / ref.norm())


@dataclass
class Layout:
    """One cache in every baseline's layout: contiguous `(B, S, H_KV, D)`
    and 128-key pages `(pages, PAGE, H_KV, D)` with FA's page table and
    FlashInfer's indptr/indices/last-page lengths."""

    k_s: torch.Tensor
    v_s: torch.Tensor
    kc: torch.Tensor
    vc: torch.Tensor
    table: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    last: torch.Tensor
    page: int


def paged_layout(k, v, lens, page=PAGE) -> Layout:
    """A dense, in-order page table over `k`, `v` `(B, H_KV, S, D)`, so every
    arm reads the same bytes in the same places."""
    B, HKV, S, D = k.shape
    dev = k.device
    assert S % page == 0, (S, page)
    per = S // page
    k_s = k.transpose(1, 2).contiguous()
    v_s = v.transpose(1, 2).contiguous()
    kc = k_s.reshape(B * per, page, HKV, D)
    vc = v_s.reshape(B * per, page, HKV, D)
    table = torch.arange(B * per, device=dev, dtype=torch.int32).view(B, per)
    npg = (lens.long() + page - 1) // page
    indptr = torch.cat([torch.zeros(1, device=dev, dtype=torch.long), npg.cumsum(0)]).int()
    indices = torch.cat([b * per + torch.arange(int(npg[b]), device=dev) for b in range(B)]).int()
    last = (lens.long() - (npg - 1) * page).int()
    return Layout(k_s, v_s, kc, vc, table, indptr, indices, last, page)


def _e4m3_per_head(x):
    """`x` `(B, S, H_KV, D)` bf16 to e4m3 with one scale per (request, KV
    head), `(B, H_KV)`. One request at a time: an fp32 copy of a whole 32K
    cache is tens of GB."""
    B, _, HKV, _ = x.shape
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(B, HKV, device=x.device, dtype=torch.float32)
    for b in range(B):
        xb = x[b].float()
        s = (xb.abs().amax(dim=(0, 2)) / E4M3_MAX).clamp_min(1e-30)
        scale[b] = s
        out[b] = (xb / s[None, :, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return out, scale


def _e4m3_per_tensor(x):
    """`x` to e4m3 with one scale for the tensor, returned as a float."""
    amax = max(float(x[i : i + 256].abs().amax()) for i in range(0, x.shape[0], 256))
    s = max(amax, 1e-30) / E4M3_MAX
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    for i in range(0, x.shape[0], 256):
        out[i : i + 256] = (x[i : i + 256].float() / s).clamp(-E4M3_MAX, E4M3_MAX).to(out.dtype)
    return out, s


def _appends(k, v, lens, lay, new_kv):
    """Writers of the token `new_kv` `(B, H_KV, D)` into each layout the arms
    read, `{layout: [(how, fn)]}`: FA/FlashInfer NHD pages, the contiguous
    `(B, S, H_KV, D)` cache and the SDPA `(B, H_KV, S, D)` one. Each writes
    at `lens - 1`."""
    kn, vn = new_kv
    B, HKV, D = kn.shape
    S = k.shape[2]
    dev = kn.device
    b = torch.arange(B, device=dev)
    h = torch.arange(HKV, device=dev)
    pos = lens.long() - 1
    pid = lay.table[b, pos // lay.page].long()
    prow = pid * lay.page + pos % lay.page
    crow = b * S + pos
    srow = ((b[:, None] * HKV + h) * S + pos[:, None]).reshape(-1)
    kf, vf = kn.reshape(-1, D), vn.reshape(-1, D)
    kc, vc = lay.kc.view(-1, HKV, D), lay.vc.view(-1, HKV, D)
    ks, vs = lay.k_s.view(-1, HKV, D), lay.v_s.view(-1, HKV, D)
    kd, vd = k.view(-1, D), v.view(-1, D)

    def nhd():
        kc.index_copy_(0, prow, kn)
        vc.index_copy_(0, prow, vn)

    def contig():
        ks.index_copy_(0, crow, kn)
        vs.index_copy_(0, crow, vn)

    def sdpa():
        kd.index_copy_(0, srow, kf)
        vd.index_copy_(0, srow, vf)

    out = {
        "nhd": [("index_copy", nhd)],
        "contig": [("index_copy", contig)],
        "sdpa": [("index_copy", sdpa)],
    }
    try:
        from flashinfer.page import append_paged_kv_cache

        bi, pi = b.int(), pos.int()

        def nhd_fi():
            append_paged_kv_cache(
                kn, vn, bi, pi, (lay.kc, lay.vc), lay.indices, lay.indptr, lay.last, "NHD"
            )

        nhd_fi()
        torch.cuda.synchronize()
        out["nhd"].append(("flashinfer.page.append_paged_kv_cache", nhd_fi))
    except Exception as e:  # noqa: BLE001
        print(f"  FlashInfer append unavailable: {repr(e)[:110]}", flush=True)
    out["rows"] = dict(pid=pid, pos=pos, prow=prow, crow=crow)
    return out


def _quantised_append(dst_k, dst_v, rows, kn, vn, ksc, vsc):
    """An FP8 arm's append: the new token quantised to e4m3 with the cache's
    scale, which is part of the step for an FP8 cache as it is for ours."""

    # CUDA's index_copy has no e4m3 kernel; the bytes move as uint8
    bk, bv = dst_k.view(torch.uint8), dst_v.view(torch.uint8)

    def f():
        qk = (kn / ksc).clamp(-E4M3_MAX, E4M3_MAX).to(dst_k.dtype)
        qv = (vn / vsc).clamp(-E4M3_MAX, E4M3_MAX).to(dst_v.dtype)
        bk.index_copy_(0, rows, qk.view(torch.uint8))
        bv.index_copy_(0, rows, qv.view(torch.uint8))

    return f


def build(
    q, k, v, lens=None, page=PAGE, *, layout: Layout | None = None, families=None, new_kv=None
):
    """Every live decode baseline for `q` `(B, H, D)` over `k`, `v`
    `(B, H_KV, S, D)` bf16. `layout` replaces the dense page table (a
    cascade's shared-prefix pages); `families` restricts what is built.

    `new_kv` `(k, v)` `(B, H_KV, D)` is the token at each request's last
    position, already in the cache. With it every arm also gets the append
    that writes it there (`ArmSpec.append`, or a fused `ArmSpec.step`), so a
    step can be timed as the token's write plus the attention, as a serving
    loop runs it. Rewriting the same bytes leaves the cache unchanged."""
    B, H, D = q.shape
    HKV, S = k.shape[1], k.shape[2]
    G = H // HKV
    dev = q.device
    scale = 1.0 / math.sqrt(D)
    if lens is None:
        lens = torch.full((B,), S, dtype=torch.int32, device=dev)
    lay = layout or paged_layout(k, v, lens, page)
    page = lay.page
    want = set(families) if families else None
    keep = [lay, lens]

    def rows(o):
        return first(o).reshape(B, HKV, G, D).float()

    specs: list[ArmSpec] = []

    def add(
        name,
        family,
        fn,
        prov,
        paged,
        extra=(),
        post=rows,
        graphable=True,
        fp8=False,
        append=None,
        step=None,
    ):
        if want and family not in want:
            return
        try:
            fn()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  dead  {name:34s} {repr(e)[:110]}", flush=True)
            return
        if new_kv is None:
            append = step = None
        if step is not None:
            try:
                step()
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                print(f"  dead  {name + ' fused step':34s} {repr(e)[:110]}", flush=True)
                step = None
        specs.append(
            ArmSpec(
                name,
                family,
                fn,
                post,
                prov,
                paged,
                graphable,
                fp8,
                list(extra) + keep,
                append,
                step,
            )
        )

    ap = _appends(k, v, lens, lay, new_kv) if new_kv is not None else {}
    keep.append(ap)

    q4 = q.view(B, 1, H, D).contiguous()
    qd = q.contiguous()
    keep += [q4, qd]
    if new_kv is not None:
        kn4, vn4 = new_kv[0][:, None].contiguous(), new_kv[1][:, None].contiguous()
        at_new = (lens - 1).to(torch.int32)
        keep += [kn4, vn4, at_new]

    try:
        import flash_attn_interface as fa3

        for paged in (True, False):
            for pg in (None, True):
                for ns in (0, 2, 4, 8, 16):
                    if not paged and ns:
                        continue
                    smd = fa3.get_scheduler_metadata(
                        B,
                        1,
                        S,
                        H,
                        HKV,
                        D,
                        lens,
                        qkv_dtype=torch.bfloat16,
                        page_size=page if paged else None,
                        causal=False,
                        pack_gqa=pg,
                        num_splits=ns,
                    )
                    tag = (
                        "FA-3"
                        + (" paged" if paged else " contig")
                        + (" pack" if pg else "")
                        + (f" s{ns}" if ns else "")
                    )

                    def f3(pg=pg, smd=smd, ns=ns, paged=paged):
                        return fa3.flash_attn_with_kvcache(
                            q4,
                            lay.kc if paged else lay.k_s,
                            lay.vc if paged else lay.v_s,
                            cache_seqlens=lens,
                            softmax_scale=scale,
                            page_table=lay.table if paged else None,
                            pack_gqa=pg,
                            scheduler_metadata=smd,
                            num_splits=ns,
                        )

                    f3s = None
                    if new_kv is not None:
                        # FA-3's own append: the token goes in at `lens - 1` and
                        # the attention covers it
                        try:
                            smd_s = fa3.get_scheduler_metadata(
                                B,
                                1,
                                S,
                                H,
                                HKV,
                                D,
                                at_new,
                                qkv_dtype=torch.bfloat16,
                                page_size=page if paged else None,
                                causal=False,
                                pack_gqa=pg,
                                num_splits=ns,
                                max_seqlen_k_new=1,
                            )
                        except Exception:  # noqa: BLE001
                            smd_s = None
                        keep.append(smd_s)

                        def f3s(pg=pg, smd_s=smd_s, ns=ns, paged=paged):
                            return fa3.flash_attn_with_kvcache(
                                q4,
                                lay.kc if paged else lay.k_s,
                                lay.vc if paged else lay.v_s,
                                k=kn4,
                                v=vn4,
                                cache_seqlens=at_new,
                                softmax_scale=scale,
                                page_table=lay.table if paged else None,
                                pack_gqa=pg,
                                scheduler_metadata=smd_s,
                                num_splits=ns,
                            )

                    add(
                        tag,
                        "FA-3",
                        f3,
                        f"flash_attn_interface.flash_attn_with_kvcache(page_size="
                        f"{page if paged else None}, pack_gqa={pg}, num_splits={ns}, "
                        "scheduler_metadata=planned)",
                        paged,
                        extra=[smd],
                        append=ap.get("nhd" if paged else "contig"),
                        step=f3s,
                    )

        k8, ks = _e4m3_per_head(lay.k_s)
        v8, vs = _e4m3_per_head(lay.v_s)
        qg = q.view(B, HKV, G, D).float()
        qs = (qg.abs().amax(dim=(2, 3)) / E4M3_MAX).clamp_min(1e-30).contiguous()
        q8 = (qg / qs[:, :, None, None]).to(torch.float8_e4m3fn).view(B, 1, H, D)
        add(
            "FA-3 FP8",
            "FA-3 FP8",
            lambda: fa3.flash_attn_with_kvcache(
                q8,
                k8,
                v8,
                cache_seqlens=lens,
                softmax_scale=scale,
                q_descale=qs,
                k_descale=ks,
                v_descale=vs,
            ),
            "flash_attn_interface.flash_attn_with_kvcache(e4m3 q/k/v, descale per "
            "(request, KV head), contiguous)",
            False,
            extra=[k8, v8, q8, ks, vs, qs],
            fp8=True,
            append=None
            if new_kv is None
            else [
                (
                    "e4m3 quantise + index_copy",
                    _quantised_append(
                        k8.view(-1, HKV, D),
                        v8.view(-1, HKV, D),
                        ap["rows"]["crow"],
                        new_kv[0],
                        new_kv[1],
                        ks[:, :, None],
                        vs[:, :, None],
                    ),
                )
            ],
        )
    except Exception as e:  # noqa: BLE001
        print(f"  FA-3 unavailable: {repr(e)[:140]}", flush=True)

    try:
        from flash_attn.cute import flash_attn_varlen_func as fa4

        for paged in (True, False):
            for pg in (None, True):
                tag = "FA-4" + (" paged" if paged else " contig") + (" pack" if pg else "")

                def f4(pg=pg, paged=paged):
                    return fa4(
                        q4,
                        lay.kc if paged else lay.k_s,
                        lay.vc if paged else lay.v_s,
                        seqused_k=lens,
                        softmax_scale=scale,
                        page_table=lay.table if paged else None,
                        pack_gqa=pg,
                    )

                add(
                    tag,
                    "FA-4",
                    f4,
                    f"flash_attn.cute.flash_attn_varlen_func(page_table="
                    f"{'yes' if paged else 'no'}, pack_gqa={pg}); SM90 has no split-KV",
                    paged,
                    append=ap.get("nhd" if paged else "contig"),
                )
    except Exception as e:  # noqa: BLE001
        print(f"  FA-4 unavailable: {repr(e)[:140]}", flush=True)

    # e4m3 pages with one scale per tensor, shared by FlashInfer's and XQA's
    # FP8 arms, whose wrappers take a scalar scale
    kc8, ksc = _e4m3_per_tensor(lay.kc)
    vc8, vsc = _e4m3_per_tensor(lay.vc)
    keep += [kc8, vc8]
    ap8 = None
    if new_kv is not None:
        ap8 = [
            (
                "e4m3 quantise + index_copy",
                _quantised_append(
                    kc8.view(-1, HKV, D),
                    vc8.view(-1, HKV, D),
                    ap["rows"]["prow"],
                    new_kv[0],
                    new_kv[1],
                    ksc,
                    vsc,
                ),
            )
        ]

    try:
        import flashinfer as fi

        for tc in (True, False):
            for fp8 in (False, True):
                try:
                    ws = torch.zeros(WS_BYTES, dtype=torch.uint8, device=dev)
                    w = fi.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
                    w.plan(
                        lay.indptr,
                        lay.indices,
                        lay.last,
                        H,
                        HKV,
                        D,
                        page,
                        q_data_type=torch.bfloat16,
                        kv_data_type=torch.float8_e4m3fn if fp8 else torch.bfloat16,
                        sm_scale=scale,
                    )
                    be = getattr(w, "_backend", "auto")
                    tag = f"FlashInfer {be}{'-tc' if tc else '-cc'}"
                    prov = (
                        f"flashinfer.BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores={tc}, "
                        f"backend={be}, page_size={page}"
                    )
                    if fp8:
                        add(
                            tag + " FP8",
                            "FlashInfer FP8",
                            lambda w=w: w.run(qd, (kc8, vc8), k_scale=ksc, v_scale=vsc),
                            prov + ", e4m3 K/V, per-tensor scale)",
                            True,
                            extra=[w, ws],
                            fp8=True,
                            append=ap8,
                        )
                    else:
                        add(
                            tag,
                            "FlashInfer",
                            lambda w=w: w.run(qd, (lay.kc, lay.vc)),
                            prov + ")",
                            True,
                            extra=[w, ws],
                            append=ap.get("nhd"),
                        )
                except Exception as e:  # noqa: BLE001
                    print(f"  FlashInfer tc={tc} fp8={fp8}: {repr(e)[:110]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  FlashInfer unavailable: {repr(e)[:140]}", flush=True)

    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache as xqa

        xws = torch.zeros(WS_BYTES, dtype=torch.uint8, device=dev)
        keep.append(xws)
        for be in XQA_BACKENDS:
            for fp8 in (False, True):
                cache = (kc8, vc8) if fp8 else (lay.kc, lay.vc)
                add(
                    f"XQA {be}" + (" FP8" if fp8 else ""),
                    "XQA FP8" if fp8 else "XQA",
                    lambda be=be, cache=cache, fp8=fp8: xqa(
                        query=qd,
                        kv_cache=cache,
                        workspace_buffer=xws,
                        block_tables=lay.table,
                        seq_lens=lens,
                        max_seq_len=S,
                        bmm1_scale=scale * (ksc if fp8 else 1.0),
                        bmm2_scale=vsc if fp8 else 1.0,
                        kv_layout="NHD",
                        backend=be,
                        out_dtype=torch.bfloat16,
                    ),
                    f"flashinfer.decode.trtllm_batch_decode_with_kv_cache(backend={be}, "
                    f"page_size={page}" + (", e4m3 K/V, per-tensor scale)" if fp8 else ")"),
                    True,
                    fp8=fp8,
                    append=ap8 if fp8 else ap.get("nhd"),
                )
    except Exception as e:  # noqa: BLE001
        print(f"  XQA unavailable: {repr(e)[:140]}", flush=True)

    # cuDNN's paged decode graph, through FlashInfer's binding: it takes
    # per-request lengths and a page table, so it covers ragged batches too.
    # Its pages are HND, so it reads its own copy of the same pages.
    try:
        from flashinfer.cudnn import cudnn_batch_decode_with_kv_cache as cudnn_paged

        kh = lay.kc.permute(0, 2, 1, 3).contiguous()
        vh = lay.vc.permute(0, 2, 1, 3).contiguous()
        cws = torch.zeros(WS_BYTES, dtype=torch.uint8, device=dev)
        keep += [kh, vh, cws]
        aph = None
        if new_kv is not None:
            r = ap["rows"]
            hrow = (
                (r["pid"][:, None] * HKV + torch.arange(HKV, device=dev)) * page
                + (r["pos"] % page)[:, None]
            ).reshape(-1)
            khr, vhr = kh.view(-1, D), vh.view(-1, D)
            kf, vf = new_kv[0].reshape(-1, D), new_kv[1].reshape(-1, D)

            def hnd():
                khr.index_copy_(0, hrow, kf)
                vhr.index_copy_(0, hrow, vf)

            aph = [("index_copy", hnd)]
            keep += [hrow, kf, vf]
        for lshape in ((B,), (B, 1, 1, 1)):
            lk = lens.reshape(lshape).contiguous()
            add(
                "cuDNN paged" + ("" if len(lshape) == 1 else " l4"),
                "cuDNN",
                lambda lk=lk: cudnn_paged(
                    qd,
                    kh,
                    vh,
                    scale,
                    cws,
                    max_sequence_kv=S,
                    actual_seq_lens_kv=lk,
                    block_tables=lay.table,
                    is_cuda_graph_compatible=True,
                ),
                f"flashinfer.cudnn.cudnn_batch_decode_with_kv_cache(page_size={page}, HND, "
                "is_cuda_graph_compatible=True)",
                True,
                extra=[lk],
                append=aph,
            )
    except Exception as e:  # noqa: BLE001
        print(f"  cuDNN paged unavailable: {repr(e)[:140]}", flush=True)

    # `scaled_dot_product_attention` takes no per-request length, so on a
    # ragged batch it would attend over padding; it runs on uniform batches
    # over the dense cache only.
    if bool((lens != S).any()):
        print("  skip  cuDNN: SDPA has no per-request length or page table", flush=True)
    else:
        try:
            import torch.nn.functional as F
            from torch.nn.attention import SDPBackend, sdpa_kernel

            qc = q.reshape(B, H, 1, D).contiguous()
            keep.append(qc)

            def cudnn():
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    return F.scaled_dot_product_attention(qc, k, v, scale=scale, enable_gqa=True)

            add(
                "cuDNN",
                "cuDNN",
                cudnn,
                "torch F.scaled_dot_product_attention(SDPBackend.CUDNN_ATTENTION, "
                "enable_gqa=True), contiguous (B, H_KV, S, D)",
                False,
                append=ap.get("sdpa"),
            )
        except Exception as e:  # noqa: BLE001
            print(f"  cuDNN unavailable: {repr(e)[:140]}", flush=True)

    return specs


def tune(specs, ref, rounds=9, drop=2, cold=True, flusher=None) -> dict[str, Won]:
    """Each family's fastest live arm at this shape, `{family: Won}`.

    Tuning is a separate, shorter measurement, so the headline rotation holds
    one arm per family and a few rounds complete it.
    """
    from .timing import measure

    err = {}
    for s in specs:
        try:
            err[s.name] = rel_l2(s.post(s.fn()), ref)
        except Exception:  # noqa: BLE001
            err[s.name] = float("nan")
    floor = {}
    for s in specs:
        e = err[s.name]
        if not math.isnan(e):
            floor[s.fp8] = min(floor.get(s.fp8, e), e)

    by_fam: dict[str, list[ArmSpec]] = {}
    for s in specs:
        by_fam.setdefault(s.family, []).append(s)
    won = {}
    for family, arms in by_fam.items():
        live = []
        for s in arms:
            e = err[s.name]
            if not math.isnan(e) and e <= floor[s.fp8] * SANITY:
                live.append(s)
            else:
                print(
                    f"  DROP  {s.name:24s} err {e:.3e} vs best {floor.get(s.fp8, float('nan')):.3e}"
                    ": not the same function",
                    flush=True,
                )
        if not live:
            continue
        res = measure(
            {s.name: s.fn for s in live},
            graphable={s.name: s.graphable for s in live},
            rounds=max(rounds, len(live)),
            drop=drop,
            cold=cold,
            flusher=flusher,
        )
        best = min(res, key=lambda n: res[n]["us"])
        spec = next(s for s in live if s.name == best)
        won[family] = Won(spec, res[best]["us"], err[best], {n: r["us"] for n, r in res.items()})
        alts = ", ".join(f"{n} {res[n]['us']:.1f}" for n in sorted(res, key=lambda n: res[n]["us"]))
        print(
            f"  tuned {family:14s} -> {best:26s} {res[best]['us']:8.1f} us err {err[best]:.3e}"
            f"\n        [{alts}]",
            flush=True,
        )
    return won


def step_arms(won, rounds=9, drop=2, cold=True, flusher=None) -> dict[str, tuple]:
    """Each family's fastest step, `{family: (name, fn, provenance)}`: its
    tuned decode behind each of its appends, or its fused append-and-attend,
    whichever is faster. Families built without `new_kv` have none."""
    from .timing import measure

    out = {}
    for family, w in won.items():
        s = w.spec
        cands, prov = {}, {}
        if s.step is not None:
            n = f"{s.name} fused"
            cands[n] = s.step
            prov[n] = s.provenance + " with k=, v= (fused append)"
        for how, app in s.append or []:
            n = f"{s.name} + {how}"

            def f(app=app, fn=s.fn):
                app()
                return fn()

            cands[n] = f
            prov[n] = f"{how} + {s.provenance}"
        if not cands:
            continue
        res = measure(
            cands,
            graphable={n: s.graphable for n in cands},
            rounds=max(rounds, len(cands)),
            drop=drop,
            cold=cold,
            flusher=flusher,
        )
        best = min(res, key=lambda n: res[n]["us"])
        out[family] = (f"{family} step", cands[best], prov[best])
        print(f"  step  {family:14s} -> {best:40s} {res[best]['us']:8.1f} us", flush=True)
    return out
