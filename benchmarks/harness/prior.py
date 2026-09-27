"""SM90 prior art for shared prefixes and draft trees, as `ArmSpec`s over the
pages the decode baselines read.

- PAT (ASPLOS '26): packs queries by the prefixes their page tables share,
  runs its multi-tile kernel per pack and merges the splits. The schedule is
  built once per batch on the host, as its vLLM plugin does per step, and is
  not timed. PAT launches on the legacy default stream and on streams it
  creates per call, which a CUDA graph cannot capture, so it is timed
  eagerly on the default stream; the L2 flush before each sample covers its
  host launch path. At head dim 64 with eight query heads per KV head its
  output is wrong (relative error above 1 on random operands, fp16 and
  bf16) and a run that included it ended in an illegal memory access, so it
  is not run there.
- vLLM's cascade path (`vllm.v1.attention.backends.flash_attn`): FA-3 over
  the prefix every request holds with all queries stacked, FA-3 over each
  request's remainder, then `merge_attn_states`. vLLM takes only the prefix
  common to the whole batch, so a prefix tree cascades over its root.
- FastTree (MLSys '25): its Triton kernels as PAT's artifact ships them
  (`benchmark/FastTree.py`, found through `EFA_PAT`), over one copy of each
  tree node, with the authors' H100 parameters.
- SGLang's tree verification (FA-3 backend, more than one draft branch): FA-3
  of every draft query over the committed cache, FA-3 of each draft query
  over its own ancestors through a one-key-per-page table, then
  `merge_state_v2`. SGLang loads the same FA-3 source from a kernels-hub
  build when it can download it and from `sgl_kernel` otherwise; this uses
  `sgl_kernel`.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import torch

from . import baselines as B

PAT_SRC = os.environ.get("EFA_PAT")


def _pat_commit():
    if not PAT_SRC:
        return "unknown"
    try:
        return subprocess.run(
            ["git", "-C", PAT_SRC, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def pat(q, lay, lens, post, scale):
    """PAT over `lay`'s page table through its C++ tree (the path its vLLM
    plugin takes), with and without its compute model."""
    from prefix_attn import PrefixTreeCPP, prefix_attn_with_kvcache

    H, D = q.shape[1:]
    HKV = lay.kc.shape[2]
    if D == 64 and H // HKV == 8:
        raise RuntimeError("PAT's output is wrong at head dim 64 with G=8; not run")
    q4 = q.unsqueeze(1).contiguous()
    table = lay.table.cpu().contiguous()
    seqlens = lens.cpu().tolist()
    dev = torch.device("cuda", q.device.index if q.device.index is not None else 0)
    specs = []
    for model in (False, True):
        tree = PrefixTreeCPP(lay.page)
        tree.build_radix_tree(seqlens, table)
        tree.pack_schedule(MNWs=None, HRatio=H // HKV, kvHead=HKV, use_compute_model=model)
        tree.kernel_info.to_gpu(dev)
        out = torch.empty_like(q4)

        def fn(tree=tree, out=out):
            prefix_attn_with_kvcache(
                q=q4,
                k_cache_paged=lay.kc,
                v_cache_paged=lay.vc,
                tree=tree,
                softmax_scale=scale,
                out=out,
            )
            return out

        specs.append(
            B.ArmSpec(
                f"PAT{' compute model' if model else ''}",
                "PAT",
                fn,
                post,
                f"prefix_attn.prefix_attn_with_kvcache(PrefixTreeCPP, use_compute_model={model}, "
                f"page_size={lay.page}), PAT {_pat_commit()}",
                True,
                graphable=False,
                keep=[tree, table, out, q4],
            )
        )
    return specs


def vllm_cascade(q, lay, lens, prefix, post, scale):
    """vLLM's `cascade_attention` over the first `prefix` keys every request
    holds, with FA-3's own split choice (eager) and with vLLM's CUDA-graph
    split cap."""
    from vllm.v1.attention.backends.flash_attn import cascade_attention

    n = q.shape[0]
    dev = q.device
    i32 = torch.int32
    qd = q.contiguous()
    cu_q = torch.arange(n + 1, device=dev, dtype=i32)
    cu_prefix = torch.tensor([0, n], device=dev, dtype=i32)
    prefix_lens = torch.tensor([prefix], device=dev, dtype=i32)
    suffix_lens = (lens - prefix).to(torch.int32)
    max_kv = int(lens.max())
    specs = []
    for splits in (0, 32):
        out = torch.empty_like(qd)

        def fn(splits=splits, out=out):
            cascade_attention(
                out,
                qd,
                lay.kc,
                lay.vc,
                cu_query_lens=cu_q,
                max_query_len=1,
                cu_prefix_query_lens=cu_prefix,
                prefix_kv_lens=prefix_lens,
                suffix_kv_lens=suffix_lens,
                max_kv_len=max_kv,
                softmax_scale=scale,
                alibi_slopes=None,
                sliding_window=(-1, -1),
                logits_soft_cap=0.0,
                block_table=lay.table,
                common_prefix_len=prefix,
                max_num_splits=splits,
                fa_version=3,
            )
            return out

        specs.append(
            B.ArmSpec(
                f"vLLM cascade splits={splits or 'auto'}",
                "vLLM cascade",
                fn,
                post,
                f"vllm.v1.attention.backends.flash_attn.cascade_attention(common_prefix_len="
                f"{prefix}, max_num_splits={splits}, fa_version=3, page_size={lay.page})",
                True,
                keep=[qd, cu_q, cu_prefix, prefix_lens, suffix_lens, out],
            )
        )
    return specs


def _fasttree_module():
    if not PAT_SRC:
        raise RuntimeError("EFA_PAT is not set")
    path = Path(PAT_SRC) / "benchmark" / "FastTree.py"
    spec = importlib.util.spec_from_file_location("fasttree", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"no FastTree at EFA_PAT/{path.relative_to(PAT_SRC)}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fasttree(q, nodes, post, scale):
    """FastTree over a tree given by `nodes`: `[(layer, [(k, v), ...])]` from
    the root, each node's keys `(H_KV, n, D)`, the last layer one node per
    request in request order. Children split their parents' layer evenly and
    in order, which is the tree FastTree's `generate_tree` builds."""
    ft = _fasttree_module()
    n, H, D = q.shape
    HKV = nodes[0][0][0].shape[0]
    G = H // HKV
    sizes = [len(layer) for layer in nodes]
    lengths = [layer[0][0].shape[1] for layer in nodes]
    info = ft.generate_tree(sizes, lengths)
    flat = [kv for layer in nodes for kv in layer]
    ptrs = [0]
    for k, _ in flat:
        ptrs.append(ptrs[-1] + k.shape[1])
    req = 0
    for i, node in enumerate(info):
        if node.num_children == 0:
            j = i
            while j != -1:
                info[j].requests.append(req)
                j = info[j].parent
            req += 1
    assert req == n, (req, n)
    kt = torch.cat([k.transpose(0, 1) for k, _ in flat]).contiguous()
    vt = torch.cat([v.transpose(0, 1) for _, v in flat]).contiguous()

    # the authors' H100 settings: 64 and 16 query-head rows per tile in the
    # two phases (their G=1, 4, 16 presets), 132-SM parallelism thresholds
    tsq = [64 // G, 16 // G]
    tsk = [32, 32]
    params = ft.FastTreeParams()
    params.set_values(0.66, 0.33, 0.1)
    params.set_q_tile_sizes(tsq)
    params.set_kv_tile_sizes(tsk)
    params.set_kv_group_num(G)
    aux, _ = ft.fasttree_preparation(
        info, ptrs, n, H, HKV, D, [1024, 128], [132, 528], [132, 132], params, q.device
    )
    qd = q.contiguous()
    out = torch.empty_like(qd)

    def fn():
        ft.fasttree_decode(qd, kt, vt, out, *aux, tsq, tsk, scale)
        return out

    return [
        B.ArmSpec(
            "FastTree",
            "FastTree",
            fn,
            post,
            f"FastTree.fasttree_decode(Q tiles {tsq}, KV tiles {tsk}, one copy per node), "
            f"PAT artifact {_pat_commit()}",
            True,
            keep=[kt, vt, qd, out, aux],
        )
    ]


def sglang_tree_verify(q, lay, S, mask, post, scale):
    """SGLang's FA-3 target verify for a draft tree: `q` `(B*T, H, D)`, each
    request's last `T` of `S` keys its draft, visible per `mask` `(T, T)`."""
    from sgl_kernel import merge_state_v2
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    T = mask.shape[0]
    Bz = q.shape[0] // T
    dev = q.device
    i32 = torch.int32
    page = lay.page
    qd = q.contiguous()
    cu_q = torch.arange(Bz + 1, device=dev, dtype=i32) * T
    committed = torch.full((Bz,), S - T, device=dev, dtype=i32)

    # each draft query's visible draft keys first, as SGLang's sorted table
    pos = torch.arange(S - T, S, device=dev)
    slots = lay.table[:, pos // page].long() * page + pos % page  # (B, T)
    m = mask.to(dev)
    cols = torch.arange(T, device=dev).expand(T, T)
    order = torch.where(m, cols, cols + T).argsort(1)  # (T, T)
    tab2 = slots[:, None, :].expand(Bz, T, T).gather(2, order.expand(Bz, T, T))
    tab2 = tab2.reshape(Bz * T, T).to(torch.int32).contiguous()
    len2 = m.sum(1).to(torch.int32).repeat(Bz).contiguous()
    cu_q2 = torch.arange(Bz * T + 1, device=dev, dtype=i32)
    HKV, D = lay.kc.shape[2], lay.kc.shape[3]
    k1 = lay.kc.view(-1, 1, HKV, D)
    v1 = lay.vc.view(-1, 1, HKV, D)

    def fn():
        o1, lse1, *_ = flash_attn_with_kvcache(
            q=qd,
            k_cache=lay.kc,
            v_cache=lay.vc,
            page_table=lay.table,
            cache_seqlens=committed,
            cu_seqlens_q=cu_q,
            max_seqlen_q=T,
            softmax_scale=scale,
            causal=False,
            return_softmax_lse=True,
        )
        o2, lse2, *_ = flash_attn_with_kvcache(
            q=qd,
            k_cache=k1,
            v_cache=v1,
            page_table=tab2,
            cache_seqlens=len2,
            cu_seqlens_q=cu_q2,
            max_seqlen_q=1,
            softmax_scale=scale,
            causal=False,
            return_softmax_lse=True,
        )
        o, _ = merge_state_v2(o1, lse1.T.contiguous(), o2, lse2.T.contiguous())
        return o

    return [
        B.ArmSpec(
            "SGLang tree verify",
            "SGLang",
            fn,
            post,
            "sgl_kernel.flash_attn.flash_attn_with_kvcache x2 (committed cache, "
            f"page_size={page}; ancestors, page_size=1) + sgl_kernel.merge_state_v2",
            True,
            keep=[qd, cu_q, committed, tab2, len2, cu_q2],
        )
    ]
