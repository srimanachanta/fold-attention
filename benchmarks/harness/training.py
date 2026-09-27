"""Training-attention arms over one set of inputs: the backward alone and the
forward with it.

Every backward arm reads the same `q, k, v, o, dO` and LSE, from FA-4's
forward, so their gradients are comparable bit for bit and against one FP32
reference. FA-3 and DASH are both the module `flash_attn_interface`, so a
process loads one of them (`load_interface`); FA-4, cuDNN and FoldAttention
share either process, and FA-4 is the control that joins the two.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

# The FA-3 build with a backward, and DASH's, each a directory holding its
# `flash_attn_interface`. Unset, whatever `flash_attn_interface` imports is used.
FA3_PATH = os.environ.get("EFA_FA3_BWD")
# downloaded data and the training captures `grad_elements` reads
CACHE = Path(os.environ.get("EFA_TRAIN_CACHE", Path.home() / ".cache" / "efa_train"))
DASH_PATH = os.environ.get("EFA_DASH")


def load_interface(impl: str):
    """Import `flash_attn_interface` from the FA-3 (`impl="fa3"`) or DASH
    build; returns the module, or None if it does not import."""
    path = FA3_PATH if impl == "fa3" else DASH_PATH
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        import flash_attn_interface as m

        return m
    except Exception as e:  # noqa: BLE001
        print(f"  {impl} unavailable: {repr(e)[:140]}", flush=True)
        return None


@dataclass
class Inputs:
    """Dense operands are `(B, S, H, D)` and `lse` `(B, H, S)`; packed ones
    `(total, H, D)` with `cu` `(B + 1,)` and `lse` `(H, total)`."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    do: torch.Tensor
    lse: torch.Tensor
    causal: bool
    scale: float
    lens: list
    cu: torch.Tensor | None = None

    @property
    def varlen(self):
        return self.cu is not None

    @property
    def max_s(self):
        return max(self.lens)


def make(B, H, HKV, S, D, causal, seed=0) -> Inputs:
    from flash_attn.cute.interface import _flash_attn_fwd

    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, S, H, D, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, HKV, D, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, HKV, D, generator=g, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, S, H, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sc = 1.0 / math.sqrt(D)
    o, lse = _flash_attn_fwd(q, k, v, softmax_scale=sc, causal=causal, return_lse=True)[:2]
    assert lse is not None
    return Inputs(q, k, v, o, do, lse.contiguous(), causal, sc, [S] * B)


def make_varlen(lens, H, HKV, D, seed=0) -> Inputs:
    from flash_attn.cute.interface import _flash_attn_fwd

    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    q = torch.randn(T, H, D, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(T, HKV, D, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(T, HKV, D, generator=g, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(T, H, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sc = 1.0 / math.sqrt(D)
    o, lse = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(lens),
        max_seqlen_k=max(lens),
        causal=True,
        softmax_scale=sc,
        return_lse=True,
    )[:2]
    assert lse is not None
    return Inputs(q, k, v, o, do, lse.contiguous(), True, sc, list(lens), cu)


def _segments(inp: Inputs):
    """`(q, k, v, do)` of each request, `(S, H, D)`."""
    if inp.varlen:
        assert inp.cu is not None
        c = inp.cu.tolist()
        for a, b in itertools.pairwise(c):
            yield slice(a, b), (inp.q[a:b], inp.k[a:b], inp.v[a:b], inp.do[a:b])
    else:
        for b in range(inp.q.shape[0]):
            yield b, (inp.q[b], inp.k[b], inp.v[b], inp.do[b])


def reference_grads(inp: Inputs):
    """FP32 dQ, dK, dV of softmax attention, in the inputs' layout, one
    (request, query head) at a time so a 16K score matrix is the largest
    temporary."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        dq = torch.zeros_like(inp.q, dtype=torch.float32)
        dk = torch.zeros_like(inp.k, dtype=torch.float32)
        dv = torch.zeros_like(inp.v, dtype=torch.float32)
        H, HKV = inp.q.shape[-2], inp.k.shape[-2]
        G = H // HKV
        for idx, (q, k, v, do) in _segments(inp):
            S = q.shape[0]
            mask = (
                torch.ones(S, S, device=q.device, dtype=torch.bool).tril() if inp.causal else None
            )
            for h in range(H):
                hk = h // G
                qh, kh, vh, doh = (
                    q[:, h].float(),
                    k[:, hk].float(),
                    v[:, hk].float(),
                    do[:, h].float(),
                )
                s = (qh @ kh.T) * inp.scale
                if mask is not None:
                    s = s.masked_fill(~mask, float("-inf"))
                p = s.softmax(-1)
                del s
                oh = p @ vh
                dp = doh @ vh.T
                ds = p * (dp - (doh * oh).sum(-1, keepdim=True))
                del dp
                dq[idx, ..., h, :] = (ds @ kh) * inp.scale
                dk[idx, ..., hk, :] += (ds.T @ qh) * inp.scale
                dv[idx, ..., hk, :] += p.T @ doh
                del p, ds
        return dq, dk, dv
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def rel_l2(x, ref) -> float:
    return float((x.float() - ref).norm() / ref.norm())


@dataclass
class Arm:
    """`run()` computes the gradients; `grads()` returns them as `(dq, dk, dv)`
    in the inputs' layout."""

    name: str
    family: str
    run: Callable
    grads: Callable
    provenance: str
    deterministic: bool
    graphable: bool = True
    held: tuple = ()


def backward_arms(inp: Inputs, names, fa3=None, dash=None) -> dict[str, Arm]:
    """The backward arms in `names` that build and run once."""
    out = {}
    for n in names:
        try:
            arm = _backward_arm(n, inp, fa3, dash)
            arm.run()
            torch.cuda.synchronize()
            out[n] = arm
        except Exception as e:  # noqa: BLE001
            print(f"  dead  {n:18s} {repr(e)[:140]}", flush=True)
    return out


def _backward_arm(name, inp: Inputs, fa3, dash) -> Arm:
    q, k, v, o, do, lse, sc, causal = (
        inp.q,
        inp.k,
        inp.v,
        inp.o,
        inp.do,
        inp.lse,
        inp.scale,
        inp.causal,
    )
    max_s = inp.max_s if inp.varlen else None
    if name in ("FA-3", "FA-3 det"):
        if fa3 is None:
            raise RuntimeError("FA-3 is not loaded in this process")
        det = name == "FA-3 det"
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))

        def run3():
            fa3._flash_attn_backward(
                do,
                q,
                k,
                v,
                o,
                lse,
                cu_seqlens_q=inp.cu,
                cu_seqlens_k=inp.cu,
                max_seqlen_q=max_s,
                max_seqlen_k=max_s,
                dq=dq,
                dk=dk,
                dv=dv,
                softmax_scale=sc,
                is_causal=causal,
                deterministic=det,
            )

        return Arm(
            name,
            "FA-3",
            run3,
            lambda: (dq, dk, dv),
            f"flash_attn_interface._flash_attn_backward(deterministic={det}), the EFA_FA3_BWD build",
            det,
            held=(dq, dk, dv),
        )
    if name == "DASH":
        if dash is None:
            raise RuntimeError("DASH is not loaded in this process")
        if inp.varlen:
            # DASH schedules dense batches only: its shift schedules read the
            # sequence length from shape_K, the token total under varlen, and
            # its deterministic varlen backward never returns
            raise RuntimeError("DASH has no varlen schedule")
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))

        def run_dash():
            dash._flash_attn_backward(
                do,
                q,
                k,
                v,
                o,
                lse,
                inp.cu,
                inp.cu,
                None,
                None,
                max_s,
                max_s,
                dq,
                dk,
                dv,
                sc,
                causal,
                deterministic=True,
            )

        return Arm(
            name,
            "DASH",
            run_dash,
            lambda: (dq, dk, dv),
            "DASH flash_attn_interface._flash_attn_backward(deterministic=True), the "
            "EFA_DASH build (d87bcc9 with dash_d87bcc9_causal_mask.patch)",
            True,
            held=(dq, dk, dv),
        )
    if name in ("FA-4", "FA-4 det"):
        from flash_attn.cute.interface import _flash_attn_bwd

        det = name == "FA-4 det"
        res = []

        def run4():
            res[:] = _flash_attn_bwd(
                q,
                k,
                v,
                o,
                do,
                lse,
                softmax_scale=sc,
                causal=causal,
                deterministic=det,
                cu_seqlens_q=inp.cu,
                cu_seqlens_k=inp.cu,
                max_seqlen_q=max_s,
                max_seqlen_k=max_s,
            )[:3]

        return Arm(
            name,
            "FA-4",
            run4,
            lambda: tuple(res),
            f"flash_attn.cute.interface._flash_attn_bwd(deterministic={det})",
            det,
        )
    if name == "FoldAttention":
        from exact_fold_attn.backward import prepare_backward

        if inp.varlen:
            launch = prepare_backward(
                q,
                k,
                v,
                o,
                do,
                lse,
                causal=True,
                softmax_scale=sc,
                cu_seqlens=inp.cu,
                max_seqlen=inp.max_s,
            )
            grads = launch.run
        else:
            t = [x.transpose(1, 2) for x in (q, k, v, o, do)]
            launch = prepare_backward(
                t[0], t[1], t[2], t[3], t[4], lse, causal=causal, softmax_scale=sc
            )

            def grads():
                return tuple(x.transpose(1, 2) for x in launch.run())

        return Arm(
            name,
            "FoldAttention",
            launch.run,
            grads,
            f"exact_fold_attn.backward.prepare_backward(plan={launch.plan})",
            True,
            held=(launch,),
        )
    if name == "cuDNN":
        if inp.varlen:
            raise RuntimeError("SDPA takes no cu_seqlens")
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        qh, kh, vh = (x.transpose(1, 2).detach().requires_grad_(True) for x in (q, k, v))
        doh = do.transpose(1, 2)
        G = qh.shape[1] // kh.shape[1]
        how = "enable_gqa=True"
        try:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                out = F.scaled_dot_product_attention(
                    qh, kh, vh, is_causal=causal, scale=sc, enable_gqa=G > 1
                )
        except RuntimeError:
            how = "K/V repeated to H heads inside the graph"
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                out = F.scaled_dot_product_attention(
                    qh,
                    kh.repeat_interleave(G, 1),
                    vh.repeat_interleave(G, 1),
                    is_causal=causal,
                    scale=sc,
                )
        res = []

        def run_cudnn():
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                res[:] = torch.autograd.grad(out, (qh, kh, vh), doh, retain_graph=True)

        return Arm(
            name,
            "cuDNN",
            run_cudnn,
            lambda: tuple(x.transpose(1, 2) for x in res),
            f"torch SDPA backward, SDPBackend.CUDNN_ATTENTION, {how}; its own forward's O",
            False,
            graphable=False,
            held=(qh, kh, vh, out),
        )
    raise ValueError(f"unknown arm {name!r}")


def train_arms(inp: Inputs, names, fa3=None, dash=None) -> dict[str, Arm]:
    """Forward plus backward through each library's autograd entry point,
    timed eagerly as a training step calls it."""
    out = {}
    for n in names:
        try:
            arm = _train_arm(n, inp, fa3, dash)
            arm.run()
            torch.cuda.synchronize()
            out[n] = arm
        except Exception as e:  # noqa: BLE001
            print(f"  dead  {n:18s} {repr(e)[:140]}", flush=True)
    return out


def _train_arm(name, inp: Inputs, fa3, dash) -> Arm:
    if inp.varlen:
        raise RuntimeError("the training comparison is dense")
    q, k, v = (x.detach().requires_grad_(True) for x in (inp.q, inp.k, inp.v))
    do, causal, sc = inp.do, inp.causal, inp.scale
    if name == "FoldAttention":
        from exact_fold_attn import fold_attn_func

        def f():
            return fold_attn_func(q, k, v, softmax_scale=sc, causal=causal)

        prov, det = "exact_fold_attn.fold_attn_func (FA-4 forward, folded backward)", True
    elif name in ("FA-3", "FA-3 det"):
        if fa3 is None:
            raise RuntimeError("FA-3 is not loaded in this process")
        det = name == "FA-3 det"

        def f():
            return fa3.flash_attn_func(q, k, v, softmax_scale=sc, causal=causal, deterministic=det)

        prov = f"flash_attn_interface.flash_attn_func(deterministic={det})"
    elif name == "DASH":
        if dash is None:
            raise RuntimeError("DASH is not loaded in this process")
        det = True

        def f():
            return dash.flash_attn_func(
                q, k, v, softmax_scale=sc, causal=causal, deterministic=True
            )

        prov = "DASH flash_attn_interface.flash_attn_func(deterministic=True)"
    elif name in ("FA-4", "FA-4 det"):
        from flash_attn.cute import flash_attn_func as fa4

        det = name == "FA-4 det"

        def f():
            return fa4(q, k, v, softmax_scale=sc, causal=causal, deterministic=det)

        prov = f"flash_attn.cute.flash_attn_func(deterministic={det})"
    elif name == "cuDNN":
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        G = q.shape[2] // k.shape[2]
        det = False

        def f():
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2).repeat_interleave(G, 1),
                    v.transpose(1, 2).repeat_interleave(G, 1),
                    is_causal=causal,
                    scale=sc,
                ).transpose(1, 2)

        prov = "torch SDPA, SDPBackend.CUDNN_ATTENTION, K/V repeated to H heads"
    else:
        raise ValueError(f"unknown arm {name!r}")

    def step():
        out = f()
        out = out[0] if isinstance(out, tuple) else out
        return torch.autograd.grad(out, (q, k, v), do)

    return Arm(name, name.split(" ")[0], step, lambda: (), prov, det, graphable=False)
