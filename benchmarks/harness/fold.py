"""FoldAttention's arms over a prompt-written `FoldKVCache`.

Each member is timed at two call boundaries:

- `decode`: the attention launch and its split combine, the boundary every
  baseline's decode call has. Only this one is divided into a baseline's time.
- `step`: Q's planes, the token's append, the reference, the decode and the
  combine, which is what a serving stack pays per token.

The scored output is the timed launch's own output, in bf16 like every
baseline's.
"""

from __future__ import annotations

import torch

from exact_fold_attn import FoldKVCache


def prompt(shape, lens):
    """The packed prompt (every key but the last) and the last token, which
    the member's first step appends."""
    B, k, v = shape["B"], shape["k"], shape["v"]
    dev = k.device
    n = [int(x) for x in lens]
    cu = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.cumsum(torch.tensor([x - 1 for x in n], device=dev), 0)
    return dict(
        kp=torch.cat([k[b, :, : n[b] - 1].transpose(0, 1) for b in range(B)]),
        vp=torch.cat([v[b, :, : n[b] - 1].transpose(0, 1) for b in range(B)]),
        kn=torch.stack([k[b, :, n[b] - 1] for b in range(B)]).contiguous(),
        vn=torch.stack([v[b, :, n[b] - 1] for b in range(B)]).contiguous(),
        cu=cu,
        lens=[x - 1 for x in n],
    )


def name_of(depth, v8):
    return ("Fold dense" if depth is None else f"Fold T={depth:g}") + (" v8" if v8 else "")


def kernel_of(config) -> str:
    """The decode kernel a prepared launch runs: `decode.packed`'s 128-key
    tiles, which `config.pack_for` picks for groups of at most four rows at
    D = 64 on a bf16 V, or `decode.kernel`'s 64-key tiles."""
    return "packed 128-key tile" if config.pack == 2 else "64-key tile"


def member(
    shape,
    lens,
    pr,
    *,
    depth,
    v8=False,
    page=128,
    split=None,
    tail="auto",
    refine_k=None,
    weight_terms=None,
    refine_v=None,
    name=None,
    chunk=None,
):
    """One prepared member. Its cache holds the prompt and one real step, so
    it stands at `lens`. `refine_k=refine_v=-1e4` is the capacity member,
    which never reads a second plane and so needs none stored."""
    B, H, HKV, D, S = shape["B"], shape["H"], shape["HKV"], shape["D"], shape["S"]
    G = H // HKV
    fa = FoldKVCache(
        B,
        H,
        HKV,
        D,
        max_len=S + 64,
        page_size=page,
        depth=depth,
        v8=v8,
        split=split,
        tail=tail,
        refine_k=refine_k,
        refine_v=refine_v,
        weight_terms=weight_terms,
        chunk=chunk,
    )
    fa.write_prompt(pr["kp"], pr["vp"], pr["cu"], pr["lens"])
    fa.decode(shape["q"], pr["kn"], pr["vn"])
    run = fa.prepare_replay_decode(out_dtype=torch.bfloat16)
    out, _, counts = run()
    out = out.reshape(B, HKV, G, D).float().clone()
    step = fa.prepare_replay_step(shape["q"], pr["kn"], pr["vn"], at=fa.seq_lens - 1)
    assert run.config is not None
    sp = run.config.split
    kernel = kernel_of(run.config)
    return dict(
        name=name or name_of(depth, v8),
        fa=fa,
        depth=depth,
        v8=v8,
        out=out,
        decode=run,
        step=step,
        split=sp,
        kernel=kernel,
        **fractions(counts, lens, HKV),
        provenance=(
            f"exact_fold_attn.FoldKVCache(page_size={page}, depth={depth}, v8={v8}, "
            f"tail={tail!r}, refine_k={refine_k}, refine_v={refine_v}, "
            f"weight_terms={weight_terms}, chunk={chunk}) split={sp}, {kernel}, "
            "bf16 out"
        ),
    )


def fractions(counts, lens, HKV):
    """What the kernel counted, per key row considered: `live` read V, and
    `refined` read K's second plane."""
    c = counts.float()
    total = max(float(sum(int(x) for x in lens)) * HKV, 1.0)
    return dict(live=float(c[:, 0].sum()) / total, refined=float(c[:, 1].sum()) / total)


def bytes_read(m, keys, D):
    """Bytes one decode reads from the cache, from the kernel's counters:
    plane A of K and its bf16 scale for every key, plane B for the refined
    keys, and V for the live keys (2 D in bf16; D for an 8-bit V's first
    plane, a lower bound since its second plane is gated again and not
    counted)."""
    v = D if m["v8"] else 2 * D
    return keys * (D + 2 + m["refined"] * D + m["live"] * v)
