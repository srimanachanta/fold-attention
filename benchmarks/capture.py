"""Capture post-RoPE q, k and v at several layers of one model in one prefill.

Writes one file per layer in the format `harness/captures.py` reads: {"q": (1,
H, S, D), "k": (1, H_KV, S, D), "v": (1, H_KV, S, D), "hd": D}, bf16, named by
a hash of the model, sequence length and layer, which are the names
`harness/captures.py` expects. A prefill capture is a decode trace: row p of q
is the query at step p, against exactly the keys that step sees.

Layers are sampled by index across the stack, never "the first N": attention
in the first few layers of a trained model is nearly uniform and says nothing
about the rest. The suite's five captures are, over WikiText-103 at 32K tokens:

    python -m benchmarks.capture --model Qwen/Qwen3-30B-A3B --layers 24
    python -m benchmarks.capture --model THUDM/glm-4-9b-chat-hf --layers 8 28
    python -m benchmarks.capture --model openai/gpt-oss-20b --layers 9 21 --impl flex_attention

**The hook is the attention interface, not SDPA.** Patching
`scaled_dot_product_attention` only sees models whose implementation calls it,
and the ones worth capturing at other head dims do not: gpt-oss carries an
attention sink and routes through its own entry in `ALL_ATTENTION_FUNCTIONS`.
Wrapping the registry entry sees every implementation, and it hands over the
module as well, which is what makes the two corrections below possible.

**The scale is folded into q.** `harness/captures.py` reconstructs logits as
`q . k * LOG2E / sqrt(D)`, which is the right temperature only for a model
that scales by `1/sqrt(D)`. Granite scales by an `attention_multiplier` that
is nothing like it, and a capture that ignored that would hand the kernel
logits eight times too narrow, and every truncation depth would then measure
the capture rather than the model. `q` is stored premultiplied by
`scaling * sqrt(D)`, so the reconstruction is the model's own temperature
whatever it scales by.

**Sliding-window layers are refused, not captured.** A layer that attends a
window is not a decode trace over the whole cache, and half of gpt-oss's
layers are one. `--layers` naming a windowed layer says so and skips it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch


def real_text(want):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    buf, n = [], 0
    for r in ds:
        t = r["text"]
        if t.strip():
            buf.append(t)
            n += len(t) // 4
            if n > want * 1.5:
                break
    return "\n".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--seq", type=int, default=32768)
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 16, 36, 44])
    ap.add_argument(
        "--out", default=os.environ.get("EFA_CAPTURE_DIR", str(Path.home() / "efa_captures"))
    )
    ap.add_argument("--impl", default="sdpa", help="attention implementation to run and wrap")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation=a.impl, device_map="cuda"
    ).eval()
    hd = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    hkv = model.config.num_key_value_heads
    ids = tok(
        real_text(a.seq), return_tensors="pt", truncation=True, max_length=a.seq
    ).input_ids.to(model.device)
    S = ids.shape[1]
    seen, got, skipped = [], {}, {}

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = model.config._attn_implementation

    def record(module, q, k, v, scaling=None, **kw):
        i = len(seen)
        seen.append(i)
        if i not in a.layers:
            return
        # A model that interleaves windowed and full layers (gpt-oss does,
        # every other one) still carries `sliding_window` on every module, so
        # the attribute alone marks the full layers windowed. The layer's
        # declared type is what separates them; only fall back to the
        # attribute when there is no such declaration.
        at = getattr(module, "attention_type", None) or getattr(module, "layer_type", None)
        win = getattr(module, "sliding_window", None)
        if at is not None and "sliding" not in str(at):
            win = None
        if win and win < S:
            # a windowed layer attends `win` keys, so its rows are not a
            # decode trace over the cache the kernel is given
            skipped[i] = f"sliding window {win}"
            return
        # the interface may be handed KV already expanded to every query head
        # (`repeat_kv` before the call) or not; the unique heads are what the
        # cache holds either way
        g = max(1, k.shape[1] // hkv)
        ku, vu = (k[:, ::g], v[:, ::g]) if k.shape[1] != hkv else (k, v)
        # the suite rebuilds the logit as q.k * LOG2E / sqrt(D); fold the
        # model's own scale in so that is its temperature and not 1/sqrt(D)
        sc = float(scaling) if scaling is not None else hd**-0.5
        qs = q.float() * (sc * hd**0.5)
        got[i] = (
            qs.to(q.dtype).detach().to("cpu", copy=True),
            ku.detach().to("cpu", copy=True),
            vu.detach().to("cpu", copy=True),
            sc,
        )

    # `eager` is not in the registry: a module calls the
    # `eager_attention_forward` bound in its own modeling file, so that symbol
    # is what has to be wrapped. Everything else goes through the registry.
    import sys as _sys

    mod = None
    if impl == "eager":
        attn = model.model.layers[0].self_attn
        mod = _sys.modules[type(attn).__module__]
        real_fn = mod.eager_attention_forward
    elif impl in ALL_ATTENTION_FUNCTIONS:
        real_fn = ALL_ATTENTION_FUNCTIONS[impl]
    else:
        raise SystemExit(
            f"attention implementation {impl!r} is neither 'eager' nor in the "
            f"registry {sorted(ALL_ATTENTION_FUNCTIONS)}"
        )

    def patched(module, q, k, v, *args, **kw):
        record(module, q, k, v, kw.get("scaling"))
        return real_fn(module, q, k, v, *args, **kw)

    if mod is not None:
        mod.eager_attention_forward = patched
    else:
        ALL_ATTENTION_FUNCTIONS[impl] = patched
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        if mod is not None:
            mod.eager_attention_forward = real_fn
        else:
            ALL_ATTENTION_FUNCTIONS[impl] = real_fn
    print(f"{a.model}: {len(seen)} attention calls at S={S}, impl {impl}", flush=True)
    lt = getattr(model.config, "layer_types", None)
    if lt:
        full = [i for i, t in enumerate(lt) if "sliding" not in str(t)]
        print(f"  full-attention layers: {full}", flush=True)
    for i in a.layers:
        if i in skipped:
            print(f"  layer {i} skipped: {skipped[i]}", flush=True)
            continue
        if i not in got:
            print(f"  layer {i} never fired", flush=True)
            continue
        q, k, v, sc = got[i]
        # harness/captures.py names the files this exact key hashes to
        key = f"refund|{a.model}|{a.seq}|{i}"
        p = out / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".pt")
        torch.save(
            {"q": q, "k": k, "v": v, "hd": hd, "model": a.model, "layer": i, "scaling": sc}, p
        )
        print(
            f"  layer {i}: q {tuple(q.shape)} k {tuple(k.shape)} "
            f"scale {sc:.6g} (1/sqrt(D) is {hd**-0.5:.6g}) -> {p}",
            flush=True,
        )


if __name__ == "__main__":
    main()
