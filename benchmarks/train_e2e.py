"""A training step of a whole model, with each library's attention.

The models are Llama-architecture decoders at random initialisation, built
from their configurations, so nothing is downloaded: every arm runs the same
weights and the same batch, and only the attention call differs. A step is
the forward, the cross-entropy loss and the backward, without the optimizer,
timed eagerly as a training loop runs it.

Attention goes through the HF attention registry: each arm installs one
entry that calls its library's autograd function on the module's already
rotated `(B, H, S, D)` query and `(B, H_KV, S, D)` key and value. FA-3 and
DASH are one module name, so `--impl dash` times DASH (with FA-4 and
FoldAttention to join the runs) in its own process.

    python -m benchmarks.train_e2e --out train_e2e_fa3
    python -m benchmarks.train_e2e --impl dash --out train_e2e_dash
"""

from __future__ import annotations

import argparse
import traceback

import torch

from benchmarks.harness import training as T
from benchmarks.harness.report import Report
from benchmarks.harness.timing import measure, paired_ratio

# name -> (hidden, intermediate, layers, heads, kv heads, head dim)
MODELS = {
    "llama-3.2-1b": (2048, 8192, 16, 32, 8, 64),
    "llama-3.1-8b (8 of 32 layers)": (4096, 14336, 8, 32, 8, 128),
}
# (batch, sequence): 16K tokens a step, FlashAttention's benchmark scale
BATCHES = ((2, 8192), (4, 4096))
VOCAB = 32000


def attention_fn(name, fa3, dash):
    """The registry entry for arm `name`: `(module, q, k, v, mask, ...)` with
    `(B, H, S, D)` operands, returning `(B, S, H, D)`."""
    if name == "FoldAttention":
        from fold_attention import fold_attn_func as f

        def call(q, k, v, sc):
            return f(q, k, v, softmax_scale=sc, causal=True)
    elif name in ("FA-3", "FA-3 det", "DASH"):
        lib = dash if name == "DASH" else fa3
        if lib is None:
            raise RuntimeError(f"{name} is not loaded in this process")
        det = name != "FA-3"

        def call(q, k, v, sc):
            out = lib.flash_attn_func(q, k, v, softmax_scale=sc, causal=True, deterministic=det)
            return out[0] if isinstance(out, tuple) else out
    elif name in ("FA-4", "FA-4 det"):
        from flash_attn.cute import flash_attn_func as fa4

        det = name == "FA-4 det"

        def call(q, k, v, sc):
            out = fa4(q, k, v, softmax_scale=sc, causal=True, deterministic=det)
            return out[0] if isinstance(out, tuple) else out
    elif name == "cuDNN":
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def call(q, k, v, sc):
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=True,
                    scale=sc,
                    enable_gqa=True,
                ).transpose(1, 2)
    else:
        raise ValueError(f"unknown arm {name!r}")

    def entry(module, query, key, value, attention_mask=None, scaling=None, **kw):
        q, k, v = (x.transpose(1, 2) for x in (query, key, value))
        return call(q, k, v, scaling), None

    return entry


def build_model(spec):
    from transformers import LlamaConfig, LlamaForCausalLM

    hidden, inter, layers, heads, kv, hd = spec
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=hd,
        max_position_embeddings=32768,
        attn_implementation="efa_bench",
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to("cuda", torch.bfloat16).train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["fa3", "dash"], default="fa3")
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from transformers import AttentionInterface

    names = (
        ["FA-3", "FA-3 det", "FA-4", "FA-4 det", "cuDNN", "FoldAttention"]
        if args.impl == "fa3"
        else ["DASH", "FA-4", "FA-4 det", "FoldAttention"]
    )
    fa3 = T.load_interface("fa3") if args.impl == "fa3" else None
    dash = T.load_interface("dash") if args.impl == "dash" else None
    entries = {}
    for n in names:
        try:
            entries[n] = attention_fn(n, fa3, dash)
        except Exception as e:  # noqa: BLE001
            print(f"  dead  {n}: {repr(e)[:140]}", flush=True)
    rep = Report(args.out, impl=args.impl, models=MODELS, batches=BATCHES, vocab=VOCAB)
    # the config names the entry, so it has to exist before a model is built
    AttentionInterface.register("efa_bench", next(iter(entries.values())))

    for mname, spec in MODELS.items():
        model = build_model(spec)
        for Bz, S in BATCHES:
            label = f"{mname} B{Bz} S{S}"
            print(f"\n=== {label} ===", flush=True)
            g = torch.Generator(device="cuda").manual_seed(1)
            ids = torch.randint(0, VOCAB, (Bz, S), generator=g, device="cuda")
            try:
                steps = {}
                for n, entry in entries.items():

                    def step(entry=entry, model=model, ids=ids):
                        AttentionInterface.register("efa_bench", entry)
                        model.zero_grad(set_to_none=True)
                        model(input_ids=ids, labels=ids).loss.backward()

                    try:
                        step()
                        torch.cuda.synchronize()
                        steps[n] = step
                    except Exception as e:  # noqa: BLE001
                        print(f"  dead  {n}: {repr(e)[:140]}", flush=True)
                res = measure(
                    steps, graphable=dict.fromkeys(steps, False), rounds=args.rounds, cold=False
                )
                rows = []
                for n in steps:
                    rows.append(
                        dict(
                            arm=n,
                            cold=res[n],
                            over_ours=paired_ratio(res, n, "FoldAttention")
                            if "FoldAttention" in res
                            else None,
                        )
                    )
                    print(
                        f"  {n:14s} {res[n]['us'] / 1e3:9.2f} ms  "
                        f"{res[n]['us'] / res['FoldAttention']['us']:.3f}x ours",
                        flush=True,
                    )
                rep.add(model=mname, spec=spec, B=Bz, S=S, label=label, arms=rows)
                del steps
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rep.add(model=mname, B=Bz, S=S, label=label, error=repr(e)[:400])
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    rep.write()


if __name__ == "__main__":
    main()
