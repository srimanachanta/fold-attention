"""Loss curves of a 1B Llama trained from scratch with each library's attention.

Every run starts from the same initialisation and reads the same batches in
the same order: WikiText-103 tokenised with the Qwen3 tokenizer, cut into
8192-token sequences and shuffled once with a fixed seed. Only the attention
call differs between arms. The rest of the step runs under
`torch.use_deterministic_algorithms`, with cuBLAS's fixed workspace, so two
runs of one arm can differ only through its attention: a deterministic arm
must repeat its curve bit for bit, and a nondeterministic one shows how soon
its runs part.

Training is AdamW over fp32 weights with bf16 autocast, linear warmup and a
cosine decay to a tenth of the peak rate, gradient clipping at 1.0, and two
sequences a step as two micro-batches. Each run records the loss, the
gradient norm before clipping and the learning rate of every step, the
validation loss every `--eval-every` steps, and a digest of its final weights.

`--capture` saves one micro-batch's attention inputs (q, k, v and dO) of the
first, middle and last layers at the given steps of the first run of
`--capture-arm`, for `grad_elements.py`.

    python -m benchmarks.train_curve --steps 2000 --out train_curve
"""

from __future__ import annotations

import os

# cuBLAS picks reduction orders by workspace unless it is fixed before the
# first handle is created
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from benchmarks.harness import training as T
from benchmarks.harness.report import OUT, Report, node_id
from benchmarks.train_e2e import MODELS, attention_fn

MODEL = "llama-3.2-1b"
TOKENIZER = "Qwen/Qwen3-8B"
SEQ = 8192
MICRO = 2
# every arm runs these, in this order; the repeats test run-to-run bits
RUNS = (
    ("FoldAttention", 0),
    ("FoldAttention", 1),
    ("FA-3", 0),
    ("FA-3", 1),
    ("FA-3 det", 0),
    ("FA-4 det", 0),
    ("FA-4", 0),
)
CACHE = T.CACHE


def tokens(split, want, tok):
    """The first `want` tokens of WikiText-103's `split`, cached as uint32."""
    path = CACHE / f"wikitext103_{split}_qwen3_{want}.npy"
    if path.exists():
        return np.load(path)
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    out, n, block = [], 0, 20000
    for i in range(0, len(ds), block):
        lines = [t for t in ds[i : i + block]["text"] if t]
        ids = np.concatenate(
            [np.asarray(x, dtype=np.uint32) for x in tok(lines, add_special_tokens=False).input_ids]
        )
        out.append(ids)
        n += len(ids)
        if n >= want:
            break
    arr = np.concatenate(out)[:want]
    if len(arr) < want:
        raise RuntimeError(f"{split} has {len(arr)} tokens, {want} wanted")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def build_model(vocab):
    from transformers import LlamaConfig, LlamaForCausalLM

    hidden, inter, layers, heads, kv, hd = MODELS[MODEL]
    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=hd,
        max_position_embeddings=SEQ,
        rope_theta=500000.0,
        tie_word_embeddings=True,
        attn_implementation="efa_curve",
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to("cuda").train()


def lr_at(step, steps, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def weights_digest(model):
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        h.update(n.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


class Capture:
    """Saves one micro-batch's attention operands for `layers` while `tag`
    is set."""

    def __init__(self, layers, out_dir):
        self.layers, self.out_dir = set(layers), Path(out_dir)
        self.tag = None

    def __call__(self, module, q, k, v, out):
        """`q`, `k`, `v` as the attention call receives them, `(B, H, S, D)`,
        and its output `(B, S, H, D)`, whose gradient is the dO it will see."""
        layer = getattr(module, "layer_idx", None)
        if self.tag is None or layer not in self.layers:
            return
        path = self.out_dir / f"{self.tag}_layer{layer}.pt"
        saved = dict(
            q=q.transpose(1, 2).detach().clone(),
            k=k.transpose(1, 2).detach().clone(),
            v=v.transpose(1, 2).detach().clone(),
            scale=float(module.scaling),
            layer=layer,
            tag=self.tag,
        )

        def hook(g):
            saved["do"] = g.detach().clone()
            torch.save(saved, path)

        out.register_hook(hook)


def run(arm, index, args, data, val, vocab, fa3, capture):
    from transformers import AttentionInterface

    base = attention_fn(arm, fa3, None)

    def entry(module, query, key, value, *a, **kw):
        # RoPE runs in fp32 against the fp32 residual stream under autocast
        q, k, v = (x.to(torch.bfloat16) for x in (query, key, value))
        out, w = base(module, q, k, v, *a, **kw)
        if capture is not None:
            capture(module, q, k, v, out)
        return out, w

    AttentionInterface.register("efa_curve", entry)
    torch.cuda.reset_peak_memory_stats()
    model = build_model(vocab)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    rest = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [dict(params=decay, weight_decay=0.1), dict(params=rest, weight_decay=0.0)],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=True,
    )
    order = np.random.default_rng(1).permutation(len(data) // SEQ)
    need = args.steps * MICRO
    if len(order) < need:
        raise RuntimeError(f"{len(order)} sequences for {need}")
    losses, norms, lrs, times, evals = [], [], [], [], []
    cap_steps = set(args.capture) if capture else set()
    for step in range(args.steps):
        t0 = time.perf_counter()
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        total = torch.zeros((), device="cuda")
        for m in range(MICRO):
            j = int(order[step * MICRO + m])
            ids = torch.from_numpy(data[j * SEQ : (j + 1) * SEQ].astype(np.int64))[None].cuda()
            last = step == args.steps - 1 and -1 in cap_steps
            if capture is not None and m == 0 and (step in cap_steps or last):
                capture.tag = f"step{step}"
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=ids).loss / MICRO
            loss.backward()
            if capture is not None:
                capture.tag = None
            total += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(total))
        norms.append(float(norm))
        lrs.append(lr)
        times.append(time.perf_counter() - t0)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            evals.append((step, evaluate(model, val)))
        if step % 50 == 0 or step == args.steps - 1:
            ev = f"  val {evals[-1][1]:.4f}" if evals else ""
            print(
                f"  {arm}#{index} step {step:5d} loss {losses[-1]:.4f} "
                f"gnorm {norms[-1]:.3f} lr {lr:.2e} {times[-1] * 1e3:.0f} ms{ev}",
                flush=True,
            )
    digest = weights_digest(model)
    del model, opt
    torch.cuda.empty_cache()
    return dict(
        arm=arm,
        run=index,
        loss=losses,
        loss_hex=[float(x).hex() for x in losses],
        grad_norm=norms,
        lr=lrs,
        val=[dict(step=s, loss=v) for s, v in evals],
        step_ms_median=float(np.median(times[10:]) * 1e3) if len(times) > 10 else None,
        peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30,
        node=node_id(),
        weights_sha256=digest,
    )


@torch.no_grad()
def evaluate(model, val):
    model.eval()
    tot = 0.0
    for i in range(len(val) // SEQ):
        ids = torch.from_numpy(val[i * SEQ : (i + 1) * SEQ].astype(np.int64))[None].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += float(model(input_ids=ids, labels=ids).loss)
    model.train()
    return tot / (len(val) // SEQ)


def compare(a, b):
    """Where two runs' loss curves part, and by how much."""
    la, lb = np.array(a["loss"]), np.array(b["loss"])
    n = min(len(la), len(lb))
    diff = np.nonzero(np.array(a["loss_hex"][:n]) != np.array(b["loss_hex"][:n]))[0]
    d = np.abs(la[:n] - lb[:n])
    return dict(
        runs=[f"{a['arm']}#{a['run']}", f"{b['arm']}#{b['run']}"],
        bit_identical=len(diff) == 0 and a["weights_sha256"] == b["weights_sha256"],
        first_differing_step=int(diff[0]) if len(diff) else None,
        max_abs_loss_diff=float(d.max()),
        mean_abs_loss_diff_last100=float(d[-100:].mean()),
        final_loss=[float(la[-100:].mean()), float(lb[-100:].mean())],
        final_val=[a["val"][-1]["loss"], b["val"][-1]["loss"]],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--val-seqs", type=int, default=8)
    p.add_argument(
        "--runs", nargs="+", default=[f"{a}#{i}" for a, i in RUNS], help="arm#index, in order"
    )
    p.add_argument("--capture", type=int, nargs="*", default=[500, -1], help="-1 is the last step")
    p.add_argument("--capture-arm", default="FA-3 det")
    p.add_argument("--capture-dir", default=str(CACHE / "grad_states"))
    p.add_argument(
        "--resume",
        action="store_true",
        help="keep the finished runs already in the output file and run only the rest",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()
    kept = []
    if args.resume and (OUT / f"{args.out}.json").exists():
        # a file rewritten on another node can come back padded with NULs
        old = json.loads((OUT / f"{args.out}.json").read_bytes().rstrip(b"\0"))
        for r in old["rows"]:
            if r.get("kind") == "run" and len(r.get("loss", ())) == args.steps:
                kept.append(dict(r, node=r.get("node", old["env"]["node"])))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    vocab = -(-len(tok) // 128) * 128
    data = tokens("train", args.steps * MICRO * SEQ + SEQ, tok)
    val = tokens("validation", args.val_seqs * SEQ, tok)
    fa3 = T.load_interface("fa3")
    runs = [(r.rsplit("#", 1)[0], int(r.rsplit("#", 1)[1])) for r in args.runs]
    rep = Report(
        args.out,
        model=MODEL,
        spec=MODELS[MODEL],
        vocab=vocab,
        tokenizer=TOKENIZER,
        data="Salesforce/wikitext wikitext-103-raw-v1 train, shuffled 8192-token sequences",
        seq=SEQ,
        micro_batches=MICRO,
        steps=args.steps,
        optimizer=dict(
            name="AdamW",
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            warmup=args.warmup,
            schedule="cosine to 0.1 x peak",
            clip=1.0,
            weights="fp32",
            autocast="bf16",
        ),
        deterministic_algorithms=True,
        cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        runs=args.runs,
    )
    results = []
    for r in kept:
        results.append(r)
        rep.add(**r)
    done = {(r["arm"], r["run"]) for r in kept}
    captured = False
    for arm, index in runs:
        if (arm, index) in done:
            print(f"\n=== {arm} run {index}: kept from the earlier file ===", flush=True)
            continue
        print(f"\n=== {arm} run {index} ===", flush=True)
        capture = None
        if arm == args.capture_arm and not captured and args.capture:
            Path(args.capture_dir).mkdir(parents=True, exist_ok=True)
            layers = MODELS[MODEL][2]
            capture = Capture((0, layers // 2 - 1, layers - 1), args.capture_dir)
            captured = True
        try:
            r = run(arm, index, args, data, val, vocab, fa3, capture)
            results.append(r)
            rep.add(kind="run", **r)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rep.add(kind="run", arm=arm, run=index, error=repr(e)[:400])
            torch.cuda.empty_cache()
    for i, a in enumerate(results):
        for b in results[i + 1 :]:
            c = compare(a, b)
            rep.add(kind="compare", **c)
            print(
                f"  {c['runs'][0]:16s} vs {c['runs'][1]:16s} "
                f"{'bit-identical' if c['bit_identical'] else 'differ from step ' + str(c['first_differing_step'])}"
                f"  max |dloss| {c['max_abs_loss_diff']:.2e}  final val {c['final_val'][0]:.4f} / "
                f"{c['final_val'][1]:.4f}",
                flush=True,
            )
    rep.write()


if __name__ == "__main__":
    main()
