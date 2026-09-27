"""End-to-end decode of a whole model, and what finite depth does to its answers.

Qwen3-8B (36 layers, 32 query heads over 8 KV heads, head dim 128) decodes
one token per request per step. Every arm runs the same weights, the same
prompts and the same step: embedding, every layer's projections, QK norms,
RoPE and MLP, the final norm and the LM head, captured whole in one CUDA
graph. Only the attention call and the cache it reads differ. The timed step
writes the token at a fixed position, as a steady-state decode step does, so
it can be replayed.

- `speed`: inter-token latency and tokens per second at 16K tokens per step
  batch (B=32 at 8K, 16 at 16K, 8 at 32K context), cold L2, for FA-3,
  FlashInfer, cuDNN and the Fold members.

The quality modes decode for real in every arm of `QUALITY_ARMS` (FA-3,
FlashInfer, cuDNN, FlashInfer over an e4m3 cache, and the Fold members),
each appending its tokens to its own cache:

- `nll`: teacher-forced documents. Each arm decodes `--tf-tokens` tokens of
  a document after a prompt of each context and is scored on the next token:
  mean NLL, its difference from FA-3's, KL(FA-3 || arm) and top-1 agreement.
- `diverge`: free-running greedy generation, `--gen-tokens` per prompt: the
  first position where each arm's tokens differ from FA-3's and the fraction
  that match, beside FlashInfer's and cuDNN's, which bound how far BF16
  kernels already disagree with each other.
- `ruler`: RULER-style synthetic retrieval (several keys, several values,
  several queries, variable tracking), exact match.
- `longbench`: LongBench v1's sixteen English tasks with the benchmark's own
  prompts, truncation, generation lengths and metrics
  (`harness/longbench.py`).

Every live Fold arm also records each row's reference headroom, the
log-sum-exp over Z (`log2` of the denominator the kernel returns), and how
many rows fall outside the range certificate's window.

`--yarn` extends Qwen3-8B past its 32K training context with the YaRN
scaling its model card gives (factor 4 over 32768 positions), for the 64K
and 128K runs.

Prompts are WikiText-103 text, each request its own passage. The prompt's K
and V are computed once with FA-4 and staged on the host; each arm's cache is
written from them, so every arm attends over the same prompt.

    python -m benchmarks.serve --out serve
    python -m benchmarks.serve --parts nll --out serve_nll
    python -m benchmarks.serve --parts diverge --out serve_diverge
    python -m benchmarks.serve --parts ruler --out serve_ruler
    python -m benchmarks.serve --parts longbench --out serve_longbench
    python -m benchmarks.serve --yarn --parts nll --contexts 65536 131072 --out serve_nll_long
"""

from __future__ import annotations

import argparse
import math
import random
import re
import statistics
import string
import time
import traceback

import torch

from benchmarks.harness.report import Report
from benchmarks.harness.timing import L2Flush, measure

MODEL = "Qwen/Qwen3-8B"
SPEED = ((32, 8192), (16, 16384), (8, 32768))
QUALITY_CONTEXTS = (8192, 16384, 32768)
PAGE = 128
WS = 256 << 20
# the fixed members; the capacity members never read a second plane, since
# no logit reaches a refine depth of -1e4
FOLD = {
    "Fold dense": dict(depth=None),
    "Fold dense v8": dict(depth=None, v8=True),
    "Fold T=16": dict(depth=16.0),
    "Fold T=14": dict(depth=14.0),
    "Fold capacity": dict(depth=None, refine_k=-1e4, refine_v=-1e4),
    "Fold capacity v8": dict(depth=None, v8=True, refine_k=-1e4, refine_v=-1e4),
}


YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}


def load(yarn=False):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)
    if yarn:
        # transformers 5 moved rope settings into `rope_parameters`
        if isinstance(getattr(cfg, "rope_parameters", None), dict):
            cfg.rope_parameters = {**cfg.rope_parameters, **YARN}
        else:
            cfg.rope_scaling = dict(YARN)
        cfg.max_position_embeddings = 131072
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    return tok, model


def corpus(tok, want):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    out, n = [], 0
    for r in ds:
        t = r["text"]
        if t.strip():
            out.append(t)
            n += len(t) // 4
            if n > want:
                break
    return tok("\n".join(out), return_tensors="pt").input_ids[0]


class Model:
    """The decoder's pieces, called one layer at a time."""

    def __init__(self, hf):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        self.m = hf.model
        self.head = hf.lm_head
        self.rope = apply_rotary_pos_emb
        c = hf.config
        self.L, self.H, self.HKV, self.D = (
            c.num_hidden_layers,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        )

    def qkv(self, i, x, pos):
        """Layer `i`'s normed, rotated q `(B, H, D)`, k and v `(B, H_KV, D)`
        for one token per request at positions `pos`."""
        lay = self.m.layers[i]
        a = lay.self_attn
        h = lay.input_layernorm(x)
        B = x.shape[0]
        q = a.q_norm(a.q_proj(h).view(B, 1, self.H, self.D)).transpose(1, 2)
        k = a.k_norm(a.k_proj(h).view(B, 1, self.HKV, self.D)).transpose(1, 2)
        v = a.v_proj(h).view(B, self.HKV, self.D)
        cos, sin = self.m.rotary_emb(h, pos[:, None])
        q, k = self.rope(q, k, cos, sin)
        return q[:, :, 0], k[:, :, 0], v

    def finish(self, i, x, o):
        lay = self.m.layers[i]
        x = x + lay.self_attn.o_proj(o.reshape(x.shape[0], 1, -1))
        return x + lay.mlp(lay.post_attention_layernorm(x))

    def step(self, attend, tok, pos, logits=False):
        """Next tokens `(B,)` for `tok` `(B,)` at `pos` `(B,)`, or with `logits`
        their logits `(B, vocab)`; `attend(i, q, k, v)` is the layer's
        attention with the token appended."""
        x = self.m.embed_tokens(tok)[:, None]
        for i in range(self.L):
            q, k, v = self.qkv(i, x, pos)
            x = self.finish(i, x, attend(i, q, k, v))
        out = self.head(self.m.norm(x))[:, -1]
        return out if logits else out.argmax(-1)


def prefill(hf, prompts):
    """Every layer's post-RoPE K and V `(T, H_KV, D)` for the prompts,
    packed and staged on the host, and each prompt's next token. FA-4
    computes the prompt attention."""
    from flash_attn.cute import flash_attn_func
    from transformers import AttentionInterface

    ks, vs = (
        [[] for _ in range(hf.config.num_hidden_layers)],
        [[] for _ in range(hf.config.num_hidden_layers)],
    )
    seen = []

    def attn(module, q, k, v, attention_mask=None, scaling=None, **kw):
        i = len(seen) % hf.config.num_hidden_layers
        seen.append(i)
        ks[i].append(k[0].transpose(0, 1).to("cpu"))
        vs[i].append(v[0].transpose(0, 1).to("cpu"))
        o = flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            softmax_scale=scaling,
            causal=True,
        )
        return (o[0] if isinstance(o, tuple) else o), None

    AttentionInterface.register("efa_prefill", attn)
    hf.config._attn_implementation = "efa_prefill"
    nxt = []
    with torch.no_grad():
        for p in prompts:
            logits = hf(p[None].cuda(), use_cache=False, logits_to_keep=1).logits
            nxt.append(int(logits[0, -1].argmax()))
    hf.config._attn_implementation = "sdpa"
    return [torch.cat(x) for x in ks], [torch.cat(x) for x in vs], nxt


class Paged:
    """A bf16 paged cache, pages of `PAGE` keys, each request's pages in
    order, and the fixed slot a replayed step writes."""

    def __init__(self, k, v, lens, HKV, D, hnd=False):
        B = len(lens)
        per = -(-(max(lens) + 1) // PAGE)
        shape = (B * per, HKV, PAGE, D) if hnd else (B * per, PAGE, HKV, D)
        self.k = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        self.hnd = hnd
        off = 0
        for b, n in enumerate(lens):
            for x, dst in ((k, self.k), (v, self.v)):
                t = torch.zeros(per * PAGE, HKV, D, dtype=torch.bfloat16)
                t[:n] = x[off : off + n]
                t = t.view(per, PAGE, HKV, D).cuda()
                dst[b * per : (b + 1) * per] = t.transpose(1, 2) if hnd else t
            off += n
        self.table = torch.arange(B * per, device="cuda", dtype=torch.int32).view(B, per)
        self.at = torch.tensor(lens, device="cuda", dtype=torch.int32)
        self.page = self.table.gather(1, (self.at // PAGE)[:, None].long())[:, 0].long()
        self.slot = (self.at % PAGE).long()
        self.lens = self.at + 1

    def write(self, k, v):
        if self.hnd:
            self.k[self.page, :, self.slot] = k
            self.v[self.page, :, self.slot] = v
        else:
            self.k[self.page, self.slot] = k
            self.v[self.page, self.slot] = v


def arm_fa3(k, v, lens, M, room=1):
    """`room` slots past the longest prompt: one for a replayed step, one per
    generated token for a real decode."""
    import flash_attn_interface as fa3

    B = len(lens)
    S = max(lens) + room
    kc = torch.zeros(M.L, B, S, M.HKV, M.D, device="cuda", dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    for i in range(M.L):
        off = 0
        for b, n in enumerate(lens):
            kc[i, b, :n] = k[i][off : off + n].cuda()
            vc[i, b, :n] = v[i][off : off + n].cuda()
            off += n
    at = torch.tensor(lens, device="cuda", dtype=torch.int32)

    def attend(i, q, kn, vn):
        o = fa3.flash_attn_with_kvcache(
            q[:, None],
            kc[i],
            vc[i],
            k=kn[:, None],
            v=vn[:, None],
            cache_seqlens=at,
            softmax_scale=1.0 / math.sqrt(M.D),
        )
        return (o[0] if isinstance(o, tuple) else o)[:, 0]

    return attend, (kc, vc, at), "flash_attn_interface.flash_attn_with_kvcache, contiguous"


def arm_flashinfer(k, v, lens, M):
    import flashinfer as fi

    caches = [Paged(k[i], v[i], lens, M.HKV, M.D) for i in range(M.L)]
    c0 = caches[0]
    B = len(lens)
    npg = (c0.lens.long() + PAGE - 1) // PAGE
    indptr = torch.cat([torch.zeros(1, device="cuda", dtype=torch.long), npg.cumsum(0)]).int()
    per = c0.table.shape[1]
    indices = torch.cat(
        [b * per + torch.arange(int(npg[b]), device="cuda") for b in range(B)]
    ).int()
    last = (c0.lens.long() - (npg - 1) * PAGE).int()
    ws = torch.zeros(WS, dtype=torch.uint8, device="cuda")
    w = fi.BatchDecodeWithPagedKVCacheWrapper(
        ws,
        "NHD",
        use_cuda_graph=True,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=indptr,
        paged_kv_indices_buffer=indices,
        paged_kv_last_page_len_buffer=last,
    )
    w.plan(
        indptr,
        indices,
        last,
        M.H,
        M.HKV,
        M.D,
        PAGE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=1.0 / math.sqrt(M.D),
    )

    def attend(i, q, kn, vn):
        caches[i].write(kn, vn)
        return w.run(q, (caches[i].k, caches[i].v))

    return (
        attend,
        (caches, ws, w, indptr, indices, last),
        "flashinfer.BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True), page 128",
    )


def arm_cudnn(k, v, lens, M):
    from flashinfer.cudnn import cudnn_batch_decode_with_kv_cache as dec

    caches = [Paged(k[i], v[i], lens, M.HKV, M.D, hnd=True) for i in range(M.L)]
    ws = torch.zeros(WS, dtype=torch.uint8, device="cuda")
    S = max(lens) + 1
    lk = caches[0].lens.view(-1, 1, 1, 1).contiguous()

    def attend(i, q, kn, vn):
        caches[i].write(kn, vn)
        return dec(
            q.contiguous(),
            caches[i].k,
            caches[i].v,
            1.0 / math.sqrt(M.D),
            ws,
            max_sequence_kv=S,
            actual_seq_lens_kv=lk,
            block_tables=caches[i].table,
            is_cuda_graph_compatible=True,
        )

    return attend, (caches, ws, lk), "flashinfer.cudnn.cudnn_batch_decode_with_kv_cache, page 128"


def fold_caches(k, v, lens, M, kw, room=64, v_scales=None):
    """One `FoldKVCache` per layer holding the prompts, with `room` tokens to
    grow; an 8-bit V takes `v_scales[i]` in layer `i` where given."""
    from exact_fold_attn import FoldKVCache

    B = len(lens)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    caches = []
    for i in range(M.L):
        kwi = dict(kw, v_scale=v_scales[i]) if (v_scales and kw.get("v8")) else kw
        c = FoldKVCache(B, M.H, M.HKV, M.D, max(lens) + room, page_size=PAGE, **kwi)
        c.write_prompt(k[i].cuda(), v[i].cuda(), cu, list(lens))
        caches.append(c)
    return caches


def arm_fold(k, v, lens, M, kw, bufs):
    caches = fold_caches(k, v, lens, M, kw, v_scales=V_SCALES)
    at = torch.tensor(lens, device="cuda", dtype=torch.int32)
    qb, kb, vb = bufs
    replays = [c.prepare_replay_step(qb, kb, vb, at=at) for c in caches]

    def attend(i, q, kn, vn):
        qb.copy_(q)
        kb.copy_(kn)
        vb.copy_(vn)
        o = replays[i]()[0]
        return o.view(q.shape).to(q.dtype)

    return attend, (caches, replays, at), f"exact_fold_attn.FoldKVCache({kw}), page 128"


def graph_step(M, attend, tok, pos):
    """The whole step as a CUDA graph."""
    out = M.step(attend, tok, pos)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        M.step(attend, tok, pos)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        nxt = M.step(attend, tok, pos)
    return g, nxt, out


def speed(args, rep, tok, hf, M, flusher):
    ids = corpus(tok, 1_500_000)
    for Bz, S in SPEED:
        label = f"B{Bz} S{S}"
        print(f"\n=== speed {label} ===", flush=True)
        starts = random.Random(0).sample(range(ids.numel() - S - 1), Bz)
        prompts = [ids[s : s + S - 1] for s in starts]
        lens = [S - 1] * Bz
        k, v, nxt = prefill(hf, prompts)
        tokn = torch.tensor(nxt, device="cuda")
        pos = torch.tensor(lens, device="cuda")
        qb = torch.empty(Bz, M.H, M.D, device="cuda", dtype=torch.bfloat16)
        kb = torch.empty(Bz, M.HKV, M.D, device="cuda", dtype=torch.bfloat16)
        vb = torch.empty_like(kb)
        builders = [
            ("FA-3", arm_fa3, ()),
            ("FlashInfer", arm_flashinfer, ()),
            ("cuDNN", arm_cudnn, ()),
        ]
        builders += [(name, arm_fold, (kw, (qb, kb, vb))) for name, kw in FOLD.items()]
        results, first = {}, {}
        for name, build, extra in builders:
            try:
                with torch.no_grad():
                    attend, held, prov = build(k, v, lens, M, *extra)
                    g, _, out = graph_step(M, attend, tokn, pos)
                    first[name] = out.tolist()
                    r = measure(
                        {name: g.replay},
                        graphable={name: False},
                        rounds=21,
                        cold=True,
                        flusher=flusher,
                    )
                    results[name] = dict(r[name], provenance=prov)
                print(
                    f"  {name:12s} {results[name]['us'] / 1e3:8.3f} ms/step  "
                    f"{Bz / results[name]['us'] * 1e6:8.0f} tok/s",
                    flush=True,
                )
                del attend, held, g
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                results[name] = dict(error=repr(e)[:300])
            torch.cuda.empty_cache()
        agree = {
            n: sum(a == b for a, b in zip(t, first.get("FA-3", t), strict=True)) / Bz
            for n, t in first.items()
        }
        rep.add(kind="speed", B=Bz, S=S, label=label, arms=results, next_token_agreement=agree)
        del k, v
        torch.cuda.empty_cache()


NAMES = ["Ada", "Basil", "Cyra", "Dorian", "Elio", "Fenna", "Gideon", "Hollis", "Ines", "Jory"]


# ------------------------------------------------------------ real decoding
#
# The quality modes decode for real: every step appends the token where its
# request ends and attends over everything so far, in every arm.


QUALITY_ARMS = ["FA-3", "FlashInfer", "cuDNN", "FlashInfer FP8", *FOLD]
BF16_PEERS = ("FlashInfer", "cuDNN")


class Live:
    """An arm decoding for real: `begin()` before each model step,
    `attend(i, q, k, v)` in every layer, `end()` after it."""

    def __init__(self, attend, begin=None, end=None, held=(), headroom=None):
        self.attend = attend
        self.begin = begin or (lambda: None)
        self.end = end or (lambda: None)
        self.held = held
        self.headroom = headroom


class Headroom:
    """Each decoded row's log-sum-exp over its reference, `log2` of the
    denominator the kernel returns, gathered on the device: the extremes,
    the rows outside the range certificate's window, and the row count."""

    def __init__(self):
        self.lo = torch.full((), math.inf, device="cuda", dtype=torch.float64)
        self.hi = torch.full((), -math.inf, device="cuda", dtype=torch.float64)
        self.out = torch.zeros((), device="cuda", dtype=torch.int64)
        self.n = 0

    def add(self, cache):
        den = cache.den.double()
        lo, hi = cache._range_window()
        t = torch.log2(den)
        self.lo = torch.minimum(self.lo, t.min())
        self.hi = torch.maximum(self.hi, t.max())
        self.out += ((den < lo) | (den > hi) | ~torch.isfinite(den)).sum()
        self.n += den.numel()

    def summary(self):
        return dict(
            lse_minus_z_min=float(self.lo),
            lse_minus_z_max=float(self.hi),
            outside_window=int(self.out),
            rows=self.n,
        )


def merge_headroom(acc, name, arm):
    """Fold one arm run's headroom into `acc[name]`."""
    if arm.headroom is None or arm.headroom.n == 0:
        return
    h = arm.headroom.summary()
    a = acc.get(name)
    acc[name] = (
        h
        if a is None
        else dict(
            lse_minus_z_min=min(a["lse_minus_z_min"], h["lse_minus_z_min"]),
            lse_minus_z_max=max(a["lse_minus_z_max"], h["lse_minus_z_max"]),
            outside_window=a["outside_window"] + h["outside_window"],
            rows=a["rows"] + h["rows"],
        )
    )


class Growing:
    """Pages of `PAGE` keys holding the prompts with room for `room` tokens
    per request, bf16 or e4m3 with one scale per tensor (`scale`)."""

    def __init__(self, k, v, lens, HKV, D, room, hnd=False, scale=None):
        B = len(lens)
        self.per = -(-(max(lens) + room) // PAGE)
        self.hnd, self.scale = hnd, scale
        dt = torch.float8_e4m3fn if scale else torch.bfloat16
        shape = (B * self.per, HKV, PAGE, D) if hnd else (B * self.per, PAGE, HKV, D)
        self.k = torch.zeros(shape, device="cuda", dtype=dt)
        self.v = torch.zeros_like(self.k)
        off = 0
        for b, n in enumerate(lens):
            for j, (x, dst) in enumerate(((k, self.k), (v, self.v))):
                t = torch.zeros(self.per * PAGE, HKV, D, dtype=torch.bfloat16)
                t[:n] = x[off : off + n]
                t = self._enc(t.cuda(), j).view(self.per, PAGE, HKV, D)
                dst[b * self.per : (b + 1) * self.per] = t.transpose(1, 2) if hnd else t
            off += n
        self.table = torch.arange(B * self.per, device="cuda", dtype=torch.int32).view(B, self.per)

    def _enc(self, x, j):
        if not self.scale:
            return x
        m = B_E4M3_MAX
        return (x.float() / self.scale[j]).clamp(-m, m).to(torch.float8_e4m3fn)

    def write(self, at, k, v):
        page = self.table.gather(1, (at // PAGE)[:, None].long())[:, 0].long()
        slot = (at % PAGE).long()
        k, v = self._enc(k, 0), self._enc(v, 1)
        if self.hnd:
            self.k[page, :, slot] = k
            self.v[page, :, slot] = v
        else:
            self.k[page, slot] = k
            self.v[page, slot] = v


B_E4M3_MAX = 448.0


def live_fa3(k, v, lens, M, room):
    attend, held, _ = arm_fa3(k, v, lens, M, room=room)
    at = held[2]
    return Live(attend, end=lambda: at.add_(1), held=held)


def live_flashinfer(k, v, lens, M, room, fp8=False):
    """FlashInfer's paged tensor-core decode, planned every step for the
    lengths it reaches. The FP8 arm stores K and V in e4m3 with one scale
    per layer and tensor, set from the prompt's largest value."""
    import flashinfer as fi

    scales = [None] * M.L
    if fp8:
        scales = [
            (
                max(float(k[i].abs().max()), 1e-30) / B_E4M3_MAX,
                max(float(v[i].abs().max()), 1e-30) / B_E4M3_MAX,
            )
            for i in range(M.L)
        ]
    caches = [Growing(k[i], v[i], lens, M.HKV, M.D, room, scale=scales[i]) for i in range(M.L)]
    at = torch.tensor(lens, device="cuda", dtype=torch.int32)
    ws = torch.zeros(WS, dtype=torch.uint8, device="cuda")
    w = fi.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
    table = caches[0].table
    pidx = torch.arange(table.shape[1], device="cuda")

    def begin():
        n = at.long() + 1
        npg = (n + PAGE - 1) // PAGE
        indptr = torch.cat([torch.zeros(1, device="cuda", dtype=torch.long), npg.cumsum(0)]).int()
        indices = table[pidx[None] < npg[:, None]].int()
        last = (n - (npg - 1) * PAGE).int()
        w.plan(
            indptr,
            indices,
            last,
            M.H,
            M.HKV,
            M.D,
            PAGE,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn if fp8 else torch.bfloat16,
            sm_scale=1.0 / math.sqrt(M.D),
        )

    def attend(i, q, kn, vn):
        c = caches[i]
        c.write(at, kn, vn)
        if fp8:
            return w.run(q, (c.k, c.v), k_scale=scales[i][0], v_scale=scales[i][1])
        return w.run(q, (c.k, c.v))

    return Live(attend, begin, lambda: at.add_(1), (caches, ws, w))


def live_cudnn(k, v, lens, M, room):
    from flashinfer.cudnn import cudnn_batch_decode_with_kv_cache as dec

    caches = [Growing(k[i], v[i], lens, M.HKV, M.D, room, hnd=True) for i in range(M.L)]
    at = torch.tensor(lens, device="cuda", dtype=torch.int32)
    lk = (at + 1).view(-1, 1, 1, 1).contiguous()
    ws = torch.zeros(WS, dtype=torch.uint8, device="cuda")
    S = caches[0].per * PAGE

    def attend(i, q, kn, vn):
        c = caches[i]
        c.write(at, kn, vn)
        return dec(
            q.contiguous(),
            c.k,
            c.v,
            1.0 / math.sqrt(M.D),
            ws,
            max_sequence_kv=S,
            actual_seq_lens_kv=lk,
            block_tables=c.table,
            is_cuda_graph_compatible=True,
        )

    return Live(
        attend, lambda: lk.copy_((at + 1).view(-1, 1, 1, 1)), lambda: at.add_(1), (caches, ws)
    )


V_SCALES: list[float] = []


def calibrate_v(hf, tok):
    """Each layer's fixed 8-bit V scale, set once from a fixed calibration
    prompt (the first 16K tokens of the corpus) rather than from any batch:
    the power of two leaving a factor two of headroom over its largest V."""
    ids = corpus(tok, 40_000)[:16384]
    _, v, _ = prefill(hf, [ids])
    V_SCALES[:] = [2.0 ** math.ceil(math.log2(float(x.abs().max()) * 2 / B_E4M3_MAX)) for x in v]
    return list(V_SCALES)


def live_fold(k, v, lens, M, room, kw):
    if kw.get("v8") and not V_SCALES:
        raise RuntimeError("calibrate_v first: an 8-bit V takes a fixed scale per layer")
    caches = fold_caches(k, v, lens, M, kw, room=room + 64, v_scales=V_SCALES)
    head = Headroom()

    def attend(i, q, kn, vn):
        o = caches[i].decode(q, kn, vn)
        head.add(caches[i])
        return o

    return Live(attend, held=caches, headroom=head)


def live_arm(name, k, v, lens, M, room):
    if name == "FA-3":
        return live_fa3(k, v, lens, M, room)
    if name == "FlashInfer":
        return live_flashinfer(k, v, lens, M, room)
    if name == "FlashInfer FP8":
        return live_flashinfer(k, v, lens, M, room, fp8=True)
    if name == "cuDNN":
        return live_cudnn(k, v, lens, M, room)
    return live_fold(k, v, lens, M, room, FOLD[name])


def run_live(M, arm, first, lens, steps, inputs=None, on_logits=None):
    """`steps` decode steps from token `first` `(B,)` at positions `lens`,
    fed `inputs[:, t]` at step t (teacher forcing) or else each step's own
    greedy token. Returns every step's greedy token `(B, steps)`."""
    tok = first
    pos = torch.tensor(lens, device="cuda")
    gen = []
    with torch.no_grad():
        for t in range(steps):
            arm.begin()
            lg = M.step(arm.attend, tok, pos, logits=True)
            arm.end()
            nxt = lg.argmax(-1)
            gen.append(nxt)
            if on_logits is not None:
                on_logits(t, lg)
            tok = inputs[:, t + 1] if inputs is not None and t + 1 < inputs.shape[1] else nxt
            pos = pos + 1
    return torch.stack(gen, 1)


def _arms(args):
    arms = [a for a in args.arms if a == "FA-3" or a in QUALITY_ARMS]
    return ["FA-3", *[a for a in arms if a != "FA-3"]]


def _batches(items, size):
    return [items[i : i + size] for i in range(0, len(items), size)]


def _stamp(t0):
    return f"[{time.time() - t0:7.0f}s]"


def nll(args, rep, tok, hf, M):
    """Teacher-forced documents: each arm decodes the same `--tf-tokens`
    tokens after a prompt of each context, and is scored on the next token.

    Documents are disjoint spans of WikiText-103 text (articles in order,
    concatenated). The scored span of a document is the same at every
    context; the prompt is the text just before it. KL is FA-3's
    distribution against the arm's, in nats per token."""
    T = args.tf_tokens
    ctx = max(args.contexts)
    ids = corpus(tok, args.docs * (ctx + T + 1) + 200_000)
    stride = ids.numel() // args.docs
    if stride < ctx + T + 1:
        raise ValueError(f"{ids.numel()} tokens hold no {args.docs} spans of {ctx + T + 1}")
    docs = [ids[d * stride : d * stride + ctx + T + 1] for d in range(args.docs)]
    arms = _arms(args)
    t0 = time.time()
    for S in args.contexts:
        Bz = max(1, args.batch_tokens // S)
        per_doc = {a: [] for a in arms}
        head = {}
        for bi, batch in enumerate(_batches(list(range(args.docs)), Bz)):
            prompts = [docs[d][ctx - S : ctx] for d in batch]
            inp = torch.stack([docs[d][ctx : ctx + T] for d in batch]).cuda()
            tgt = torch.stack([docs[d][ctx + 1 : ctx + T + 1] for d in batch]).cuda()
            k, v, _ = prefill(hf, prompts)
            lens = [S] * len(batch)
            ref_lp, ref_top = [], []
            for name in arms:
                acc = dict(
                    nll=torch.zeros(len(batch), device="cuda", dtype=torch.float64),
                    kl=torch.zeros(len(batch), device="cuda", dtype=torch.float64),
                    agree=torch.zeros(len(batch), device="cuda", dtype=torch.float64),
                )

                def on_logits(t, lg, name=name, acc=acc, tgt=tgt, ref_lp=ref_lp, ref_top=ref_top):
                    lp = lg.float().log_softmax(-1)
                    acc["nll"] -= lp.gather(1, tgt[:, t : t + 1])[:, 0]
                    top = lp.argmax(-1)
                    if name == "FA-3":
                        ref_lp.append(lp.cpu())
                        ref_top.append(top)
                        acc["agree"] += 1.0
                    else:
                        r = ref_lp[t].cuda()
                        acc["kl"] += (r.exp() * (r - lp)).sum(-1)
                        acc["agree"] += (top == ref_top[t]).double()

                try:
                    arm = live_arm(name, k, v, lens, M, room=T + 1)
                    run_live(M, arm, inp[:, 0], lens, T, inputs=inp, on_logits=on_logits)
                    merge_headroom(head, name, arm)
                    del arm
                    for j, d in enumerate(batch):
                        per_doc[name].append(
                            dict(
                                doc=d,
                                nll=float(acc["nll"][j]) / T,
                                kl=float(acc["kl"][j]) / T,
                                agree=float(acc["agree"][j]) / T,
                            )
                        )
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    per_doc[name].append(dict(error=repr(e)[:300], docs=batch))
                torch.cuda.empty_cache()
                done = [x for x in per_doc[name] if "nll" in x]
                if done:
                    print(
                        f"  {_stamp(t0)} S{S} batch {bi} {name:16s} nll "
                        f"{sum(x['nll'] for x in done) / len(done):.4f}  kl "
                        f"{sum(x['kl'] for x in done) / len(done):.2e}  agree "
                        f"{sum(x['agree'] for x in done) / len(done):.4f}",
                        flush=True,
                    )
            ref_lp.clear()
            del k, v
            torch.cuda.empty_cache()
        rep.add(
            kind="nll",
            S=S,
            B=Bz,
            docs=args.docs,
            tokens=T,
            arms=_nll_summary(per_doc),
            per_doc=per_doc,
            headroom=head,
        )


def _nll_summary(per_doc):
    ref = {x["doc"]: x["nll"] for x in per_doc["FA-3"] if "nll" in x}
    out = {}
    for name, rows in per_doc.items():
        ok = [x for x in rows if "nll" in x]
        if not ok:
            out[name] = dict(error=[x.get("error") for x in rows][:1])
            continue
        n = len(ok)
        d = [x["nll"] - ref[x["doc"]] for x in ok if x["doc"] in ref]
        mean_d = sum(d) / len(d) if d else None
        se = (statistics.stdev(d) / math.sqrt(len(d))) if len(d) > 1 else None
        out[name] = dict(
            docs=n,
            nll=sum(x["nll"] for x in ok) / n,
            nll_delta=mean_d,
            nll_delta_se=se,
            kl=sum(x["kl"] for x in ok) / n,
            top1_agree=sum(x["agree"] for x in ok) / n,
        )
    return out


def _chat(tok, text):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
    return torch.tensor(ids, dtype=torch.long)


def diverge(args, rep, tok, hf, M):
    """Free-running greedy decoding: where each arm's tokens first leave
    FA-3's, and how many match position by position, up to FA-3's first end
    of turn. Half the prompts ask for a summary through the chat template,
    half continue the text raw."""
    ids = corpus(tok, 1_500_000)
    rng = random.Random(2)
    prompts, kinds = [], []
    lo, hi = args.div_context
    for i in range(args.div_prompts):
        n = rng.randrange(lo, hi - 64)
        s = rng.randrange(0, ids.numel() - n)
        if i % 2 == 0:
            p = _chat(tok, tok.decode(ids[s : s + n]) + "\n\nSummarize the text above in detail.")
            kinds.append("summarize")
        else:
            p = ids[s : s + n]
            kinds.append("continue")
        prompts.append(p)
    arms = _arms(args)
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    G = args.gen_tokens
    toks = {a: [None] * len(prompts) for a in arms}
    t0 = time.time()
    for bi, batch in enumerate(_batches(list(range(len(prompts))), args.div_batch)):
        ps = [prompts[i] for i in batch]
        lens = [p.numel() for p in ps]
        k, v, nxt = prefill(hf, ps)
        first = torch.tensor(nxt, device="cuda")
        for name in arms:
            try:
                arm = live_arm(name, k, v, lens, M, room=G + 1)
                gen = run_live(M, arm, first, lens, G - 1)
                del arm
                seqs = torch.cat([first[:, None], gen], 1).tolist()
                for j, i in enumerate(batch):
                    toks[name][i] = seqs[j]
            except Exception:  # noqa: BLE001
                traceback.print_exc()
            torch.cuda.empty_cache()
            print(f"  {_stamp(t0)} batch {bi} {name} done", flush=True)
        del k, v
        torch.cuda.empty_cache()
    rows = {}
    for name in arms:
        if name == "FA-3":
            continue
        first_div, match, ended = [], [], []
        for i in range(len(prompts)):
            a, r = toks[name][i], toks["FA-3"][i]
            if a is None or r is None:
                continue
            end = next((j + 1 for j, x in enumerate(r) if x in stop), len(r))
            fd = next((j for j in range(end) if a[j] != r[j]), None)
            first_div.append(end if fd is None else fd)
            ended.append(fd is None)
            match.append(sum(a[j] == r[j] for j in range(end)) / end)
        if not first_div:
            rows[name] = dict(error="no sequences")
            continue
        q = statistics.quantiles(first_div, n=10) if len(first_div) > 1 else first_div
        rows[name] = dict(
            prompts=len(first_div),
            first_divergence_median=statistics.median(first_div),
            first_divergence_deciles=q,
            never_diverged=sum(ended) / len(ended),
            match_fraction=sum(match) / len(match),
            first_divergence=first_div,
            match=match,
        )
        print(
            f"  {name:16s} first divergence median {rows[name]['first_divergence_median']}"
            f"  never {rows[name]['never_diverged']:.2f}  match {rows[name]['match_fraction']:.3f}",
            flush=True,
        )
    rep.add(
        kind="diverge",
        prompts=len(prompts),
        context=list(args.div_context),
        gen_tokens=G,
        kinds=kinds,
        lens=[p.numel() for p in prompts],
        arms=rows,
        tokens=toks,
        bf16_reference=list(BF16_PEERS),
    )


RULER_TASKS = ("multikey", "multivalue", "multiquery", "vartrack")


def _keys(rng, n):
    out = set()
    while len(out) < n:
        out.add(rng.choice(NAMES) + " " + rng.choice(NAMES))
    return list(out)


def ruler_prompt(tok, ids, S, rng, task):
    """A RULER-style prompt of `S` tokens over a WikiText haystack:
    `(tokens, answers, scoring)`, scored on every answer appearing in the
    output (`all`) or on the first seven-digit number (`first`)."""
    num = lambda: str(rng.randrange(1_000_000, 9_999_999))
    if task == "multikey":
        keys = _keys(rng, 4)
        vals = [num() for _ in keys]
        needles = [
            f" One of the special magic numbers for {k} is: {x}. " for k, x in zip(keys, vals)
        ]
        rng.shuffle(needles)
        q = (
            f"\nQuestion: What is the special magic number for {keys[0]} mentioned in the "
            f"provided text?\nAnswer: The special magic number for {keys[0]} mentioned in the "
            "provided text is"
        )
        answers, scoring = [vals[0]], "first"
    elif task == "multivalue":
        key = _keys(rng, 1)[0]
        vals = [num() for _ in range(4)]
        needles = [f" One of the special magic numbers for {key} is: {x}. " for x in vals]
        q = (
            f"\nQuestion: What are all the special magic numbers for {key} mentioned in the "
            f"provided text?\nAnswer: The special magic numbers for {key} mentioned in the "
            "provided text are"
        )
        answers, scoring = vals, "all"
    elif task == "multiquery":
        keys = _keys(rng, 4)
        vals = [num() for _ in keys]
        needles = [
            f" One of the special magic numbers for {k} is: {x}. " for k, x in zip(keys, vals)
        ]
        rng.shuffle(needles)
        ks = ", ".join(keys[:-1]) + f" and {keys[-1]}"
        q = (
            f"\nQuestion: What are the special magic numbers for {ks} mentioned in the provided "
            f"text?\nAnswer: The special magic numbers for {ks} mentioned in the provided text are"
        )
        answers, scoring = vals, "all"
    elif task == "vartrack":
        val = str(rng.randrange(10_000, 99_999))
        names = set()
        while len(names) < 5:
            names.add("".join(rng.choice(string.ascii_uppercase) for _ in range(5)))
        names = list(names)
        needles = [f" VAR {names[0]} = {val} . "]
        needles += [f" VAR {names[i]} = {names[i - 1]} . " for i in range(1, 5)]
        q = (
            f"\nQuestion: Find all variables that are assigned the value {val} in the text "
            "above.\nAnswer: According to the chain(s) of variable assignment in the text "
            f"above, 5 variables are assigned the value {val}, they are:"
        )
        answers, scoring = names, "all"
    else:
        raise ValueError(task)
    nt = [tok(x, return_tensors="pt").input_ids[0] for x in needles]
    qt = tok(q, return_tensors="pt").input_ids[0]
    n_hay = S - sum(x.numel() for x in nt) - qt.numel()
    s = rng.randrange(0, ids.numel() - n_hay)
    hay = ids[s : s + n_hay]
    # needles in the order given, at sorted random depths: a variable chain
    # stays in order
    at = sorted(rng.sample(range(n_hay), len(nt)))
    parts, prev = [], 0
    for a, x in zip(at, nt):
        parts += [hay[prev:a], x]
        prev = a
    parts += [hay[prev:], qt]
    return torch.cat(parts), answers, scoring


def _score(text, answers, scoring):
    if scoring == "first":
        m = re.search(r"\d{7}", text)
        ok = bool(m and m.group(0) == answers[0])
        return ok, float(ok)
    hit = [a in text for a in answers]
    return all(hit), sum(hit) / len(hit)


def ruler(args, rep, tok, hf, M):
    ids = corpus(tok, 1_500_000)
    arms = _arms(args)
    t0 = time.time()
    for task in args.tasks:
        for S in args.contexts:
            rng = random.Random(f"{task} {S}")
            Bz = max(1, args.batch_tokens // S)
            nb = -(-args.samples // Bz)
            samples = [ruler_prompt(tok, ids, S, rng, task) for _ in range(nb * Bz)]
            res = {a: dict(correct=0, recall=0.0, n=0, agree_tokens=0, tokens=0) for a in arms}
            head = {}
            texts = {a: [] for a in arms}
            for bi, batch in enumerate(_batches(samples, Bz)):
                ps = [p for p, _, _ in batch]
                lens = [p.numel() for p in ps]
                k, v, nxt = prefill(hf, ps)
                first = torch.tensor(nxt, device="cuda")
                ref = None
                for name in arms:
                    try:
                        arm = live_arm(name, k, v, lens, M, room=args.answer_tokens + 1)
                        gen = run_live(M, arm, first, lens, args.answer_tokens - 1)
                        merge_headroom(head, name, arm)
                        del arm
                        seqs = torch.cat([first[:, None], gen], 1).tolist()
                        if name == "FA-3":
                            ref = seqs
                        r = res[name]
                        for j, (_, ans, sc) in enumerate(batch):
                            text = tok.decode(seqs[j])
                            ok, rc = _score(text, ans, sc)
                            r["correct"] += ok
                            r["recall"] += rc
                            r["n"] += 1
                            texts[name].append(text)
                            if ref is not None:
                                r["agree_tokens"] += sum(a == b for a, b in zip(seqs[j], ref[j]))
                                r["tokens"] += len(seqs[j])
                    except Exception:  # noqa: BLE001
                        traceback.print_exc()
                    torch.cuda.empty_cache()
                del k, v
                torch.cuda.empty_cache()
                print(
                    f"  {_stamp(t0)} {task} S{S} batch {bi + 1}/{nb}: "
                    + "  ".join(f"{a} {res[a]['correct']}/{res[a]['n']}" for a in arms),
                    flush=True,
                )
            summary = {
                a: dict(
                    accuracy=r["correct"] / max(r["n"], 1),
                    recall=r["recall"] / max(r["n"], 1),
                    n=r["n"],
                    token_agreement=r["agree_tokens"] / max(r["tokens"], 1),
                )
                for a, r in res.items()
            }
            rep.add(
                kind="ruler",
                task=task,
                S=S,
                B=Bz,
                samples=nb * Bz,
                arms=summary,
                answers=[a for _, a, _ in samples],
                texts=texts,
                headroom=head,
            )


def generate(M, arm, first, lens, steps, stops):
    """Greedy decoding of up to `steps` tokens after `first` `(B,)`,
    stopping once every request has emitted a token in `stops`."""
    tok = first
    pos = torch.tensor(lens, device="cuda")
    stop = torch.tensor(sorted(stops), device="cuda")
    done = torch.isin(first, stop)
    gen = [first]
    with torch.no_grad():
        for t in range(steps - 1):
            # a host check every 16 steps keeps the loop from syncing per token
            if t % 16 == 0 and bool(done.all()):
                break
            arm.begin()
            tok = M.step(arm.attend, tok, pos)
            arm.end()
            gen.append(tok)
            done |= torch.isin(tok, stop)
            pos = pos + 1
    return torch.stack(gen, 1).tolist()


def _cut(seq, stops):
    return next((seq[:j] for j, x in enumerate(seq) if x in stops), seq)


def longbench(args, rep, tok, hf, M):
    """LongBench v1, English: each task's samples in the benchmark's prompt,
    cut in the middle to `LB_MAX_LENGTH` tokens as its `pred.py` does, and
    decoded greedily by every arm to the task's generation length."""
    from benchmarks.harness import longbench as LB

    arms = _arms(args)
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    t0 = time.time()
    for task in args.lb_tasks:
        data = LB.load(task)
        data = data[args.lb_offset :]
        if args.lb_samples:
            data = data[: args.lb_samples]
        steps = LB.MAXLEN[task]
        # samsum's generations stop at a newline too, as in pred.py
        stops = eos | (
            {tok.encode("\n", add_special_tokens=False)[-1]} if task == "samsum" else set()
        )
        prompts = []
        for x in data:
            text = LB.PROMPT[task].format(**x)
            ids = tok(text, return_tensors="pt").input_ids[0]
            if ids.numel() > LB_MAX_LENGTH:
                h = LB_MAX_LENGTH // 2
                text = tok.decode(ids[:h], skip_special_tokens=True) + tok.decode(
                    ids[-h:], skip_special_tokens=True
                )
                ids = tok(text, return_tensors="pt").input_ids[0]
            prompts.append(ids if task in LB.NO_CHAT else _chat(tok, text))
        order = sorted(range(len(data)), key=lambda i: -prompts[i].numel())
        batches, i = [], 0
        while i < len(order):
            Bz = max(1, args.batch_tokens // (prompts[order[i]].numel() + steps))
            batches.append(order[i : i + Bz])
            i += Bz
        preds = {a: [None] * len(data) for a in arms}
        head = {}
        for bi, batch in enumerate(batches):
            ps = [prompts[j] for j in batch]
            lens = [p.numel() for p in ps]
            k, v, nxt = prefill(hf, ps)
            first = torch.tensor(nxt, device="cuda")
            for name in arms:
                try:
                    arm = live_arm(name, k, v, lens, M, room=steps + 1)
                    seqs = generate(M, arm, first, lens, steps, stops)
                    merge_headroom(head, name, arm)
                    del arm
                    for j, s in zip(batch, seqs):
                        preds[name][j] = tok.decode(_cut(s, stops), skip_special_tokens=True)
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
                torch.cuda.empty_cache()
            del k, v
            torch.cuda.empty_cache()
            print(f"  {_stamp(t0)} {task} batch {bi + 1}/{len(batches)} B={len(batch)}", flush=True)
        summary = {}
        for name in arms:
            sc = [LB.score(task, p, x) for p, x in zip(preds[name], data) if p is not None]
            summary[name] = dict(score=100 * sum(sc) / max(len(sc), 1), n=len(sc))
        print(f"  {task}: " + "  ".join(f"{a} {summary[a]['score']:.2f}" for a in arms), flush=True)
        rep.add(
            kind="longbench",
            task=task,
            category=LB.CATEGORY[task],
            samples=len(data),
            max_gen=steps,
            max_length=LB_MAX_LENGTH,
            arms=summary,
            predictions=preds,
            headroom=head,
        )


# pred.py's limit for 32K-context models
LB_MAX_LENGTH = 31500


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parts",
        nargs="+",
        default=["speed"],
        choices=["speed", "nll", "diverge", "ruler", "longbench"],
    )
    p.add_argument("--yarn", action="store_true", help="YaRN rope scaling to 128K positions")
    p.add_argument("--answer-tokens", type=int, default=64, help="greedy tokens per RULER answer")
    p.add_argument("--arms", nargs="+", default=QUALITY_ARMS)
    p.add_argument("--contexts", type=int, nargs="+", default=list(QUALITY_CONTEXTS))
    p.add_argument(
        "--batch-tokens", type=int, default=131072, help="prompt tokens per batch in nll and ruler"
    )
    p.add_argument("--docs", type=int, default=32)
    p.add_argument("--tf-tokens", type=int, default=1024)
    p.add_argument("--div-prompts", type=int, default=64)
    p.add_argument("--div-context", type=int, nargs=2, default=[8192, 16384])
    p.add_argument("--div-batch", type=int, default=16)
    p.add_argument("--gen-tokens", type=int, default=1024)
    p.add_argument("--tasks", nargs="+", default=list(RULER_TASKS), choices=list(RULER_TASKS))
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--lb-tasks", nargs="+", default=None, help="LongBench tasks (default all)")
    p.add_argument("--lb-samples", type=int, default=0, help="first N samples per task (0: all)")
    p.add_argument("--lb-offset", type=int, default=0, help="skip the first N samples per task")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from benchmarks.harness.longbench import TASKS

    args.lb_tasks = args.lb_tasks or list(TASKS)
    tok, hf = load(args.yarn)
    M = Model(hf)
    # an 8-bit V takes one fixed scale per layer, set before any arm is built
    fold_arms = FOLD if "speed" in args.parts else {a: FOLD[a] for a in _arms(args) if a in FOLD}
    vsc = calibrate_v(hf, tok) if any(kw.get("v8") for kw in fold_arms.values()) else None
    rep = Report(
        args.out,
        model=MODEL,
        speed=SPEED,
        contexts=QUALITY_CONTEXTS,
        fold=FOLD,
        page=PAGE,
        quality_arms=_arms(args),
        v_scales=vsc,
        yarn=YARN if args.yarn else None,
        args=vars(args),
    )
    flusher = L2Flush()
    if "speed" in args.parts:
        speed(args, rep, tok, hf, M, flusher)
    if "nll" in args.parts:
        nll(args, rep, tok, hf, M)
    if "diverge" in args.parts:
        diverge(args, rep, tok, hf, M)
    if "ruler" in args.parts:
        ruler(args, rep, tok, hf, M)
    if "longbench" in args.parts:
        longbench(args, rep, tok, hf, M)
    rep.write()


if __name__ == "__main__":
    main()
