"""One attention layer's serving state: FlashAttention-4's prefill, then
FoldAttention's decode over a quantised paged cache.

The prompt is ordinary causal attention, so it runs on FlashAttention-4's
SM90 kernel and its output is FA's to the bit. The same call writes the
prompt's K and V into the two-plane paged cache the decode reads, and every
step after that appends one token per request and attends over the cache.

Every key carries its own scale, so a key appended later is quantised as
precisely as the prompt's and nothing is reserved for it. An 8-bit V takes
one power of two for the layer, fixed at prefill (`v_headroom`).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
from flash_attn.cute import flash_attn_varlen_func

from .decode import write
from .decode.cache import KBR, TAIL_BLOCK, TailState, swizzle_of
from .decode.config import BN, smem_bytes
from .decode.front import prepare_front
from .decode.heuristics import (
    capped_footprint,
    front_early,
    front_ownership,
    pick_split,
    refine_bands_for,
    refine_for,
    tail_rank_for,
    v_regs_default,
    weight_terms_for,
)
from .decode.launch import _prepare
from .utils import Launch


def _batched(fn, *xs):
    """`fn` over a batch of matrices through its batched path: a batch of one
    takes a different algorithm, so it is padded with an identity."""
    if xs[0].shape[0] > 1:
        return fn(*xs)
    pad = [
        torch.cat([x, torch.eye(x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)[None]])
        for x in xs
    ]
    out = fn(*pad)
    if isinstance(out, tuple):
        return tuple(o[:1] for o in out)
    return out[:1]


@dataclass(eq=False)
class ReplayStep(Launch):
    """`FoldKVCache.prepare_replay_step`'s result; `ready` is its private copy
    of the front's words."""

    ready: torch.Tensor | None = None


class FoldKVCache:
    """A paged, quantised KV cache for `batch` requests of one layer, and the
    decode attention over it.

    `prefill` takes the prompts, packed, and returns their causal attention;
    `decode` appends one token per request and returns its attention over
    everything so far (`append` and `attend` do the two halves).

    `depth` is the truncation depth: a key whose weight is under
    `2^-depth` of its row's mass is dropped, and None keeps every key. The
    reference is each row's log-sum-exp, estimated every step from the cache
    and the step's query alone: the sink and the most recent keys are scored
    exactly, and 128 stratum centres of the rest stand for the keys nothing
    scored. Nothing passes from one step to the next.

    `v8` stores V as two e4m3 planes rather than bf16; its one scale is the
    layer's, `v_scale` if given (a power of two) and otherwise set by the
    prefill from the batch's largest value, so a request's bits then depend
    on the batch it was prefilled with. `tail` is where a truncated step's
    dropped mass re-enters: the rank of the K-predicted term
    (`decode.cache.tail_model`), 0 for the 64-key block rows alone, or None
    for one V mean per head; "auto" follows `heuristics.tail_rank_for`. The
    rank's map is fitted on each prompt, and the append keeps every block's
    row current.

    `refine_k`, `refine_v` (default `heuristics.refine_for`) and
    `weight_terms` (default `heuristics.weight_terms_for`) trade accuracy for
    time; `refine_k=refine_v=-inf` never reads a second plane.

    `check_range` certifies each step's reference after the fact. A row's
    denominator is `t = 2^(LSE - Z)` and bounds its largest weight from both
    sides (`t / n <= 2^(m - Z) <= t`), so `t` inside `_range_window` and a
    finite output mean no weight or sum overflowed, and the gates sit no
    more than a binade shallower than asked. A row outside it is decoded
    again with `Z + log2 t`, its log-sum-exp, at the same cut. The check
    reads `t` on the host, one synchronisation per step.

    `split` fixes the CTAs per row group. Left None it follows `pick_split`,
    which reads the batch, so the combine's sum of partials is grouped by the
    batch and a request's output depends on its batch at the level of fp32
    rounding. A fixed split makes it depend on the request alone, and so does
    `chunk`, a fixed number of keys per split (a multiple of 64): the batch's
    longest request sets how many splits run, and a request's splits past its
    own end add zeros, which change no bits.
    """

    def __init__(
        self,
        batch,
        n_heads,
        n_kv_heads,
        head_dim,
        max_len,
        *,
        depth=None,
        v8=False,
        page_size=256,
        split=None,
        tail="auto",
        refine_k=None,
        refine_v=None,
        weight_terms=None,
        v_headroom=2.0,
        v_scale=None,
        check_range=False,
        chunk=None,
        device="cuda",
    ):
        if n_heads % n_kv_heads:
            raise ValueError(f"{n_heads} query heads do not group over {n_kv_heads} KV heads")
        self.batch, self.n_heads, self.n_kv_heads, self.head_dim = (
            batch,
            n_heads,
            n_kv_heads,
            head_dim,
        )
        self.group = n_heads // n_kv_heads
        self.n_groups = batch * n_kv_heads
        self.page_size = page_size
        self.max_len = max_len
        self.depth = depth
        self.v8 = bool(v8)
        self.v_headroom = float(v_headroom)
        self.split = split
        self.device = torch.device(device)

        dev, D, HKV, PS = self.device, head_dim, n_kv_heads, page_size
        pages = -(-max_len // PS)
        rows = batch * pages * HKV * PS
        self.page_table = torch.arange(batch * pages, device=dev, dtype=torch.int32).view(
            batch, pages
        )
        self.ka = torch.zeros((rows, D), device=dev, dtype=torch.int8)
        self.kb = torch.zeros_like(self.ka)
        if self.v8:
            self.va = torch.zeros((rows, D), device=dev, dtype=torch.float8_e4m3fn)
            self.vb = torch.zeros_like(self.va)
        else:
            self.va = torch.zeros((rows, D), device=dev, dtype=torch.bfloat16)
            self.vb = None
        # each pool row's key scale
        self.ek = torch.zeros((rows,), device=dev, dtype=torch.bfloat16)
        if v_scale is not None and 2.0 ** round(math.log2(v_scale)) != v_scale:
            raise ValueError(f"v_scale={v_scale} is not a power of two")
        self._fixed_v_scale = v_scale
        self._v_scale = float(v_scale) if (self.v8 and v_scale is not None) else 1.0
        self.seq_lens = torch.zeros((batch,), device=dev, dtype=torch.int32)
        self.lens = [0] * batch
        self.order = torch.arange(self.n_groups, device=dev, dtype=torch.int32)
        self.vsum = torch.zeros((self.n_groups, D), device=dev, dtype=torch.float32)
        self.vmean = torch.zeros_like(self.vsum)
        self.qa = torch.zeros((self.n_groups, self.group, D), device=dev, dtype=torch.int8)
        self.qb = torch.zeros_like(self.qa)
        self.eq = torch.ones((self.n_groups, self.group), device=dev, dtype=torch.float32)
        self.z = torch.zeros((self.n_groups, self.group), device=dev, dtype=torch.float32)
        # each row's depth below its reference
        self.cut = torch.full_like(self.z, 1e4 if depth is None else float(depth))
        # the front's `z - cut`
        self.zc = torch.empty_like(self.z)
        # the front's words per row group, which the decode spins on
        self.ready = torch.zeros((self.n_groups, 1 + 2 * self.group), device=dev, dtype=torch.int32)
        if tail == "auto":
            tail = tail_rank_for(D, self.group, self.v8) if depth is not None else -1
        self.tail_rank = -1 if tail is None else int(tail)
        self._refine_k_fixed = refine_k
        self._refine_v_fixed = refine_v
        self.check_range = bool(check_range)
        if chunk is not None and (split is not None or chunk % BN):
            raise ValueError(f"chunk={chunk} is a multiple of {BN} and replaces split")
        self.chunk = chunk
        self.den = None
        self.weight_terms = weight_terms_for(depth) if weight_terms is None else int(weight_terms)
        self.tail: TailState | None = None
        if self.tail_rank >= 0:
            if depth is None:
                raise ValueError(
                    "the tail re-enters truncated mass, and depth=None truncates nothing"
                )
            if self.v8:
                raise ValueError("the decode tail needs a bf16 V cache; pass tail=None with v8")
            if PS % TAIL_BLOCK:
                raise ValueError(f"page_size={PS}: the tail keeps one row per {TAIL_BLOCK} keys")
            nblk = rows // TAIL_BLOCK
            R = self.tail_rank
            self.tail = TailState(
                rows=torch.zeros((nblk, D), device=dev, dtype=self.va.dtype),
                vsum=torch.zeros((nblk, D), device=dev, dtype=torch.float32),
                ysum=torch.zeros((nblk, R), device=dev, dtype=torch.float32) if R else None,
                u=torch.zeros((self.n_groups, R, D), device=dev, dtype=torch.int8) if R else None,
                vr=(
                    torch.zeros((self.n_groups, D, R), device=dev, dtype=torch.float32)
                    if R
                    else None
                ),
            )
        # forces a front ownership mode (`heuristics.front_ownership`) when set
        self._front_qk = None
        self._active_front_qk = 0
        self._launches = {}
        self._steps = {}
        self._has_front_state = False

    @property
    def v_scale(self) -> float:
        """V's scale. The prefill leaves it on the device, and the first read
        after it is where the host takes it."""
        if torch.is_tensor(self._v_scale):
            self._v_scale = float(self._v_scale)
        return self._v_scale

    def prefill(self, q, k, v, cu_seqlens, max_seqlen=None):
        """Causal attention over `batch` prompts packed along the first axis.

        `q` is `(total, H, D)`, `k` and `v` `(total, H_KV, D)`, `cu_seqlens`
        `(batch + 1,)` int32 on the device. Returns `(total, H, D)` in q's
        dtype. The prompts replace the cache's contents.
        """
        B, D = self.batch, self.head_dim
        if cu_seqlens.shape != (B + 1,):
            raise ValueError(f"cu_seqlens is ({B + 1},) for a batch of {B}")
        lens = torch.diff(cu_seqlens).tolist()
        if min(lens) < 1 or max(lens) > self.max_len:
            raise ValueError(
                f"prompt lengths {min(lens)}..{max(lens)} are not in 1..{self.max_len}"
            )
        if max_seqlen is None:
            max_seqlen = max(lens)
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            softmax_scale=1.0 / math.sqrt(D),
        )
        if isinstance(out, tuple):
            out = out[0]
        self.write_prompt(k, v, cu_seqlens, lens)
        return out

    def write_prompt(self, k, v, cu_seqlens, lens=None):
        """Quantise packed prompts' K and V into the cache, which they replace,
        without computing their attention. `lens` are the host lengths
        `cu_seqlens` describes."""
        if lens is None:
            lens = torch.diff(cu_seqlens).tolist()
        self._v_scale = write.write_prompt(
            k,
            v,
            lens,
            self.page_table,
            self.page_size,
            self.ka,
            self.kb,
            self.va,
            self.vb,
            self.ek,
            self.vsum,
            self.vmean,
            v_headroom=self.v_headroom,
            v_scale=self._fixed_v_scale if self.v8 else None,
        )
        if self.tail is not None:
            self._fit_tail(k, v, lens)
        self.seq_lens.copy_(torch.tensor(lens, dtype=torch.int32), non_blocking=True)
        self.lens = list(lens)
        self._set_order()
        self._launches.clear()
        self._steps.clear()
        self._has_front_state = False
        self._active_front_qk = 0

    def _fit_tail(self, k, v, lens):
        """Each prompt's rank map and every block row it fills, in the same
        units and pool slots the append then keeps current.

        `cache.tail_basis`'s fit with the batch's linear algebra in one call.
        Each request's Gram matrices and 64-key block sums are fp32 GEMMs and
        reductions over its own rows, shapes that are its alone, of plane A
        times each key's scale; the centred Gram is
        `K^T K - sum_b s_b s_b^T / n_b` from the block sums `s_b`. The
        solves and eigendecompositions are one batched call each, and a
        batched call's per-matrix result does not depend on the other
        matrices (`_batched` pads a batch of one, which takes another path).
        """
        HKV, D, PS, R, t = self.n_kv_heads, self.head_dim, self.page_size, self.tail_rank, self.tail
        assert t is not None
        B, dev, TB = len(lens), k.device, TAIL_BLOCK
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + n)
        n_t = torch.tensor(lens, device=dev)
        # plane A as the prompt writer stored it, read back and un-swizzled
        # rather than rotated and quantised again
        N = cu[-1]
        req = torch.repeat_interleave(torch.arange(B, device=dev), n_t)
        pos = torch.arange(N, device=dev) - torch.tensor(cu[:-1], device=dev).repeat_interleave(n_t)
        pg = self.page_table[req, pos // PS].long()
        heads = torch.arange(HKV, device=dev)[None, :]
        prow = (pg[:, None] * HKV + heads) * PS + (pos % PS)[:, None]
        nu, sh, _ = swizzle_of(D)
        unit = torch.arange(nu, device=dev)[None, :] ^ ((pos >> sh) & (nu - 1))[:, None]
        kint = torch.gather(
            self.ka[prow].view(N, HKV, nu, D // nu),
            2,
            unit[:, None, :, None].expand(N, HKV, nu, D // nu),
        ).view(N, HKV, D)
        ekr = self.ek[prow].float()
        # plane A times the key's scale is exact in f32: 8 bits by 8 bits
        kqa = kint.float() * ekr[..., None]
        NB = -(-max(lens) // TB)
        cnt = (n_t[:, None] - torch.arange(NB, device=dev)[None, :] * TB).clamp(0, TB)
        f64 = torch.float64
        Kb = torch.zeros(B, HKV, NB, D, device=dev, dtype=f64)
        Vb = torch.zeros_like(Kb)
        Araw = torch.zeros(B, HKV, D, D, device=dev, dtype=f64)
        Ckv = torch.zeros_like(Araw)
        for b, n in enumerate(lens):
            kb = kqa[cu[b] : cu[b + 1]]
            vb = v[cu[b] : cu[b + 1]].float()
            nf, rem = n // TB, n % TB
            for src, dst in ((kb, Kb), (vb, Vb)):
                if nf:
                    dst[b, :, :nf] = src[: nf * TB].view(nf, TB, HKV, D).sum(1).transpose(0, 1)
                if rem:
                    dst[b, :, nf] = src[nf * TB :].sum(0)
            if R:
                kt = kb.permute(1, 2, 0)
                Araw[b] = kt @ kb.transpose(0, 1)
                Ckv[b] = kt @ vb.transpose(0, 1)
        vsum = Vb.float()
        rows = vsum / cnt[:, None, :, None].clamp_min(1)
        if R:
            # every full block's term is exact at 1/64; a prompt's last,
            # partial block adds its own, rounded once
            nf = n_t // TB
            wf = (torch.arange(NB, device=dev)[None, :] < nf[:, None]).to(f64) / TB
            kw = Kb * wf[:, None, :, None]
            Sk = kw.transpose(-1, -2) @ Kb
            Sv = kw.transpose(-1, -2) @ Vb
            rem = n_t % TB
            bi = torch.arange(B, device=dev)
            li = nf.clamp(max=NB - 1)
            kl, vl = Kb[bi, :, li], Vb[bi, :, li]
            inv = torch.where(rem > 0, 1.0 / rem.clamp_min(1).to(f64), 0.0)[:, None, None, None]
            Sk = Sk + kl[..., :, None] * kl[..., None, :] * inv
            Sv = Sv + kl[..., :, None] * vl[..., None, :] * inv
            A0 = (Araw - Sk).reshape(B * HKV, D, D)
            C = (Ckv - Sv).reshape(B * HKV, D, D)
            reg = 1e-3 * A0.diagonal(dim1=1, dim2=2).sum(-1) / D
            A = A0 + reg[:, None, None] * torch.eye(D, device=dev, dtype=f64)
            M = _batched(torch.linalg.solve, A, C).float()
            Md = M.to(f64)
            _, evec = _batched(torch.linalg.eigh, Md.transpose(1, 2) @ A0 @ Md)
            basis = evec[:, :, -R:].flip(-1).float()
            uu = M @ basis
            eu = uu.abs().amax(1).clamp_min(1e-30) / 127.0
            q8 = torch.round(uu / eu[:, None, :]).clamp(-127, 127)
            # the kernel's projection carries `ek / 256`, so the map back
            # carries the 256
            vr = (basis.transpose(1, 2) * (eu * KBR)[:, :, None]).transpose(1, 2).contiguous()
            assert t.u is not None and t.vr is not None
            t.u[: B * HKV] = q8.transpose(1, 2).to(torch.int8)
            t.vr[: B * HKV] = vr
            # each key's projection as the append forms it: the exact integer
            # product, rounded once by the key's `ek / 256`
            q8h = q8.view(B, HKV, D, R)
            ysum = torch.zeros(B, HKV, NB, R, device=dev, dtype=f64)
            for b, n in enumerate(lens):
                ki = kint[cu[b] : cu[b + 1]].float().transpose(0, 1)
                yk = (ki @ q8h[b]) * (ekr[cu[b] : cu[b + 1]].t() * (1.0 / KBR))[..., None]
                nf, rem = n // TB, n % TB
                if nf:
                    ysum[b, :, :nf] = yk[:, : nf * TB].double().view(HKV, nf, TB, R).sum(2)
                if rem:
                    ysum[b, :, nf] = yk[:, nf * TB :].double().sum(1)
            ysum = ysum.float()
            rows = rows - (ysum / cnt[:, None, :, None].clamp_min(1)) @ vr.view(
                B, HKV, D, R
            ).transpose(-1, -2)
        bi, ji = (cnt > 0).nonzero(as_tuple=True)
        pos0 = ji * TB
        pg = self.page_table[bi, pos0 // PS].long()
        hh = torch.arange(HKV, device=dev)
        blk = (((pg[:, None] * HKV + hh[None, :]) * PS + (pos0 % PS)[:, None]) // TB).reshape(-1)
        t.vsum[blk] = vsum[bi, :, ji].reshape(-1, D)
        if R:
            assert t.ysum is not None
            t.ysum[blk] = ysum[bi, :, ji].reshape(-1, R)
        t.rows[blk] = rows[bi, :, ji].reshape(-1, D).to(k.dtype).view(t.rows.dtype)

    def append(self, k, v):
        """Write one token per request, `(batch, H_KV, D)` each, at its length."""
        if max(self.lens) >= self.max_len:
            raise ValueError(f"a request is at max_len={self.max_len}")
        self._step(None, k, v)
        self.lens = [x + 1 for x in self.lens]

    def attend(self, q, split=None):
        """Attention of `q` `(batch, H, D)` over the cache as it stands."""
        self._step(q, None, None)
        return self._run(q, split)

    def decode(self, q, k, v, split=None):
        """One decode step: append `k`, `v` `(batch, H_KV, D)` and attend with
        `q` `(batch, H, D)`. The front writes Q, the token and the reference
        in one launch, then the decode and its combine run."""
        if max(self.lens) >= self.max_len:
            raise ValueError(f"a request is at max_len={self.max_len}")
        self._step(q, k, v)
        self.lens = [x + 1 for x in self.lens]
        return self._run(q, split)

    def tune_split(self, candidates=(4, 6, 8, 10, 12, 14, 16, 20, 24, 32), rounds=7, reps=10):
        """Time the decode at each split in `candidates` and `pick_split`'s on
        the cache as it stands, and keep the fastest as `split`.

        `pick_split` is a model of the machine, a few percent off the best
        split on some ragged grids. Each candidate is replayed from a CUDA
        graph, so the host's enqueue is not what is timed, in rotating order
        across `rounds` so a drifting clock cancels. The replays write only
        buffers of their own, so the layer's state is untouched. Returns
        `{split: us}`."""
        tiles = -(-(sum(self.lens) // self.batch) // BN)
        auto = self._auto_split()
        cands = sorted({auto} | {int(c) for c in candidates if -(-tiles // int(c)) >= 4})
        graphs = {}
        # a graph keeps device addresses, not the tensors that own them
        runs = {}
        for sp in cands:
            run = self.prepare_replay_decode(sp)
            runs[sp] = run
            for _ in range(2):
                run()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run()
            torch.cuda.current_stream().wait_stream(side)
            with torch.cuda.graph(g):
                run()
            graphs[sp] = g
        torch.cuda.synchronize()
        times = {sp: [] for sp in cands}
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for r in range(rounds):
            order = cands[r % len(cands) :] + cands[: r % len(cands)]
            for sp in order:
                graphs[sp].replay()
                ev[0].record()
                for _ in range(reps):
                    graphs[sp].replay()
                ev[1].record()
                ev[1].synchronize()
                times[sp].append(ev[0].elapsed_time(ev[1]) * 1e3 / reps)
        med = {sp: sorted(t)[len(t) // 2] for sp, t in times.items()}
        self.split = min(med, key=lambda sp: med[sp])
        return med

    def prepare_replay_step(self, q, k=None, v=None, split=None, at=None):
        """A repeatable full step on private buffers, for timing under CUDA
        graphs.

        The returned closure writes Q and, with `k` and `v`, the token at
        `at` (the current lengths by default) without advancing the lengths,
        then runs the front and its dependent decode through the same ready
        words a real step uses. Its reference, ready words and running sums
        are its own; only the token's cache slot and the Q planes are
        rewritten.
        """
        if (k is None) != (v is None):
            raise ValueError("k and v are both given or both left out")
        q = q.reshape(-1, self.head_dim)
        if at is None:
            at = self.seq_lens
        at = at.clone()
        if k is not None and int(at.max()) >= self.max_len:
            raise ValueError(f"a replay position is at max_len={self.max_len}")
        lens_out = torch.empty_like(at)
        vsum = self.vsum.clone() if k is not None else self.vsum
        vmean = self.vmean.clone() if k is not None else self.vmean
        z = self.z.clone()
        zc = self.zc.clone()
        ready = torch.zeros_like(self.ready)
        tail = self.tail
        if k is not None and tail is not None:
            tail = dataclasses.replace(
                tail,
                rows=tail.rows.clone(),
                vsum=tail.vsum.clone(),
                ysum=None if tail.ysum is None else tail.ysum.clone(),
            )
        front_qk = self._front_ownership() if k is not None else 0
        write_step = self._bind_front(
            at,
            vsum,
            vmean,
            z,
            zc,
            ready,
            q.dtype,
            k is not None,
            front_qk,
            tail=tail,
            lens_out=lens_out,
        )
        split = self._resolve_split(split)
        decode = self._prepare_decode(
            split,
            ready,
            True,
            front_qk,
            z=z,
            zc=zc,
            vmean=vmean,
            tail_state=tail,
            refine=self._refine(int(at.max()) + (k is not None)),
        )

        def replay():
            write_step(q, k, v)
            return decode()

        held = (q, k, v, at, lens_out, vsum, vmean, z, zc, ready, tail, write_step, decode)
        return ReplayStep(replay, held, ready)

    def prepare_replay_decode(self, split=None, out_dtype=torch.float32):
        """A repeatable decode-only launch over the last step's front state,
        for timing. `out_dtype` is the output it writes, so a benchmark times
        the launch it scores."""
        split = self._resolve_split(split)
        ready = self._front_seed()
        return self._prepare_decode(
            split,
            ready,
            False,
            self._active_front_qk,
            self.z.clone(),
            self.zc.clone(),
            out_dtype=out_dtype,
        )

    def _set_order(self):
        """Give the longest requests the lowest grid slots.

        A row group's CTAs are contiguous in the grid, so a long request placed
        late has nothing behind it to backfill its tail. Longest-first is 4-8%
        faster on a ragged batch and changes no bit, since the grid slot only
        chooses which row group a CTA takes. It is set once per prompt: a step
        advances every request by one, which keeps a descending order. A
        request keeps its KV heads in one run so their shared page reads stay
        adjacent.
        """
        rank = sorted(range(self.batch), key=lambda b: (-self.lens[b], b))
        order = [r * self.n_kv_heads + h for r in rank for h in range(self.n_kv_heads)]
        self.order.copy_(torch.tensor(order, dtype=torch.int32), non_blocking=True)

    def _front_ownership(self):
        if self._front_qk is not None:
            mode = int(self._front_qk)
            if mode not in (0, 1, 2):
                raise ValueError("_front_qk is 0, 1, 2 or None")
            return mode
        return front_ownership(
            sum(self.lens) * self.n_kv_heads,
            self.n_groups,
            self.group,
            self.head_dim,
            self.v8,
            self.tail is not None,
        )

    def _front_early(self):
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        return front_early(self.n_groups, sms, self.depth is not None)

    def _auto_split(self):
        mean = sum(self.lens) / self.batch
        return pick_split(
            self.n_groups,
            max(1, int(mean)),
            self.v8,
            max_len=max(self.lens),
            D=self.head_dim,
            G=self.group,
            truncate=self.depth is not None,
            front=True,
            tail=self.tail_rank,
            weight_terms=self.weight_terms,
        )

    def refine_gates(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Each request's refine gates at its current length, `(refine_k,
        refine_v)` as `(batch,)` float32 depths below Z: what the next step's
        kernel compares a key's coarse logit against, a request's own length
        band under the default dense gates."""
        rk, rv = self._refine()
        n = torch.tensor(self.lens, dtype=torch.float32)
        if not self._banded():
            return torch.full_like(n, float(rk)), torch.full_like(n, float(rv))
        e0, e1, rk1, rk2, rv1, rv2 = refine_bands_for(
            self.v8, head_dim=self.head_dim, group=self.group
        )
        rk_t = torch.where(n > e1, rk2, torch.where(n > e0, rk1, float(rk)))
        rv_t = torch.where(n > e1, rv2, torch.where(n > e0, rv1, float(rv)))
        return rk_t, rv_t

    def coarse_logits(self) -> torch.Tensor:
        """Every cached key's plane-A logit against the last step's query
        rows, `(batch, n_kv_heads, group, max(lens))` float32 in base 2, as the
        kernel forms it: `(256 c1 + c2) eq ek / 256` from plane A, the key
        scales and both Q planes, exact in float64 before the one rounding.
        Keys past a request's length are -inf. With `z` it gives each key's
        verdict: `s - z` against `-depth` and against `-refine_gates()`."""
        B, HKV, D, G, PS = self.batch, self.n_kv_heads, self.head_dim, self.group, self.page_size
        n = torch.tensor(self.lens, device=self.ka.device)
        S = int(n.max())
        dev = self.ka.device
        pos = torch.arange(S, device=dev)
        pg = self.page_table[:, pos // PS].long()
        prow = (pg[:, :, None] * HKV + torch.arange(HKV, device=dev)) * PS + (pos % PS)[
            None, :, None
        ]
        nu, sh, _ = swizzle_of(D)
        unit = torch.arange(nu, device=dev)[None, :] ^ ((pos >> sh) & (nu - 1))[:, None]
        ka = self.ka[prow].view(B, S, HKV, nu, D // nu)
        ka = torch.gather(ka, 3, unit[None, :, None, :, None].expand_as(ka)).view(B, S, HKV, D)
        kd = ka.permute(0, 2, 1, 3).double()
        qa = self.qa.view(B, HKV, G, D).double()
        qb = self.qb.view(B, HKV, G, D).double()
        c = torch.einsum("bhsd,bhgd->bhgs", kd, qa) * KBR + torch.einsum("bhsd,bhgd->bhgs", kd, qb)
        # the kernel's clamp of a slot's scale, which a real key never reaches
        ek = self.ek[prow].float().clamp(0, 2.0**100) * (1.0 / KBR)
        eq = self.eq.view(B, HKV, G)
        s = c.float() * (eq[..., None] * ek.permute(0, 2, 1)[:, :, None, :])
        return s.masked_fill(~(pos[None] < n[:, None])[:, None, None], float("-inf"))

    def _resolve_split(self, split):
        if self.chunk is not None:
            # every request is cut at the same key offsets whatever its batch;
            # the batch's longest request only sets how many splits run
            return max(1, -(-max(self.lens) // self.chunk))
        if split is None:
            split = self.split
        if split is None:
            split = self._auto_split()
        return split

    def _banded(self):
        """Whether the kernel picks the dense gates from each request's length."""
        return self.depth is None and self._refine_k_fixed is None and self._refine_v_fixed is None

    def _refine(self, seq_len=None):
        """`(refine_k, refine_v)` at `seq_len`, the longest request by default.
        Banded dense gates are band 0's; the kernel moves each request to its
        own band."""
        if self._banded():
            seq_len = 1
        elif seq_len is None:
            seq_len = max(self.lens)
        rk, rv = refine_for(
            self.depth,
            self.v8,
            seq_len=seq_len,
            head_dim=self.head_dim,
            group=self.group,
            tail=self.tail_rank >= 0,
        )
        if self._refine_k_fixed is not None:
            rk = self._refine_k_fixed
        if self._refine_v_fixed is not None:
            rv = self._refine_v_fixed
        return rk, rv

    def _range_window(self):
        """`(lo, hi)` for a row's denominator `t = 2^(LSE - Z)`. Under `hi` no
        weight or sum can overflow (an 8-bit V's weights carry `E4W` = 2^64
        into bf16, which lowers it); `lo` keeps Z within a binade over the
        log-sum-exp, so a gate or cut set at depth T below Z is at least
        T - 1 below the row's mass, and flushed weights are far under 2^-24
        of `t`."""
        return 0.5, 2.0 ** (60 if self.v8 else 100)

    def _out_of_range(self, o, den):
        lo, hi = self._range_window()
        ok = (den >= lo) & (den <= hi)
        ok &= torch.isfinite(o.view(self.n_groups, self.group, self.head_dim)).all(-1)
        return ~ok

    def _rerun(self, bad, den, split, out_dtype, z=None):
        """This step again from the front's Q and cache, with every flagged
        row's reference moved to its log-sum-exp; the other rows keep theirs.
        An infinite or empty denominator moves the reference by 64 binades
        and tries again."""
        z = (self.z if z is None else z).clone()
        for _ in range(4):
            shift = torch.where(
                torch.isfinite(den) & (den > 0),
                torch.log2(den.clamp_min(torch.finfo(torch.float32).tiny)),
                torch.where(den == 0, -64.0, 64.0),
            )
            z = torch.where(bad, z + shift, z)
            key = ("given", split, out_dtype)
            if key not in self._launches:
                self._z_given = torch.empty_like(self.z)
                self._launches[key] = self._prepare_decode(
                    split, None, True, 0, z=self._z_given, out_dtype=out_dtype, reference="given"
                )
            self._z_given.copy_(z)
            o, den2, _ = self._launches[key].run()
            bad2 = self._out_of_range(o, den2)
            if not bool(bad2.any()):
                return o, den2
            bad, den = bad2, den2
        raise FloatingPointError("a row's reference left the exponent range after four moves")

    def _front_seed(self):
        """Ready words as a finished front leaves them, for a decode-only replay."""
        if not self._has_front_state:
            raise RuntimeError("a replay needs one attend or decode step first")
        lens = self.seq_lens.repeat_interleave(self.n_kv_heads)
        front_qk = self._active_front_qk
        done = 2 if (front_qk == 2 or (self.tail_rank >= 0 and front_qk == 1)) else 1
        return torch.cat(
            (
                (lens * 4 + done)[:, None],
                self.z.view(torch.int32) ^ -1,
                self.zc.view(torch.int32) ^ -1,
            ),
            dim=1,
        )

    def _bind_front(
        self, lens, vsum, vmean, z, zc, ready, dtype, has_kv, front_qk, tail, lens_out=None
    ):
        D = self.head_dim
        return prepare_front(
            self.qa.view(-1, D),
            self.qb.view(-1, D),
            self.eq.view(-1),
            self.ek,
            self.page_table,
            self.page_size,
            lens,
            self.ka,
            self.kb,
            self.va,
            self.vb,
            self.v_scale,
            vsum,
            vmean,
            z,
            self.cut,
            zc,
            ready,
            heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            dtype=dtype,
            has_kv=has_kv,
            tail=tail,
            lens_out=lens_out,
            qk_owner=front_qk,
            early=self._front_early(),
            footprint=self._decode_footprint(),
        )

    def _decode_footprint(self):
        """A decode CTA's shared memory, as `launch._prepare` sizes it for the
        default split."""
        truncate = self.depth is not None
        v_regs = v_regs_default(self.v8, self.head_dim, self.group, truncate)
        by = smem_bytes(
            self.head_dim, self.group, self.v8, v_regs, self.tail_rank, self.weight_terms
        )
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        # the decode pads itself the same way (`launch._prepare`); a split
        # passed per call can differ, which only costs the padding's effect
        return capped_footprint(by + 16, self.n_groups * self._resolve_split(None), sms)

    def _prepare_decode(
        self,
        split,
        ready,
        clear_ready,
        front_qk,
        z=None,
        zc=None,
        vmean=None,
        tail_state=None,
        out_dtype=torch.float32,
        refine=None,
        reference="front",
    ):
        truncate = self.depth is not None
        refine_k, refine_v = self._refine() if refine is None else refine
        z = self.z if z is None else z
        zc = self.zc if zc is None else zc
        vmean = self.vmean if vmean is None else vmean
        tail_state = self.tail if tail_state is None else tail_state
        tail = None if tail_state is None else tail_state.decode_tail()
        return _prepare(
            self.qa,
            self.qb,
            self.eq,
            self.ka,
            self.kb,
            self.ek,
            self.va,
            z,
            zc,
            split=split,
            truncate=truncate,
            refine_k=refine_k,
            refine_v=refine_v,
            v8=self.v8,
            v2=self.vb,
            v_scale=self.v_scale,
            vmean=vmean if (truncate and tail is None) else None,
            tail=tail,
            page_table=self.page_table,
            page_size=self.page_size,
            seq_lens=self.seq_lens,
            n_kv_heads=self.n_kv_heads,
            reference=reference,
            group_cut="auto" if truncate else 0,
            weight_terms=self.weight_terms,
            ready=ready,
            front_qk=front_qk,
            clear_ready=clear_ready,
            order=self.order,
            out_dtype=out_dtype,
            refine_bands=(
                refine_bands_for(self.v8, head_dim=self.head_dim, group=self.group)
                if self._banded()
                else ()
            ),
            chunk_keys=self.chunk or 0,
        )

    def _launch(self, split=None, out_dtype=torch.float32):
        """The prepared decode for this batch's lengths, built once per split."""
        split = self._resolve_split(split)
        refine = self._refine()
        key = (split, self._active_front_qk, out_dtype, refine)
        if key not in self._launches:
            self._launches[key] = self._prepare_decode(
                split,
                self.ready,
                True,
                self._active_front_qk,
                out_dtype=out_dtype,
                refine=refine,
            )
        return self._launches[key]

    def _step(self, q, k, v):
        q = None if q is None else q.reshape(-1, self.head_dim)
        x = q if q is not None else k
        front_qk = self._front_ownership() if k is not None else 0
        key = (q is not None, k is not None, x.dtype, front_qk)
        if key not in self._steps:
            if q is not None:
                # Q, the token and the reference in one launch, the decode behind it
                self._steps[key] = self._bind_front(
                    self.seq_lens,
                    self.vsum,
                    self.vmean,
                    self.z,
                    self.zc,
                    self.ready,
                    x.dtype,
                    k is not None,
                    front_qk,
                    tail=self.tail,
                )
            else:
                # an append alone has no query and so no reference
                D = self.head_dim
                self._steps[key] = write.prepare_step(
                    self.qa.view(-1, D),
                    self.qb.view(-1, D),
                    self.eq.view(-1),
                    self.ek,
                    self.page_table,
                    self.page_size,
                    self.seq_lens,
                    self.ka,
                    self.kb,
                    self.va,
                    self.vb,
                    self.v_scale,
                    self.vsum,
                    self.vmean,
                    self.seq_lens,
                    heads=self.n_heads,
                    n_kv_heads=self.n_kv_heads,
                    dtype=x.dtype,
                    has_q=False,
                    has_kv=True,
                    tail=self.tail,
                )
        # `.run` rather than the `Launch` itself: one call frame less per token
        self._steps[key].run(q, k, v)
        if q is not None:
            self._has_front_state = True
            self._active_front_qk = front_qk

    def _run(self, q, split):
        split = self._resolve_split(split)
        shape = (self.batch, self.n_heads, self.head_dim)
        if split > 1 and q.dtype in (torch.bfloat16, torch.float16):
            out = torch.empty(
                (self.n_groups, self.group, self.head_dim), device=q.device, dtype=q.dtype
            )
            o, den, self.counts = self._launch(split, out_dtype=q.dtype).run(out=out)
            if self.check_range:
                o, den = self._certify(o, den, split, q.dtype)
            self.den = den
            return o.view(shape)
        o, den, self.counts = self._launch(split).run()
        if self.check_range:
            o, den = self._certify(o, den, split, torch.float32)
        self.den = den
        return o.view(shape).to(q.dtype)

    def _certify(self, o, den, split, out_dtype):
        """`(o, den)` with every row outside `_range_window` decoded again."""
        bad = self._out_of_range(o, den)
        if not bool(bad.any()):
            return o, den
        o2, den2 = self._rerun(bad, den, split, out_dtype)
        shape = (self.n_groups, self.group, self.head_dim)
        o = torch.where(bad[..., None], o2.view(shape), o.view(shape))
        return o, torch.where(bad, den2, den)
