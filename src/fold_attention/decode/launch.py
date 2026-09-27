"""The decode's host side: validate, choose a build, compile, launch."""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from ..utils import Launch, compile_cached, current_stream, full_carveout
from .cache import TAIL_BLOCK, PrefixImage
from .combine import launch_combine
from .config import (
    BN,
    MASS_STRATA,
    MASS_TRIM,
    DecodeConfig,
    WideConfig,
    mass_tile,
    pack_for,
    smem_bytes,
    wide_slots_per_split,
)
from .heuristics import (
    cap_wanted,
    capped_footprint,
    front_min_blocks,
    packed_min_blocks,
    pick_split,
    resident,
    shared_runs_wide,
    v_regs_default,
    wide_split,
)
from .kernel import launch_decode
from .packed import launch_packed
from .reference import launch_mass
from .rows import CASCADE_WIDTHS, pack_tree
from .wide import launch_wide


@dataclass
class SharedPrefix:
    """One shared level of a cascade: a prefix read once for several
    requests, whose partials the combine adds to the unique level's.

    `rows` (`cascade_rows`) names the row group and row each stacked query
    row belongs to. `ka`, `kb`, `ek` and `v` (and `v2`, V's second e4m3
    plane) are the prefix's own cache in `quantize_k`/`quantize_v`'s format,
    its own per-key scales `ek` among them,
    contiguous `(n_prefixes * H_KV, L, D)` or paged under `page_table` and
    `seq_lens`. `length` is the prefix's key count, a multiple of eight, and
    must be given for a paged prefix, whose lengths live on the device.
    `vmean` is the prefix's dropped-mass direction and is given exactly when
    the unique level has one. `split` defaults to `pick_split`.

    Wide levels (`shared_runs_wide`: past 64 stacked rows, and at 64 with an
    image on narrow rows or a long prefix) run on the wide kernel. `image`
    (`prefix_image`, built once per prefix) holds the prefix's fp16 K, its
    scales and bf16 V as that kernel's tiles, so each step streams them with
    no conversion; without it the kernel rounds K from the planes every step.
    """

    rows: torch.Tensor
    ka: torch.Tensor
    kb: torch.Tensor
    v: torch.Tensor
    ek: torch.Tensor | None = None
    v2: torch.Tensor | None = None
    vmean: torch.Tensor | None = None
    page_table: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    length: int | None = None
    split: int | None = None
    image: PrefixImage | None = None


@dataclass(eq=False)
class DecodeLaunch(Launch):
    """A prepared decode step: calling it (with an optional `out`) returns
    `(out, denominator, counts)`. `z` is the reference it reads or writes,
    `config` its build, and `counts_shared` the shared levels' key counts."""

    z: torch.Tensor | None = None
    config: DecodeConfig | None = None
    counts_shared: torch.Tensor | list | None = None


def _compile(key, fn, *args, like=()):
    """`fn` compiled for `key` and the tensors `like`. A `from_dlpack` view's
    shape and strides are baked into the build, so a kernel compiled for one
    cache would address another of a different size out of bounds."""
    key = (key, tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in like))
    return compile_cached(
        key, lambda: full_carveout(cute.compile(fn, *args, cutlass.cuda.default_stream()))
    )


def _views(tensors, align):
    return [from_dlpack(x, assumed_align=align) for x in tensors]


def _level_views(like):
    """A decode level's tensor arguments: the 17 operands the kernel reads
    as 16-byte vectors, then the int32 tables."""
    return _views(like[:17], 16) + _views(like[17:], 4)


def _no_tail(device, v_dtype):
    """Placeholder tail operands for a build without the tail."""
    return (
        torch.zeros((1, 16), device=device, dtype=v_dtype),
        torch.zeros((1, 16), device=device, dtype=torch.int8),
        torch.zeros((1, 4), device=device, dtype=torch.float32),
    )


def _table(t, device, shape):
    """`t`, or an int32 placeholder a build without it never reads."""
    return torch.zeros(shape, device=device, dtype=torch.int32) if t is None else t


def _min_blocks(
    D, G, v8, v_regs, truncate, n_ctas, device, what, tail=-1, weight_terms=1, one=False
):
    """The register cap to ask ptxas for, or 0. It is asked only when it moves
    the residency and the grid has more CTAs than one fewer per SM holds.
    `one` is a build of a single split."""
    want = cap_wanted(D, G, v8, v_regs, truncate, tail, weight_terms, one)
    res = resident(
        v8,
        v_regs,
        D=D,
        G=G,
        truncate=truncate,
        cap=want,
        tail=tail,
        weight_terms=weight_terms,
        one=one,
    )
    if res < 2:
        # every row's accumulators live in registers, so a wide enough group
        # leaves one CTA per SM and nothing to hide its memory behind
        raise ValueError(
            f"{what} G={G} leaves {res} CTA per SM at D={D} on this V format. "
            "Split the query rows across calls on disjoint row slices; each "
            "slice reads the cache once and they do not interact."
        )
    if want:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        if n_ctas > (res - 1) * sms:
            return res
    return 0


def _key_scales(ek, ka, paged, what):
    """The per-key scales as the kernel reads them: their flat storage and a
    contiguous cache's row stride (0 when paged).

    A tile's scales arrive by one bulk copy of whole 16-byte units, so a
    contiguous cache's rows start 16 bytes apart and hold eight scales past
    the last key rounded up, which `quantize_k`'s padded rows do; a paged
    pool's pages are whole multiples of sixteen rows already."""
    if ek is None or ek.dtype != torch.bfloat16:
        raise ValueError(f"{what}K's scales are bf16, one per key (see quantize_k)")
    if paged:
        if ek.dim() != 1 or ek.numel() != int(ka.shape[0]) or not ek.is_contiguous():
            raise ValueError(f"{what}a paged cache's scales are one per pool row, (rows,)")
        flat, stride = ek, 0
    else:
        nbh, sp = int(ka.shape[0]), int(ka.shape[1])
        if ek.dim() != 2 or tuple(ek.shape) != (nbh, sp) or ek.stride(1) != 1:
            raise ValueError(f"{what}a contiguous cache's scales are (NBH, S), one per key")
        stride = int(ek.stride(0))
        need = (nbh - 1) * stride + -(-sp // 8) * 8
        room = ek.untyped_storage().nbytes() // 2 - ek.storage_offset()
        if stride % 8 or ek.data_ptr() % 16 or room < max(need, nbh * stride):
            raise ValueError(
                f"{what}a contiguous cache's scale rows are read in whole 16-byte "
                "units, so each must start 16 bytes apart with room for its last "
                "key rounded up to eight; quantize_k's scales are laid out so"
            )
        flat = torch.as_strided(ek, (nbh * stride,), (1,))
    return flat, stride


def _check_group(G):
    # `wgmma`'s N is 8i for i in 1..4 and 16i up to 256, and the coarse logit
    # stacks both Q planes in N; residency then caps G at 64
    ng = (G + 7) // 8
    if not (ng <= 4 or (ng % 2 == 0 and ng <= 16)):
        lo = 8 * (ng - 1) + 1
        raise ValueError(
            f"G={G} is not a query group width wgmma can issue: N is "
            f"8.ceil(G/8) = {8 * ng}, and the instruction takes N = 8i for "
            "i in 1..4 and N = 16i above that. Query groups of "
            f"{lo}..{8 * ng} rows have no instruction; use {8 * (ng - 1)} "
            f"or {8 * (ng + 1)} rows, or pad the group."
        )


def _check_head_dim(D):
    if D not in (64, 128):
        raise ValueError(
            f"head_dim {D} is not supported: the value matmul's M is 64 "
            "channels and a row wider than 128 bytes is more than one "
            "descriptor swizzle atom, so D is 64 or 128. Pad the head."
        )


def _check_image(image, NBH_s, L_s, D):
    if image is None:
        return None
    if not isinstance(image, PrefixImage):
        raise TypeError("shared: image is a PrefixImage (see prefix_image)")
    tiles = -(-L_s // BN)
    if (
        image.k.shape != (NBH_s, tiles, BN * D)
        or image.e.shape != (NBH_s, tiles * BN)
        or image.length != L_s
    ):
        raise ValueError(
            f"shared: the image is ({NBH_s}, {tiles}, {BN * D}) tiles and "
            f"({NBH_s}, {tiles * BN}) scales of the level's {L_s} keys; got "
            f"{tuple(image.k.shape)} and {tuple(image.e.shape)} over {image.length}"
        )
    return image


def _prepare_shared(
    piece: SharedPrefix,
    *,
    ka,
    ek,
    D,
    G,
    v8,
    v_scale,
    v_regs,
    dropped_mass,
    truncate,
    n_kv_heads,
    device,
):
    """One cascade level, validated and converted, with its build's choices."""
    if not isinstance(piece, SharedPrefix):
        raise TypeError("shared takes SharedPrefix levels")
    rows = piece.rows
    if rows.dtype != torch.int32 or rows.dim() != 2:
        raise ValueError("shared: rows is a 2-D int32 map (see `cascade_rows`)")
    NBH_s, G_s = int(rows.shape[0]), int(rows.shape[1])
    wide = shared_runs_wide(G_s, D, None if piece.image is None else int(piece.image.length))
    if G_s not in CASCADE_WIDTHS and not (wide and G_s % 8 == 0):
        raise ValueError(
            f"shared: a stacked width of {G_s} is not one the matmul can "
            f"issue; `cascade_rows` rounds up to {CASCADE_WIDTHS} or, past 64, "
            "to a multiple of eight"
        )
    ka_s, v_s = piece.ka, piece.v
    if ka_s.dtype != ka.dtype:
        raise ValueError("shared: both levels read one K format; quantise them alike")
    ek_s = piece.ek
    # V's format is per level. A shared level is out of issue slots at the
    # stacked width rather than waiting on bytes, so a bf16 prefix is faster
    # there; the combine's one V scale then divides the bf16 level, which is
    # exact because `quantize_v`'s scale is a power of two.
    v8_s = v_s.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if not v8_s and v_s.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("shared: an unquantised V cache must be bf16 or fp16")
    if v8_s and not v8:
        raise ValueError(
            "shared: a quantised prefix beside a bf16 suffix would put the "
            "combine's one V scale on the wrong level. Pass the prefix bf16, "
            "or quantise both and pass the shared scale as `v_scale`"
        )
    if not v8_s and v8 and float(v_scale) != 1.0:
        v_s = (v_s.float() * (1.0 / float(v_scale))).to(v_s.dtype).contiguous()
    v2_s = piece.v2
    if v8_s and v2_s is None:
        raise ValueError("shared: an e4m3 prefix V needs v2, its second plane")
    if v2_s is None:
        v2_s = v_s
    paged = piece.page_table is not None
    if paged:
        if piece.seq_lens is None:
            raise ValueError("shared: a paged prefix is described by seq_lens too")
        if piece.length is None:
            raise ValueError(
                "shared: pass length, the prefix's key count. seq_lens is on the "
                "device and reading it here would synchronise"
            )
        if NBH_s % n_kv_heads or int(piece.page_table.shape[0]) != NBH_s // n_kv_heads:
            raise ValueError("shared: page_table's first axis is the prefix, not the request")
        S_s = SP_s = int(ka_s.shape[0])
        L_s = int(piece.length)
    else:
        S_s = int(v_s.shape[1])
        SP_s = int(ka_s.shape[1])
        L_s = S_s if piece.length is None else int(piece.length)
    ek_s, ek_stride_s = _key_scales(ek_s, ka_s, paged, "shared: ")
    if L_s % 8:
        raise ValueError(
            f"the prefix is {L_s} keys, which is not a multiple of eight. Eight "
            "is the cache swizzle's period, and a key's phase comes from its "
            "own logical position, so move the boundary"
        )
    vm_s = piece.vmean
    if dropped_mass != (vm_s is not None):
        raise ValueError(
            "shared: the dropped-mass correction is per level, because each "
            "level's declined mass sits at its own block mean of V. Pass vmean "
            "for both levels or neither"
        )
    if vm_s is None:
        vm_s = torch.zeros((NBH_s, D), device=device, dtype=torch.float32)
    else:
        if vm_s.shape not in ((NBH_s, D), (NBH_s, G_s, D)):
            raise ValueError(
                f"shared: vmean must be (NBH, D) or (NBH, G, D), got {tuple(vm_s.shape)}"
            )
        if not v8_s and v8 and float(v_scale) != 1.0:
            vm_s = vm_s.float() * (1.0 / float(v_scale))
        vm_s = vm_s.float().contiguous()
    if wide and v8_s:
        raise NotImplementedError(
            "shared: a stacked width past 64 reads V as bf16; pass the prefix's V "
            "bf16, or split the batch into ranges of at most 64 rows"
        )
    v_regs_s = bool(v_regs) if v8_s else False
    sp_s = piece.split
    if sp_s is None:
        if wide:
            sp_s = wide_split(NBH_s, L_s, G_s, device)
        else:
            sp_s = pick_split(NBH_s, L_s, v8_s, v_regs=v_regs_s, D=D, G=G_s, truncate=truncate)
    sp_s = int(sp_s)
    minb_s = 0
    if not wide:
        minb_s = _min_blocks(
            D,
            G_s,
            v8_s,
            v_regs_s,
            truncate,
            NBH_s * sp_s,
            device,
            "a stacked width of",
            one=sp_s == 1,
        )
    return dict(
        rows=rows,
        ka=ka_s,
        kb=piece.kb,
        v=v_s,
        ek=ek_s,
        ek_stride=ek_stride_s,
        v2=v2_s,
        vmean=vm_s,
        page_table=piece.page_table,
        seq_lens=piece.seq_lens,
        v8=v8_s,
        v_regs=v_regs_s,
        paged=paged,
        S=S_s,
        SP=SP_s,
        split=sp_s,
        min_blocks=minb_s,
        NBH=NBH_s,
        G=G_s,
        wide=wide,
        image=_check_image(piece.image, NBH_s, L_s, D) if wide else None,
    )


def prepare_fold_decode(
    qa,
    qb,
    eq,
    ka,
    kb,
    ek,
    v,
    z,
    cut,
    *,
    split=None,
    truncate=False,
    refine_k=12.0,
    refine_v=8.0,
    v8=False,
    v2=None,
    v_scale=1.0,
    vmean=None,
    tail=None,
    page_table=None,
    page_size=0,
    seq_lens=None,
    n_kv_heads=1,
    group_cut=False,
    draft_len=1,
    causal=False,
    mask=None,
    shared=None,
    parallel_shared=True,
    reference="given",
    weight_terms=1,
    sound=False,
    out_dtype=torch.float32,
):
    """Bind one decode step of the truncated softmax over a two-plane cache.

    Returns a zero-argument callable that launches the step on the current
    torch stream and returns `(out, denominator, counts)`, the same tensors
    on every call, so it can be captured in a CUDA graph (`capture_decode`).
    `counts` is `(NBH, 2)`: the keys whose plane A was read that were live,
    and the keys whose plane B was gathered. The callable also carries `z`,
    `counts_shared` (the shared levels' counts) and `config`.

    Operands: Q's planes `qa`, `qb` `(NBH, G, D)` int8 and scales `eq`
    `(NBH, G)` (`quantize_q`); K's planes and one bf16 scale per key
    (`quantize_k`; a paged cache's are one per pool row); V bf16, or with `v8` its first e4m3 plane, with `v2` the
    second and `v_scale` their power-of-two scale (`quantize_v`). A weight is
    `exp2(s - z)`, so Q is scaled by `LOG2E / sqrt(D)` before quantising, and
    `z` and `cut` are in those log2 units.

    `reference="given"` takes the caller's `z` and `cut`: every key with
    `s < cut` has weight exactly zero. `reference="mass"` estimates each
    row's log-sum-exp in a prepass from the sink, the most recent keys and
    128 stratum centres of the rest, writes it to `z`, and reads `cut` as each
    row's depth below it, so every gate is a claim about a key's share of its
    row's mass.

    `truncate` gathers only the keys some row of the group keeps; `refine_k`
    and `refine_v` gate K's and V's second planes on the final weight
    (`heuristics.refine_for` under the mass reference). `weight_terms=2`
    carries each bf16 weight's rounding error as a second bf16 weight through
    a second value matmul.
    `vmean`, in V's units, re-enters the truncated mass along a supplied
    direction, `(NBH, D)` per KV head or `(NBH, G, D)` per query row; `tail`
    (`tail_model`, bf16 V only) re-enters it at each tile's own block row of
    V instead. `group_cut` lets every gathered key enter every row at its
    true weight, which makes the truncation the group's rather than the
    row's: it costs no time and halves the truncation's error at D = 64 and
    128. "auto" selects it for a single-level decode and not for drafts or
    cascades, whose partitions would otherwise disagree. `sound` shifts the
    cut by plane A's own residual bound, so the screen drops only keys the
    full two-plane logit would also drop.

    `page_table`, `page_size`, `seq_lens` and `n_kv_heads` read a paged cache
    (`paged_cache`, `write_kv`); a paged batch keeps its lengths on the
    device, so it needs an explicit `split` (`pick_split` with the batch's
    mean and longest lengths). `draft_len` with `causal` or `mask` verifies a
    draft of that many positions per query head (`draft_mask`, `pack_rows`).
    `shared` is a cascade: a `SharedPrefix` or a list of them, read once for
    many requests; past 64 stacked rows a level runs on the wide kernel
    (`decode.wide`), whose logits are fp16. `parallel_shared` runs the levels
    on streams of their own, so one level's last wave overlaps the next.

    A single split with an fp32 output normalises its own rows and launches
    no combine.
    """
    return _prepare(
        qa,
        qb,
        eq,
        ka,
        kb,
        ek,
        v,
        z,
        cut,
        split=split,
        truncate=truncate,
        refine_k=refine_k,
        refine_v=refine_v,
        v8=v8,
        v2=v2,
        v_scale=v_scale,
        vmean=vmean,
        tail=tail,
        page_table=page_table,
        page_size=page_size,
        seq_lens=seq_lens,
        n_kv_heads=n_kv_heads,
        group_cut=group_cut,
        draft_len=draft_len,
        causal=causal,
        mask=mask,
        shared=shared,
        parallel_shared=parallel_shared,
        reference=reference,
        weight_terms=weight_terms,
        sound=sound,
        out_dtype=out_dtype,
    )


def fold_decode(qa, qb, eq, ka, kb, ek, v, z, cut, **kw):
    """`prepare_fold_decode(...)()`: one decode step, returning `(out,
    denominator, counts)`."""
    return prepare_fold_decode(qa, qb, eq, ka, kb, ek, v, z, cut, **kw)()


def _prepare(
    qa,
    qb,
    eq,
    ka,
    kb,
    ek,
    v,
    z,
    cut,
    *,
    split=None,
    truncate=False,
    refine_k=12.0,
    refine_v=8.0,
    v8=False,
    v2=None,
    v_scale=1.0,
    vmean=None,
    tail=None,
    page_table=None,
    page_size=0,
    seq_lens=None,
    n_kv_heads=1,
    group_cut=False,
    draft_len=1,
    causal=False,
    mask=None,
    shared=None,
    parallel_shared=True,
    reference="given",
    weight_terms=1,
    sound=False,
    out_dtype=torch.float32,
    v_regs=None,
    min_blocks=None,
    combine_warps=None,
    ready=None,
    front_qk=0,
    clear_ready=True,
    order=None,
    refine_bands=(),
    chunk_keys=0,
    pack=None,
):
    """`prepare_fold_decode` with the build choices a caller does not make.

    `v_regs` takes the 8-bit V's value operand from registers (default by
    `heuristics.v_regs_default`), `min_blocks` fixes the register cap asked
    of ptxas, and `combine_warps` the query rows per combine block. The rest
    serve `FoldKVCache`: `reference="front"` reads Z, the cut, the length and
    Q from `front.prepare_front`'s words in `ready` ((NBH, 1 + 2 G) int32),
    spinning on them rather than waiting on the front's grid; `front_qk` is
    the front's ownership mode; `clear_ready=False` keeps the words live for
    a decode-only replay; `order` ((NBH,) int32) maps grid slots to row
    groups, longest first; `refine_bands` (`heuristics.refine_bands_for`)
    moves the dense gates with each request's own length; `chunk_keys` fixes
    the keys per split, a multiple of 64, and `split` must cover the longest
    request with it. `pack` picks the kernel: 2 the 128-key tile of
    `decode.packed`, 1 the 64-key one, None `config.pack_for`'s choice.
    """
    NBH, G, D = qa.shape
    device = v.device
    if out_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("out_dtype must be float32, bfloat16 or float16")
    _check_head_dim(D)
    # Past 64 rows a row group fills the matmul's M on its own, and a single
    # level runs on the wide kernel (`decode.wide`), whose logits are fp16.
    wide = G > 64 and shared is None
    if wide:
        unsupported = [
            name
            for name, used in (
                ("an 8-bit V", v8),
                ("a tail", tail is not None),
                ('reference="front"', reference == "front"),
                ("chunk_keys", chunk_keys),
                ("order", order is not None),
                ("sound", sound),
            )
            if used
        ]
        if unsupported:
            raise ValueError(
                f"G={G}: past 64 rows a single level runs on the wide kernel, which takes "
                f"no {', '.join(unsupported)}; split the rows across calls of at most 64"
            )
        if reference == "mass" and G > 128:
            raise ValueError(
                f'G={G}: the mass reference holds at most 128 rows; pass reference="given"'
            )
    else:
        _check_group(G)
    q_len = int(draft_len)
    skip = bool(truncate)
    if tail is not None:
        if not skip:
            raise ValueError(
                "the tail re-enters truncated mass, and truncate=False truncates nothing"
            )
        if vmean is not None:
            raise ValueError("tail replaces vmean: pass one or the other")
        if shared is not None:
            raise ValueError(
                "a cascade keeps each row's own cut on both levels, and the tail "
                "needs group truncation; pass vmean for a cascade"
            )
        if v8:
            raise ValueError(
                "the tail runs on a bf16 V: on an 8-bit V it loses time at matched "
                "error (heuristics.tail_rank_for); pass vmean"
            )
        # a warp's dropped mass rides a key no row kept, so the truncation is
        # the group's
        group_cut = True
    if group_cut == "auto":
        group_cut = skip and G > 1 and q_len == 1 and shared is None
    if group_cut not in (0, 1, False, True):
        raise ValueError('group_cut is True, False or "auto"')
    group_cut = bool(group_cut)
    if q_len < 1:
        raise ValueError("draft_len is the number of draft positions per query head")
    if q_len > 32 and (causal or mask is not None):
        raise ValueError(
            f"draft_len={q_len}: a draft mask is one int32 per node, so at most 32 nodes"
        )
    if G % q_len:
        raise ValueError(
            f"G={G} is not a multiple of draft_len={q_len}: a row group is "
            f"G // draft_len query heads at draft_len draft positions each, laid "
            "out row g * draft_len + t (see pack_rows)"
        )
    if causal and mask is not None:
        raise ValueError(
            "causal=True and mask= are two answers to one question: put the "
            "chain in the bits (draft_mask([-1, 0, 1, ...])) or drop the mask"
        )
    if reference not in ("mass", "given", "front"):
        raise ValueError(f'reference is "given" or "mass", got {reference!r}')
    mass = reference == "mass"
    front = reference == "front"
    front_qk = int(front_qk)
    if front_qk not in (0, 1, 2):
        raise ValueError("front_qk is 0, 1 or 2")
    if front_qk and not front:
        raise ValueError('front_qk needs reference="front"')
    if front and (
        ready is None
        or ready.shape != (NBH, 1 + 2 * G)
        or ready.dtype != torch.int32
        or not ready.is_contiguous()
    ):
        raise ValueError('reference="front" reads the front kernel\'s (NBH, 1 + 2 G) int32 words')
    if front and (shared is not None or page_table is None):
        raise ValueError('reference="front" follows a paged step (front.prepare_front)')
    if front and G > 32:
        raise ValueError('reference="front" reads a row group\'s reference from one warp: G <= 32')
    # the decode reads Z and the cut after the kernel that writes them
    zpre = mass or front
    tree = q_len if (q_len > 1 and (causal or mask is not None)) else 0
    tm = (
        pack_tree(mask, bool(causal), q_len, G, NBH, device)
        if tree
        else torch.zeros((1, 1), device=device, dtype=torch.int32)
    )
    paged = page_table is not None
    if paged:
        if page_size <= 0 or page_size % 16:
            raise ValueError("page_size must be a positive multiple of 16")
        if seq_lens is None:
            raise ValueError("a paged cache is ragged; pass seq_lens")
        if NBH % n_kv_heads:
            raise ValueError(f"NBH={NBH} is not a multiple of n_kv_heads={n_kv_heads}")
        if page_table.shape[0] != NBH // n_kv_heads:
            raise ValueError("page_table's first axis is the request")
        # the flat table's height, not `seq_lens.max()`, which would synchronise
        S = SP = int(ka.shape[0])
    else:
        S = v.shape[1]
        SP = ka.shape[1]
    if ka.dtype != torch.int8 or qa.dtype != torch.int8:
        raise ValueError("K and Q are int8 planes (see quantize_k and quantize_q)")
    ek, ek_stride = _key_scales(ek, ka, paged, "")
    if v8 and v2 is None:
        raise ValueError("v8=True needs v2, V's second e4m3 plane (see quantize_v)")
    if not v8:
        if v.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("an unquantised V cache must be bf16 or fp16")
        v2 = v
    if v_regs is None:
        v_regs = v_regs_default(v8, D, G, skip)
    if v_regs and not v8:
        raise ValueError("v_regs feeds the value matmul from registers, which only an 8-bit V has")
    if weight_terms not in (1, 2):
        raise ValueError("weight_terms is 1 or 2")
    if not v8 and v.dtype == torch.float16:
        raise ValueError(
            "V must be bf16: a weight's value matmul runs in bf16, and an fp16 weight "
            "overflows 16 binades above the reference"
        )
    if mass:
        MS, MT = MASS_STRATA, MASS_TRIM
        MP, MW = mass_tile(tree)
    if group_cut and not skip:
        raise ValueError(
            "group_cut spends the truncation's gather and truncate=False truncates nothing"
        )
    dmc = vmean is not None or tail is not None
    if dmc and not skip:
        raise ValueError(
            "vmean is the truncation's correction and truncate=False leaves nothing to correct"
        )
    if vmean is None:
        vmean = torch.zeros((NBH, D), device=device, dtype=torch.float32)
    elif vmean.shape not in ((NBH, D), (NBH, G, D)):
        raise ValueError(f"vmean must be (NBH, D) or (NBH, G, D), got {tuple(vmean.shape)}")
    trank = 0
    if tail is None:
        vbk, tu, tvr = _no_tail(device, v.dtype)
    else:
        trank = tail.rank
        NGq = (G + 7) // 8
        if trank and (trank % 8 or not (trank <= 32 or trank % 16 == 0)):
            raise ValueError(
                f"tail rank {trank} is the N of a wgmma, which is 8i up to 32 and 16i above"
            )
        if NGq * 8 * trank * 4 > BN * D:
            raise ValueError(f"tail rank {trank} at G={G}: its row sums do not fit plane A's tile")
        if paged and page_size % TAIL_BLOCK:
            raise ValueError(
                f"page_size={page_size}: the tail keeps one row per {TAIL_BLOCK} keys, "
                "so a page must hold whole blocks"
            )
        nblk = int(ka.shape[0]) // TAIL_BLOCK if paged else NBH * -(-int(v.shape[1]) // TAIL_BLOCK)
        vbk = tail.vblk.reshape(-1, D)
        if vbk.dtype != v.dtype or vbk.shape[0] != nblk:
            raise ValueError(
                f"tail.vblk is {nblk} rows of V's format {v.dtype}, one per "
                f"{TAIL_BLOCK} cache rows; got {tuple(tail.vblk.shape)} {tail.vblk.dtype}"
            )
        if trank:
            if tail.u.shape != (NBH, trank, D) or tail.u.dtype != torch.int8:
                raise ValueError(f"tail.u is ({NBH}, {trank}, {D}) int8")
            if tail.vr.shape != (NBH, D, trank) or tail.vr.dtype != torch.float32:
                raise ValueError(f"tail.vr is ({NBH}, {D}, {trank}) float32")
            tu, tvr = tail.u, tail.vr
        else:
            _, tu, tvr = _no_tail(device, v.dtype)
    tail_rank = trank if tail is not None else -1
    if split is None:
        if paged:
            raise ValueError(
                "a paged batch keeps its lengths on the device: pass split "
                "(pick_split with the batch's mean and longest lengths)"
            )
    if split is None and wide:
        split = wide_split(NBH, S, G, device)
    if split is None:
        split = pick_split(
            NBH,
            S,
            v8,
            v_regs=v_regs,
            D=D,
            G=G,
            truncate=skip,
            front=front,
            tail=tail_rank,
            weight_terms=int(weight_terms),
        )
    split = int(split)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if combine_warps is None:
        combine_warps = 4 if NBH * G >= 4 * sms else 1
    if combine_warps not in (1, 2, 4, 8):
        raise ValueError("combine_warps must be 1, 2, 4 or 8")
    while (NBH * G) % combine_warps:
        combine_warps //= 2
    if pack is None:
        pack = pack_for(D, G, bool(v8))
    if pack not in (1, 2) or (pack == 2 and pack_for(D, G, bool(v8)) != 2):
        raise ValueError(f"pack={pack}: 2 is the packed kernel, for D = 64, G <= 4 and a bf16 V")
    smem = smem_bytes(D, G, bool(v8), bool(v_regs), tail_rank, int(weight_terms), bool(sound), pack)
    if wide:
        # the wide kernel runs one CTA per SM and asks ptxas for no cap
        min_blocks = 0
    if min_blocks is None and pack == 2:
        min_blocks = packed_min_blocks(G, tail_rank, int(weight_terms), smem)
    if min_blocks is None and front:
        min_blocks = front_min_blocks(
            D,
            G,
            bool(v8),
            bool(v_regs),
            skip,
            tail_rank,
            int(weight_terms),
            NBH * split,
            sms,
            one=split == 1,
        )
    if min_blocks is None:
        min_blocks = _min_blocks(
            D,
            G,
            v8,
            v_regs,
            skip,
            NBH * split,
            device,
            "",
            tail=tail_rank,
            weight_terms=int(weight_terms),
            one=split == 1,
        )

    # A cascade: each shared prefix is this kernel at the stacked width over
    # its own cache, writing the partial slots above the unique level's. Both
    # levels weigh against the same Z, so the combine's sum is the merge.
    pieces = (
        ()
        if shared is None
        else (tuple(shared) if isinstance(shared, (list, tuple)) else (shared,))
    )
    shl = []
    SPT = split
    for piece in pieces:
        sh = _prepare_shared(
            piece,
            ka=ka,
            ek=ek,
            D=D,
            G=G,
            v8=v8,
            v_scale=v_scale,
            v_regs=v_regs,
            dropped_mass=dmc,
            truncate=skip,
            n_kv_heads=n_kv_heads,
            device=device,
        )
        if sh["paged"] and (page_size <= 0 or page_size % 16):
            raise ValueError(
                "shared: a paged prefix reads the call's page_size, a positive multiple of 16"
            )
        sh["slot0"] = SPT
        SPT += sh["split"] * (wide_slots_per_split(sh["G"]) if sh["wide"] else 1)
        shl.append(sh)
    if chunk_keys and (chunk_keys % 64 or shl):
        raise ValueError(f"chunk_keys={chunk_keys}: a multiple of 64, and a single level only")
    # a fixed chunk always combines, so a request whose batch needs one split
    # takes the same path as in a batch that needs several
    direct = bool(SPT == 1 and out_dtype == torch.float32 and not chunk_keys)
    page_table = _table(page_table, device, (1, 1))
    seq_lens = _table(seq_lens, device, (1,))
    # A piece covering a run of the batch never writes the other requests'
    # slots, and the combine sums every slot, so those must start at zero.
    alloc = torch.zeros if shl else torch.empty
    o = alloc((NBH * SPT, G, D), device=device, dtype=torch.float32)
    den = alloc((NBH * SPT, G), device=device, dtype=torch.float32)
    cnt = torch.zeros((NBH * SPT, 2), device=device, dtype=torch.int32)
    if direct:
        oo, lo, co = o, den, cnt
    else:
        oo = torch.empty((NBH, G, D), device=device, dtype=out_dtype)
        lo = torch.empty((NBH, G), device=device, dtype=torch.float32)
        co = torch.empty((NBH, 2), device=device, dtype=torch.int32)
    # the prepass writes the cut here, so `cut` stays the caller's depth
    zc = torch.empty_like(z) if mass else cut
    if order is None:
        ordt = torch.zeros((1,), device=device, dtype=torch.int32)
    else:
        if shl:
            raise ValueError("order and a shared prefix are exclusive")
        if order.dtype != torch.int32 or tuple(order.shape) != (NBH,):
            raise ValueError(f"order is an int32 ({NBH},) of row-group indices")
        ordt = order
    # The splits' rank sums go to the combine, which expands them once per row:
    # the basis operand then carries each CTA's sums, and the combine reads the
    # basis. A single split or a cascade level expands in the CTA, and so does
    # G = 16, whose sixteen rows a CTA make the combine's share cost more than
    # the CTA saves on large grids (+1.5-2% at 256 row groups).
    rkc = trank in (8, 16, 32) and G <= 8 and not direct and not shl
    tvr_c = tvr
    if rkc:
        tvr = torch.empty((NBH * SPT, G, trank), device=device, dtype=torch.float32)
    smem_pad = 0
    if front:
        # the kernel's own allocation runs past the estimate by its ready word
        own = smem + 16
        smem_pad = capped_footprint(own, NBH * split, sms) - own

    cfg = DecodeConfig(
        n_groups=NBH,
        group=G,
        head_dim=D,
        split=split,
        truncate=skip,
        v8=bool(v8),
        v_regs=bool(v_regs),
        dropped_mass=dmc,
        row_vmean=vmean.dim() == 3,
        paged=paged,
        page_size=int(page_size),
        kv_heads=int(n_kv_heads),
        z_prepass=bool(zpre),
        keep_all=group_cut,
        draft=tree,
        slots=SPT,
        slot0=0,
        unique_group=G,
        shared=False,
        direct=direct,
        min_blocks=int(min_blocks),
        tail_blocks=tail is not None,
        tail_rank=trank,
        weight_terms=int(weight_terms),
        front=front,
        front_qk=front_qk,
        front_clear=bool(clear_ready),
        sound=bool(sound),
        order=order is not None,
        ek_stride=ek_stride,
        refine_bands=tuple(float(x) for x in refine_bands),
        chunk_keys=int(chunk_keys),
        rank_combine=rkc,
        smem_pad=smem_pad,
        pack=pack,
    )
    like = (
        qa,
        qb,
        eq,
        ka,
        kb,
        ek,
        v,
        v2,
        vmean,
        vbk,
        tu,
        tvr,
        z,
        zc,
        o,
        den,
        cnt,
        page_table,
        seq_lens,
        tm,
    )
    args = _level_views(like)
    # the row map only a shared level reads; reusing a view saves a conversion
    args.append(args[-1])
    if ready is None:
        ready = torch.zeros((1, 1), device=device, dtype=torch.int32)
    rdy = from_dlpack(ready.view(-1), assumed_align=4)
    args.append(rdy)
    ordv = from_dlpack(ordt.view(-1), assumed_align=4)
    args.append(ordv)
    S_i32, SP_i32 = cutlass.Int32(S), cutlass.Int32(SP)
    EVS_f32 = cutlass.Float32(v_scale)
    # the refine gates compare a relative logit against these, so a rule that
    # moves with the context length does not recompile
    RK_f32, RV_f32 = cutlass.Float32(-float(refine_k)), cutlass.Float32(-float(refine_v))
    if wide:
        wcfg = WideConfig(
            n_groups=NBH,
            group=G,
            head_dim=D,
            split=split,
            truncate=skip,
            dropped_mass=dmc,
            row_vmean=vmean.dim() == 3,
            paged=paged,
            page_size=int(page_size),
            kv_heads=int(n_kv_heads),
            slots=SPT,
            slot0=0,
            unique_group=G,
            weight_terms=int(weight_terms),
            ek_stride=ek_stride,
            shared=False,
            draft=tree,
            direct=direct,
        )
        # the row map is unread; the draft masks stand in for it
        wlike = (qa, qb, eq, ka, kb, ek, v, vmean, z, zc, o, den, cnt, page_table, seq_lens, tm, tm)
        main = _bind_wide(wcfg, wlike, S, SP, EVS_f32)
    else:
        kern = _compile(
            ("decode", cfg),
            launch_packed if cfg.pack == 2 else launch_decode,
            *args,
            S_i32,
            SP_i32,
            EVS_f32,
            RK_f32,
            RV_f32,
            cfg,
            like=like,
        )
        main = functools.partial(kern, *args, S_i32, SP_i32, EVS_f32, RK_f32, RV_f32)

    # a shared level has no draft and no prepass; each is bound to its own
    # arguments and called with a stream
    kern_s = []
    cnt_s = []
    held_s = []
    for sh in shl:
        if sh["wide"]:
            level, cn, held = _wide_level(sh, cfg, qa, qb, eq, z, zc, o, den, tm, EVS_f32, device)
            kern_s.append(level)
            cnt_s.append(cn)
            held_s += held
            continue
        cfg_s = dataclasses.replace(
            cfg,
            n_groups=sh["NBH"],
            group=sh["G"],
            split=sh["split"],
            paged=sh["paged"],
            min_blocks=sh["min_blocks"],
            z_prepass=False,
            draft=0,
            v8=sh["v8"],
            v_regs=sh["v_regs"],
            row_vmean=sh["vmean"].dim() == 3,
            slot0=sh["slot0"],
            shared=True,
            direct=False,
            order=False,
            pack=1,
            ek_stride=sh["ek_stride"],
        )
        cn = torch.zeros((sh["NBH"] * sh["split"], 2), device=device, dtype=torch.int32)
        pgt_s = _table(sh["page_table"], device, (1, 1))
        sql_s = _table(sh["seq_lens"], device, (1,))
        # a shared level never takes the tail
        sb, su, svr = _no_tail(device, sh["v"].dtype)
        slike = (
            qa,
            qb,
            eq,
            sh["ka"],
            sh["kb"],
            sh["ek"],
            sh["v"],
            sh["v2"],
            sh["vmean"],
            sb,
            su,
            svr,
            z,
            zc,
            o,
            den,
            cn,
            pgt_s,
            sql_s,
            tm,
            sh["rows"],
        )
        sargs = _level_views(slike)
        sargs.append(rdy)
        sargs.append(ordv)
        s_i32, sp_i32 = cutlass.Int32(sh["S"]), cutlass.Int32(sh["SP"])
        kern_d = _compile(
            ("decode", cfg_s),
            launch_decode,
            *sargs,
            s_i32,
            sp_i32,
            EVS_f32,
            RK_f32,
            RV_f32,
            cfg_s,
            like=slike,
        )
        kern_s.append(functools.partial(kern_d, *sargs, s_i32, sp_i32, EVS_f32, RK_f32, RV_f32))
        cnt_s.append(cn)
        held_s += [
            pgt_s,
            sql_s,
            cn,
            sh["rows"],
            sh["ka"],
            sh["kb"],
            sh["ek"],
            sh["v"],
            sh["v2"],
            sh["vmean"],
            sb,
            su,
            svr,
        ]
    parallel_shared = bool(parallel_shared and kern_s)
    aux_streams = (
        [torch.cuda.Stream(device=device, priority=-1) for _ in kern_s] if parallel_shared else []
    )
    aux_cu = [cuda.CUstream(s.cuda_stream) for s in aux_streams]
    fork_event = torch.cuda.Event()
    done_events = [torch.cuda.Event() for _ in kern_s] if parallel_shared else []

    pz = None
    pargs = []
    if mass:
        plike = (qa, qb, eq, ka, ek, page_table, seq_lens, z, cut, zc, tm)
        pargs = _views(plike[:5], 16) + _views(plike[5:7], 4)
        pargs += _views(plike[7:10], 16) + _views(plike[10:], 4)
        pkey = (
            "mass",
            NBH,
            G,
            D,
            MW,
            MS,
            MP,
            MT,
            paged,
            int(page_size),
            int(n_kv_heads),
            tree,
            ek_stride,
        )
        pz = _compile(
            pkey,
            launch_mass,
            *pargs,
            S_i32,
            SP_i32,
            NBH,
            G,
            D,
            MW,
            MS,
            MP,
            MT,
            paged,
            int(page_size),
            int(n_kv_heads),
            tree,
            ek_stride,
            like=plike,
        )
    cargs: list = []
    comb = None
    if not direct:
        clike = (o, den, cnt, oo, lo, co, tvr, tvr_c)
        cargs = _views(clike[:3], 16) + _views(clike[3:4], 4) + _views(clike[4:8], 16)
        cargs += [rdy, ordv, EVS_f32]
        out16 = 0 if out_dtype == torch.float32 else 1 if out_dtype == torch.bfloat16 else 2
        ckey = (
            "combine",
            NBH,
            G,
            D,
            SPT,
            front,
            bool(clear_ready),
            cfg.order,
            combine_warps,
            out16,
            trank if rkc else 0,
        )
        comb = _compile(
            ckey,
            launch_combine,
            *cargs,
            NBH,
            G,
            D,
            SPT,
            front,
            bool(clear_ready),
            cfg.order,
            combine_warps,
            out16,
            trank if rkc else 0,
            like=clike,
        )
    # the dlpack views do not own their tensors
    held = (
        ready,
        qa,
        qb,
        eq,
        ka,
        kb,
        ek,
        v,
        v2,
        vmean,
        vbk,
        tu,
        tvr,
        tvr_c,
        z,
        cut,
        zc,
        o,
        den,
        cnt,
        oo,
        lo,
        co,
        page_table,
        seq_lens,
        tm,
        ordt,
        *held_s,
    )

    def launch(out=None):
        if out is not None:
            if comb is None:
                raise ValueError("an external output needs the combine")
            if out.shape != (NBH, G, D) or out.dtype != out_dtype or out.device != device:
                raise ValueError(f"out must be ({NBH}, {G}, {D}) {out_dtype} on {device}")
        stream = current_stream(device)
        if pz is not None:
            pz(*pargs, S_i32, SP_i32, stream)
        if parallel_shared:
            current = torch.cuda.current_stream(device)
            fork_event.record(current)
            for aux in aux_streams:
                aux.wait_event(fork_event)
            main(stream)
            for level, aux, aux_raw, done in zip(kern_s, aux_streams, aux_cu, done_events):
                level(aux_raw)
                done.record(aux)
            for done in done_events:
                current.wait_event(done)
        else:
            main(stream)
            for level in kern_s:
                level(stream)
        if comb is not None:
            if out is None:
                comb(*cargs, stream)
            else:
                comb(*(cargs[:3] + [from_dlpack(out, assumed_align=4)] + cargs[4:]), stream)
        return oo if out is None else out, lo, co

    counts_shared = None if not cnt_s else cnt_s[0] if len(cnt_s) == 1 else cnt_s
    return DecodeLaunch(launch, held, z, cfg, counts_shared)


def _bind_wide(wcfg, wlike, S, SP, EVS_f32):
    """A wide build bound to its arguments, called with a stream. `wlike` is
    the kernel's tensors in order: 13 read as 16-byte vectors, then the page
    table, lengths, draft masks and row map."""
    wargs = _views(wlike[:13], 16) + _views(wlike[13:], 4)
    s_i32, sp_i32 = cutlass.Int32(S), cutlass.Int32(SP)
    kern = _compile(("wide", wcfg), launch_wide, *wargs, s_i32, sp_i32, EVS_f32, wcfg, like=wlike)
    return functools.partial(kern, *wargs, s_i32, sp_i32, EVS_f32)


def _wide_level(sh, cfg, qa, qb, eq, z, zc, o, den, tm, EVS_f32, device):
    """A shared level on the wide kernel, as `(call(stream), counts, held)`.
    The wide kernel takes no refine gates: it rounds every key's logit to
    fp16 alike."""
    img = sh["image"]
    wcfg = WideConfig(
        n_groups=sh["NBH"],
        group=sh["G"],
        head_dim=cfg.head_dim,
        split=sh["split"],
        truncate=cfg.truncate,
        dropped_mass=cfg.dropped_mass,
        row_vmean=sh["vmean"].dim() == 3,
        paged=sh["paged"] and img is None,
        page_size=cfg.page_size,
        kv_heads=cfg.kv_heads,
        slots=cfg.slots,
        slot0=sh["slot0"],
        unique_group=cfg.group,
        weight_terms=cfg.weight_terms,
        ek_stride=sh["ek_stride"],
        image=img is not None,
    )
    cn = torch.zeros((sh["NBH"] * sh["split"], 2), device=device, dtype=torch.int32)
    pgt_s = _table(sh["page_table"], device, (1, 1))
    sql_s = _table(sh["seq_lens"], device, (1,))
    ka_w, ek_w, v_w, S_w, SP_w = sh["ka"], sh["ek"], sh["v"], sh["S"], sh["SP"]
    if img is not None:
        # the image stands in for K's planes, the scales and V; SP is its
        # tiles per row group
        ka_w, ek_w, v_w = img.k.view(-1), img.e.view(-1), img.v.view(-1)
        S_w, SP_w = img.length, int(img.k.shape[1])
    wlike = (
        qa,
        qb,
        eq,
        ka_w,
        sh["kb"],
        ek_w,
        v_w,
        sh["vmean"],
        z,
        zc,
        o,
        den,
        cn,
        pgt_s,
        sql_s,
        tm,
        sh["rows"],
    )
    held = [
        pgt_s,
        sql_s,
        cn,
        sh["rows"],
        sh["ka"],
        sh["kb"],
        sh["ek"],
        sh["v"],
        sh["vmean"],
        ka_w,
        ek_w,
        v_w,
    ]
    return _bind_wide(wcfg, wlike, S_w, SP_w, EVS_f32), cn, held


def capture_decode(launch: DecodeLaunch, warmup=3) -> DecodeLaunch:
    """Capture a prepared decode in a CUDA graph and return its replay.

    The replay reads and writes the tensors `launch` was bound to, so their
    contents may change between replays; replacing a tensor may not. Capture
    the whole model step instead when the decode is part of a larger graph.
    """
    for _ in range(int(warmup)):
        launch()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        launch()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = launch()
    torch.cuda.synchronize()

    def replay():
        graph.replay()
        return result

    return DecodeLaunch(
        replay, (launch, side, graph), launch.z, launch.config, launch.counts_shared
    )
