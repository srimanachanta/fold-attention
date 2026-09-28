"""The backward's postprocess: an integer accumulator to the output dtype.

Adapted from FlashAttention's backward postprocess (BSD-3-Clause), which
converts an fp32 dQ accumulator and applies `softmax_scale`. This one converts
an integer accumulator and folds the grid's exact `2^-s` into that same
multiply, so reconstruction costs nothing extra.
"""

import math
from collections.abc import Callable
from functools import partial

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, const_expr, cute
from cutlass.cute.nvgpu import cpasync
from cutlass.tensor_utils import LayoutEnum
from flash_attn.cute import utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from quack import copy_utils, sm90_utils
from quack.cute_dsl_utils import ParamsBase

from . import grid


class TileConvert:
    """One `(tile_m, tile_hdim)` accumulator tile's conversion, integer to
    fp32 to the output dtype, on the MMA accumulator's own fragment layout.
    The layouts are MLIR values, so it is constructed inside the kernel."""

    def __init__(self, dtype, accum_type, tile_m, tile_hdim, num_threads, atom_layout_m):
        num_wg = num_threads // 128
        assert num_threads % 128 == 0 and num_wg % atom_layout_m == 0
        assert (tile_m * tile_hdim * accum_type.width // 128) % num_threads == 0
        self.dtype, self.accum_type = dtype, accum_type
        self.tile_m, self.tile_hdim = tile_m, tile_hdim
        self.num_threads = num_threads

        atom_layout = (atom_layout_m, num_wg // atom_layout_m)
        self.tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
            dtype,
            dtype,
            cute.nvgpu.OperandMajorMode.K,  # only the accumulator layout is used
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=atom_layout + (1,),
            tiler_mn=(tile_m // atom_layout[0], tile_hdim // atom_layout[1]),
        )
        assert num_threads == self.tiled_mma.size

        elems_g2s = 128 // accum_type.width
        assert (tile_m * tile_hdim // elems_g2s) % num_threads == 0
        self.g2s_tiled_copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
                Float32,
                num_bits_per_copy=128,
            ),
            cute.make_layout(num_threads),
            cute.make_layout(elems_g2s),
        )
        self.s2r_tiled_copy = cute.make_tiled_copy_tv(
            # the accumulator's own width: the mainloop wrote it 128 bits a
            # copy, so an int64 tile is two values a vector, not four
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), accum_type, num_bits_per_copy=128),
            cute.make_layout((128, num_wg)),
            cute.make_layout(128 // accum_type.width),
        )
        self.sAccum_layout = cute.make_layout((tile_m * tile_hdim // num_wg, num_wg))
        self.sOut_layout = sm90_utils.make_smem_layout(
            dtype,
            LayoutEnum.ROW_MAJOR,
            (tile_m, tile_hdim),
            major_mode_size=tile_hdim // (num_wg // atom_layout_m),
        )
        num_copy_elems = 128 // dtype.width
        threads_per_row = math.gcd(128, tile_hdim) // num_copy_elems
        self.gmem_tiled_copy_out = copy_utils.tiled_copy_2d(
            dtype, threads_per_row, num_threads, num_copy_elems
        )
        self.r2s_copy_atom = utils.get_smem_store_atom(90, dtype, transpose=False)

    @cute.jit
    def __call__(
        self,
        gAccum: cute.Tensor,
        gOut: cute.Tensor,
        sAccum: cute.Tensor,
        tidx: cutlass.Int32,
        rows_valid: cutlass.Int32,
        head_dim: cutlass.Int32,
        sync: cutlass.Constexpr,
        scale_fn: cutlass.Constexpr,
        zero_accum: cutlass.Constexpr = False,
    ):
        """Convert the block whose flat accumulator slice is `gAccum` into its
        output tile `gOut`, of which `rows_valid` rows are inside the
        sequence. `sync` is a CTA barrier. `scale_fn` gives the multiplier,
        `softmax_scale` and the grid's `2^-s` together, and runs under the
        accumulator's copy. `zero_accum` clears the accumulator for the next
        call, over lines just read."""
        sAccum_flat = cute.make_tensor(sAccum.iterator, cute.make_layout(cute.size(sAccum)))
        sOut = cute.make_tensor(
            cute.recast_ptr(sAccum.iterator, dtype=self.dtype), self.sOut_layout
        )

        # Staged through shared memory: the `cp.async` batch decouples issue
        # from consumption, which reading straight to registers with the same
        # partitioning does not, and that is 4.7x slower at B1 H8/2 S8192.
        g2s_thr = self.g2s_tiled_copy.get_slice(tidx)
        cute.copy(
            self.g2s_tiled_copy, g2s_thr.partition_S(gAccum), g2s_thr.partition_D(sAccum_flat)
        )
        cute.arch.cp_async_commit_group()
        scale = scale_fn()
        cute.arch.cp_async_wait_group(0)
        sync()
        if const_expr(zero_accum):
            # through an int32 view: an int64 partition's vectorised store runs
            # one 16-byte vector past the tile
            g32 = cute.make_tensor(
                cute.recast_ptr(gAccum.iterator, dtype=cutlass.Int32),
                cute.make_layout(cute.size(gAccum) * self.accum_type.width // 32),
            )
            zcopy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128
                ),
                cute.make_layout(self.num_threads),
                cute.make_layout(4),
            )
            tz = zcopy.get_slice(tidx).partition_D(g32)
            zeros = cute.make_rmem_tensor_like(tz)
            zeros.fill(0)
            cute.copy(zcopy, zeros, tz)

        tAsA = self.s2r_tiled_copy.get_slice(tidx).partition_S(sAccum)
        tile_shape = (self.tile_m, self.tile_hdim)
        # The accumulator's own type, not fp32: `autovec_copy` moves raw bits,
        # so a fragment declared fp32 would make `.to(Float32)` a no-op over
        # integer bit patterns.
        acc = cute.make_rmem_tensor(self.tiled_mma.partition_shape_C(tile_shape), self.accum_type)
        assert cute.size(acc) == cute.size(tAsA)
        cute.autovec_copy(tAsA, cute.make_tensor(acc.iterator, cute.make_layout(tAsA.shape)))

        rOut = cute.make_fragment_like(acc, self.dtype)
        rOut.store((acc.load().to(Float32) * scale).to(self.dtype))

        sync()  # every thread is done reading sAccum; the output tile aliases it
        thr_copy_r2s = cute.make_tiled_copy_C(self.r2s_copy_atom, self.tiled_mma).get_slice(tidx)
        cute.copy(thr_copy_r2s, thr_copy_r2s.retile(rOut), thr_copy_r2s.partition_D(sOut))
        sync()

        gmem_thr = self.gmem_tiled_copy_out.get_slice(tidx)
        tOsO = gmem_thr.partition_S(sOut)
        tOrO = cute.make_fragment_like(tOsO, self.dtype)
        cute.autovec_copy(tOsO, tOrO)
        tOcO = gmem_thr.partition_S(cute.make_identity_tensor(tile_shape))
        tOgO = gmem_thr.partition_D(gOut)
        tOpO = utils.predicate_k(tOcO, limit=head_dim)
        for rest_m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
            if tOcO[0, rest_m, 0][0] < rows_valid:
                cute.copy(
                    self.gmem_tiled_copy_out,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None],
                )


class FoldBackwardPostprocess:
    """One CTA per `(tile_m, head_dim)` block of dQ, or of dK or dV on the
    canonical-record path. `grid_kind` names the bound the exponent is
    derived from; `kv_group` is the GQA group, which dQ uses to find the KV
    head and dK/dV use for the column-mass bound `G n`."""

    def __init__(
        self,
        dtype,
        head_dim: int,
        tile_m: int,
        num_threads: int,
        atom_layout_m: int,
        kv_group: int,
        grid_kind: str = "dq",
        accum_bits: int = 32,
    ):
        # 0 is `backward.ablation`'s fp32 dQ accumulator, which has no grid
        assert grid_kind in ("dq", "dk", "dv") and accum_bits in (0, 32, 64)
        assert accum_bits or grid_kind == "dq"
        self.dtype = dtype
        self.tile_m = tile_m
        self.tile_hdim = head_dim
        self.num_threads = num_threads
        self.atom_layout_m = atom_layout_m
        self.kv_group = int(kv_group)
        self.grid_kind = grid_kind
        self.accum_bits = accum_bits
        self.accum_type = {0: Float32, 32: cutlass.Int32, 64: cutlass.Int64}[accum_bits]
        self.root_d = grid.root_d(head_dim)

    @cute.jit
    def __call__(
        self,
        mAccum: cute.Tensor,
        mOut: cute.Tensor,
        scale: Float32,
        mStats: cute.Tensor,
        mCuSeqlens: cute.Tensor | None = None,
        mCuTotalMBlocks: cute.Tensor | None = None,
        mBlocksToBatch: cute.Tensor | None = None,
        stream: cuda.CUstream = None,
    ):
        mAccum, mOut = [assume_tensor_aligned(t) for t in (mAccum, mOut)]
        varlen = mCuSeqlens is not None
        if const_expr(varlen):
            TileScheduler = SingleTileVarlenScheduler
            num_head, num_batch = mOut.shape[1], mCuSeqlens.shape[0] - 1
        else:
            TileScheduler = SingleTileScheduler
            num_head, num_batch = mOut.shape[2], mOut.shape[0]
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(
                mOut.shape[0] if const_expr(varlen) else mOut.shape[1], self.tile_m
            ),
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=self.tile_hdim,
            headdim_v=0,
            total_q=mOut.shape[0],
            tile_shape_mn=(self.tile_m, 1),
            mCuSeqlensQ=mCuSeqlens,
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
            blocks_to_batch_idx_ptr=mBlocksToBatch,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.kernel(
            mAccum, mOut, mCuSeqlens, scale, mStats, tile_sched_params, TileScheduler
        ).launch(
            grid=TileScheduler.get_grid_shape(tile_sched_params),
            block=[self.num_threads, 1, 1],
            smem=self.tile_m * self.tile_hdim * self.accum_type.width // 8,
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def _grid_scale(self, scale, mStats, batch_idx, head_idx, seqlen):
        """`scale` with the grid's exact `2^-s` folded in."""
        if const_expr(self.accum_bits == 0):
            return scale
        if const_expr(self.grid_kind == "dq"):
            gs = grid.dq_scale(mStats, batch_idx, head_idx // self.kv_group, self.root_d)
        else:
            st = grid.load_stats(mStats, batch_idx, head_idx)
            rows = Float32(seqlen.seqlen_q * self.kv_group)
            bits = grid.DKV_BITS if const_expr(self.accum_bits == 64) else grid.DQ_BITS
            if const_expr(self.grid_kind == "dk"):
                gs = grid.dk_scale(st, rows, self.root_d, bits)
            else:
                gs = grid.dv_scale(st, rows, bits)
        return scale * grid.inv_pow2(gs)

    @cute.kernel
    def kernel(
        self,
        mAccum: cute.Tensor,
        mOut: cute.Tensor,
        mCuSeqlens: cute.Tensor | None,
        scale: Float32,
        mStats: cute.Tensor,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        convert = TileConvert(
            self.dtype,
            self.accum_type,
            self.tile_m,
            self.tile_hdim,
            self.num_threads,
            self.atom_layout_m,
        )
        smem = cutlass.memory.SmemAllocator()
        sAccum = smem.allocate_tensor(self.accum_type, convert.sAccum_layout, byte_alignment=1024)
        tidx, _, _ = cute.arch.thread_idx()
        work_tile = TileScheduler.create(tile_sched_params).initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx
        # launched while the kernel before it drains; everything it reads is
        # that kernel's output
        cute.arch.griddepcontrol_wait()
        if work_tile.is_valid_tile:
            seqlen = SeqlenInfoQK.create(
                batch_idx,
                mOut.shape[1],
                0,
                mCuSeqlensQ=mCuSeqlens,
                mCuSeqlensK=None,
                mSeqUsedQ=None,
                mSeqUsedK=None,
                tile_m=self.tile_m,
            )
            if const_expr(mCuSeqlens is None):
                mOut_cur = mOut[batch_idx, None, head_idx, None]
                mAccum_cur = mAccum[batch_idx, head_idx, None]
                head_dim = mOut.shape[3]
            else:
                mOut_cur = cute.domain_offset((seqlen.offset_q, 0), mOut[None, head_idx, None])
                mAccum_cur = cute.domain_offset(
                    (seqlen.padded_offset_q * self.tile_hdim,), mAccum[head_idx, None]
                )
                head_dim = mOut.shape[2]
                # the offset is a whole number of tiles, so the alignment
                # survives; the compiler cannot see that
                mAccum_cur = cute.make_tensor(
                    cute.make_ptr(
                        dtype=mAccum_cur.element_type,
                        value=mAccum_cur.iterator.toint(),
                        mem_space=mAccum_cur.iterator.memspace,
                        assumed_align=mAccum.iterator.alignment,
                    ),
                    mAccum_cur.layout,
                )
            gAccum = cute.local_tile(mAccum_cur, (self.tile_m * self.tile_hdim,), (m_block,))
            convert(
                gAccum,
                cute.local_tile(mOut_cur, (self.tile_m, self.tile_hdim), (m_block, 0)),
                sAccum,
                tidx,
                seqlen.seqlen_q - m_block * self.tile_m,
                head_dim,
                cute.arch.barrier,
                partial(self._grid_scale, scale, mStats, batch_idx, head_idx, seqlen),
                # dK/dV clear their accumulators here; the preprocess clears dQ's
                zero_accum=self.grid_kind != "dq",
            )
