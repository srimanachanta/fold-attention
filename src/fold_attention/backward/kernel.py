"""SM90 attention backward with order-free cross-CTA reductions.

Adapted from FlashAttention's SM90 CuTe backward (BSD-3-Clause). Two MMA
warpgroups run in ping-pong, and each query block's dQ is issued in the next
block's turn, which the integer dQ reduction allows because it does not care
when a partial lands. dK and dV leave the CTA in one of three ways, chosen by
the plan (`plan.py`):

* private: a work tile walks a whole GQA group over the whole query range, so
  dK/dV are a register sum in a fixed order and the CTA stores them;
* combine: subgroups of two or more heads each publish an fp32 partial, and
  whichever CTA arrives last at the key block sums them in subgroup order;
* fold: heads split one to a tile, or the query range cut into canonical
  records, give many short partials, each rounded onto an integer grid and
  reduced asynchronously; the postprocess converts the sums.
"""

import math
from collections.abc import Callable
from functools import partial

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Boolean, Float32, Int32, Int64, const_expr, cute
from cutlass.cute import FastDivmodDivisorV2
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
from flash_attn.cute import pipeline, utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.named_barrier import NamedBarrierBwd
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from quack import copy_utils, layout_utils, sm90_utils
from quack.cute_dsl_utils import ParamsBase
from quack.sm90_utils import gemm_w_idx, gemm_zero_init

from ..ptx import (
    atomic_add_acq_rel_gpu,
    cvt_rni_s32_f32,
    cvt_rni_s64_f32,
    fence_acq_rel_gpu,
    store_release_gpu,
)
from . import grid
from .scheduler import ListScheduler

# `cp.reduce.async.bulk` has `.add.u32`/`.add.u64` and no signed form; two's
# complement makes them the same add.
_ACCUM = {0: Float32, 32: cutlass.Int32, 64: cutlass.Int64}
_REDUCE = {0: Float32, 32: cutlass.Uint32, 64: cutlass.Uint64}


class FoldBackwardSm90:
    arch = 90
    # S and dP are computed transposed, so each MMA warpgroup owns 64 key rows
    # of them, and dK and dV take P and dS straight from registers.
    num_wg_mma = 2
    num_threads = 384
    stages = 2
    # Each block's dQ is issued in the next block's second turn, so a dS slot
    # is read one block after it is written; the third slot keeps the next
    # store off one a dQ may still be reading, with no barrier of its own.
    dS_stages = 3
    # 24 is the least `setmaxnreg` allows, and 2 x 240 + 24 the whole file at
    # 384 threads.
    num_mma_regs = 240
    num_producer_regs = 24

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        tile_m: int,
        tile_n: int,
        atom_layout_m_dq: int,
        gqa_subgroup: int,
        record_width: int,
        dk_accum_bits: int,
        dv_accum_bits: int,
        dq_fp32: bool = False,
    ):
        self.dtype = dtype
        self.tile_hdim = int(math.ceil(head_dim / 16) * 16)
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.atom_layout_m_dq = atom_layout_m_dq
        self.num_mma_threads = self.num_wg_mma * 128
        self.root_d = grid.root_d(head_dim)
        # With the statistics shuffled from the 8 threads that share a row,
        # each thread keeps 2 rows' values instead of `tile_m / 4`, which pays
        # only where the smaller head dim leaves the shuffles room to issue.
        self.shuffle_stats = self.tile_hdim <= 64

        # One work tile per (key block, `gqa_subgroup` query heads); the heads
        # run in turn in one CTA.
        self.head_loop = max(1, int(gqa_subgroup))
        assert qhead_per_kvhead % self.head_loop == 0
        self.n_sub = qhead_per_kvhead // self.head_loop
        self.record_width = int(record_width)
        self.dkv_fold = dk_accum_bits != 0
        self.dkv_combine = self.n_sub > 1 and not self.dkv_fold
        assert self.dkv_fold == (dv_accum_bits != 0)
        assert self.dkv_fold or self.record_width == 0
        self.dk_accum_bits, self.dv_accum_bits = dk_accum_bits, dv_accum_bits
        self.dk_accum_type, self.dv_accum_type = _ACCUM[dk_accum_bits], _ACCUM[dv_accum_bits]
        self.dk_reduce_type, self.dv_reduce_type = _REDUCE[dk_accum_bits], _REDUCE[dv_accum_bits]
        self.dk_grid_bits = grid.DKV_BITS if dk_accum_bits == 64 else grid.DQ_BITS
        self.dv_grid_bits = grid.DKV_BITS if dv_accum_bits == 64 else grid.DQ_BITS
        # `backward.ablation`'s dQ: fp32 partials summed in arrival order,
        # with no grid
        self.dq_fp32 = dq_fp32
        self.dq_reduce_type = Float32 if self.dq_fp32 else cutlass.Uint32

    def _setup_attributes(self):
        # Q and dO are read as Q and Q^T (and dO and dO^T), so their swizzle
        # serves both; the M dimension does not change the layout.
        self.sQ_layout, self.sdO_layout = [
            sm90_utils.make_smem_layout(
                self.dtype,
                LayoutEnum.ROW_MAJOR,
                (self.tile_m, self.tile_hdim),
                self.stages,
                major_mode_size=self.tile_hdim,
            )
            for _ in range(2)
        ]
        wg_d_dQ = self.num_wg_mma // self.atom_layout_m_dq
        self.sK_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdim),
            major_mode_size=self.tile_hdim // wg_d_dQ,
        )
        self.sV_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdim)
        )
        self.sPdS_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n),
            stage=self.dS_stages,
            major_mode_size=self.tile_n // self.num_wg_mma,
        )
        # A persistent CTA stages dV and dK on `sdS`: side by side where both
        # fit (head_dim 64 and 96), so the two stores leave together; at 128
        # dK waits for dV's store to read out of the one buffer.
        self.dkv_stage_apart = False
        if self.dkv_stage_on_sdS:
            need = max(cute.cosize(self.sK_layout), cute.cosize(self.sV_layout))
            have = cute.cosize(self.sPdS_layout)
            assert need <= have, f"staging dK/dV on sdS needs {need} elements and sdS holds {have}"
            self.dkv_stage_apart = cute.cosize(self.sK_layout) + cute.cosize(self.sV_layout) <= have
        if self.dkv_fold:
            # The integer accumulators stage on `sdS`. At head_dim 64 one
            # tensor is half of it, so dK and dV get a half each and never
            # wait for one another; at 128 one tensor alone is twice its size,
            # and the reduce goes out in passes, each waiting for the previous
            # one to have been read out of `sdS`.
            have = self.tile_m * self.tile_n * self.dS_stages * (self.dtype.width // 8)

            def _passes(need_one):
                n = 1 if 2 * need_one <= have else -(-need_one // have)
                # a pass is a whole number of 128-bit copies per warpgroup
                grain = self.num_wg_mma * 4
                while n > 1 and n <= self.tile_n and (self.tile_n * self.tile_hdim) % (n * grain):
                    n += 1
                assert need_one // n <= have
                return n

            need_dk = self.tile_n * self.tile_hdim * (self.dk_accum_type.width // 8)
            need_dv = self.tile_n * self.tile_hdim * (self.dv_accum_type.width // 8)
            self.dk_stage_pass = _passes(need_dk)
            self.dv_stage_pass = _passes(need_dv)
            self.dkv_split_halves = (
                self.dk_stage_pass == 1 and self.dv_stage_pass == 1 and need_dk + need_dv <= have
            )
            # where dV's slice starts inside `sdS`, in bytes
            self.dv_stage_offset = need_dk if self.dkv_split_halves else 0
        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.tile_hdim // self.num_wg_mma, self.num_wg_mma)
        )
        self.r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128),
            cute.make_layout((128, self.num_wg_mma)),
            cute.make_layout(4),
        )

    def _get_tiled_mma(self):
        # S^T = K Q^T and dP^T = V dO^T: warpgroup w owns key rows [64w, 64w + 64)
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.num_wg_mma, 1, 1),
            tiler_mn=(64, self.tile_m),
        )
        # dV = P^T dO and dK = dS^T Q, A from registers
        tiled_mma_dK, tiled_mma_dV = [
            sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=(self.num_wg_mma, 1, 1),
                tiler_mn=(64, self.tile_hdim),
                a_source=warpgroup.OperandSource.RMEM,
            )
            for _ in range(2)
        ]
        # dQ = dS K
        atom_layout_dQ = (self.atom_layout_m_dq, self.num_wg_mma // self.atom_layout_m_dq, 1)
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=atom_layout_dQ,
            tiler_mn=(64, self.tile_hdim // atom_layout_dQ[1]),
        )
        return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQaccum_struct = [
            cute.struct.Align[cute.struct.MemRange[t, cute.cosize(layout)], 1024]
            for (layout, t) in [
                (self.sQ_layout, self.dtype),
                (self.sK_layout, self.dtype),
                (self.sV_layout, self.dtype),
                (self.sdO_layout, self.dtype),
                (self.sdQaccum_layout, cutlass.Int32),
            ]
        ]
        cosize_sdS = cute.cosize(self.sPdS_layout)
        stat_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.stages], 128
        ]

        @cute.struct
        class SharedStorage:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            mbar_ptr_KV: cute.struct.MemRange[cutlass.Int64, 2 if self.persistent else 0]
            sLSE: stat_struct
            sdPsum: stat_struct
            # whether this CTA was the last to reach its key block
            sLast: cute.struct.MemRange[Int32, 1 if self.dkv_combine else 0]
            # the list scheduler's two claimed tile indices and their barriers
            sWork: cute.struct.MemRange[Int32, 2 if self.persistent else 0]
            mbar_ptr_sched: cute.struct.MemRange[cutlass.Int64, 4 if self.persistent else 0]
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
            sdQaccum: sdQaccum_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        # the preprocess's maxima, (batch, nheads_kv, N_STATS) float bits
        mStats: cute.Tensor,
        mCuSeqlens: cute.Tensor | None = None,
        mCuTotalMBlocks: cute.Tensor | None = None,
        mBlocksToBatch: cute.Tensor | None = None,
        # (tiles, 4) int32 `(n_block, head, batch, record)` in dispatch order
        mWorkList: cute.Tensor | None = None,
        # (2,) int32, zero on entry: the list scheduler's claim counter
        mSchedCounter: cute.Tensor | None = None,
        # dK/dV partials, `(b, h_kv, n_blocks, parts, mma threads, floats)`
        # dense or `(h_kv, padded key blocks, ...)` packed, and their arrival
        # counters, zero on entry and self-resetting
        mPart: cute.Tensor | None = None,
        mPartCtr: cute.Tensor | None = None,
        stream: cuda.CUstream = None,
    ):
        self.varlen = mCuSeqlens is not None
        # A dense call runs its work list on a persistent grid.
        self.persistent = mWorkList is not None
        # A persistent CTA stages its bf16 dK/dV tile on `sdS`, which is dead
        # by the epilogue, so the producer can refill `sK`/`sV` while the
        # store is still reading.
        self.dkv_stage_on_sdS = self.persistent and not self.dkv_fold
        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        ]

        # dense (b, s, n, h) and packed (s, n, h) to a seqlen-major view
        def _qkv_transpose(t):
            return layout_utils.select(t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1])

        mQ, mK, mV, mdO = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO)]
        if const_expr(not self.dkv_fold):
            mdK, mdV = [_qkv_transpose(t) for t in (mdK, mdV)]
        else:
            # the accumulators are (b, n, s*h), dense only
            mdK, mdV = [layout_utils.select(t, [2, 1, 0]) for t in (mdK, mdV)]
        # statistics are (b, n, s) dense and (n, s) packed
        stat_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        mLSE, mdPsum, mdQaccum = [
            layout_utils.select(t, stat_transpose) for t in (mLSE, mdPsum, mdQaccum)
        ]

        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * 4
        self.tma_copy_bytes["dQ"] = self.tile_m * self.tile_hdim * 4 // self.num_wg_mma
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.tile_hdim * self.dk_accum_type.width // 8
        self.tma_copy_bytes["dVacc"] = self.tile_n * self.tile_hdim * self.dv_accum_type.width // 8

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_dK = tma_atom_dV = None
        tma_tensor_dK, tma_tensor_dV = mdK, mdV
        if const_expr(not self.dkv_fold):
            ragged = lambda t: (
                copy_utils.create_ragged_tensor_for_tma(t, ragged_dim=0, ptr_shift=True)
                if self.varlen
                else t
            )
            tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                ragged(mdK),
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
            )
            tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                ragged(mdV),
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
            )

        if const_expr(self.persistent):
            TileScheduler = ListScheduler
            tile_sched_params = TileScheduler.to_underlying_arguments(mWorkList, mSchedCounter)
        else:
            TileScheduler = (
                SingleTileVarlenScheduler if const_expr(self.varlen) else SingleTileScheduler
            )
            tile_sched_args = TileSchedulerArguments(
                cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
                cute.size(mQ.shape[2]) // self.head_loop,
                cute.size(mK.shape[3])
                if const_expr(not self.varlen)
                else cute.size(mCuSeqlens.shape[0] - 1),
                1,
                cute.size(mQ.shape[0]),
                # the swizzle sizes L2 sections by one work tile's Q and dO
                mQ.shape[1] * self.head_loop,
                mV.shape[1] * self.head_loop,
                total_q=cute.size(mK.shape[0])
                if const_expr(self.varlen)
                else cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
                tile_shape_mn=(self.tile_n, self.tile_m),
                mCuSeqlensQ=mCuSeqlens,
                qhead_per_kvhead_packgqa=1,
                element_size=self.dtype.width // 8,
                is_persistent=False,
                lpt=False,
                # The swizzle is kept for L2 locality alone; nothing here
                # needs FA's shortest-first reversal.
                head_swizzle=self.varlen,
                cu_total_m_blocks_ptr=mCuTotalMBlocks,
                blocks_to_batch_idx_ptr=mBlocksToBatch,
            )
            tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        qhead_per_kvhead_divmod = None
        if const_expr(self.qhead_per_kvhead > 1):
            qhead_per_kvhead_divmod = FastDivmodDivisorV2(self.qhead_per_kvhead)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            tma_tensor_dK,
            tma_tensor_dV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mdPsum,
            mdQaccum,
            mCuSeqlens,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,
            self.r2s_tiled_copy_dQaccum,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            softmax_scale * math.log2(math.e),
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            qhead_per_kvhead_divmod,
            mStats,
            mPart,
            mPartCtr,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom | None,
        tma_atom_dV: cute.CopyAtom | None,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mCuSeqlens: cute.Tensor | None,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        qhead_per_kvhead_divmod: FastDivmodDivisorV2 | None,
        mStats: cute.Tensor,
        mPart: cute.Tensor | None,
        mPartCtr: cute.Tensor | None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for atom in [tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO, tma_atom_dK, tma_atom_dV]:
                if const_expr(atom is not None):
                    cpasync.prefetch_descriptor(atom)

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
        )
        pipeline_Q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"],
            defer_sync=True,
        )
        pipeline_dO = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["LSE"],
            defer_sync=const_expr(self.persistent),
        )
        # The K/V buffers' empty direction only: their arrival rides the Q and
        # dO transaction counts. Without it a persistent producer would
        # overwrite `sK`/`sV` while the dK/dV epilogue still reads them.
        pipeline_KV = None
        if const_expr(self.persistent):
            pipeline_KV = pipeline.PipelineAsync.create(
                barrier_storage=storage.mbar_ptr_KV.data_ptr(),
                num_stages=1,
                producer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread),
                # one arrive, from the epilogue's store warp
                consumer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread),
                elect_one_release=True,
                syncwarp_before_release=False,
                defer_sync=False,
            )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        # Staging buffers seen with `sK`/`sV`'s layout and swizzle, so the TMA
        # store descriptor is the one `sK`/`sV` would use and only the base
        # moves.
        sdK_stage = sdV_stage = None
        if const_expr(self.dkv_stage_on_sdS):
            sdV_stage = storage.sdS.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            if const_expr(self.dkv_stage_apart):
                # a whole number of 1 KB swizzle atoms past dV's tile
                sdK_stage = cute.make_tensor(
                    cute.recast_ptr(
                        storage.sdS.data_ptr() + cute.cosize(sV_layout),
                        sK_layout.inner,
                        dtype=self.dtype,
                    ),
                    sK_layout.outer,
                )
            else:
                sdK_stage = storage.sdS.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        stat_layout = cute.make_layout(
            (self.tile_m, self.stages), stride=(1, cute.round_up(self.tile_m, 64))
        )
        sLSE = storage.sLSE.get_tensor(stat_layout)
        sdPsum = storage.sdPsum.get_tensor(stat_layout)
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)
        sLast = None
        if const_expr(self.dkv_combine):
            sLast = storage.sLast.get_tensor(cute.make_layout(1))

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            False,
            False,
            None,
            None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlens,
            mCuSeqlensK=mCuSeqlens,
            mSeqUsedQ=None,
            mSeqUsedK=None,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        AttentionMaskCls = partial(AttentionMask, self.tile_m, self.tile_n, swap_AB=True)
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)
        TileSchedulerProd = TileSchedulerCls
        if const_expr(self.persistent):
            sWork = storage.sWork.get_tensor(cute.make_layout(2))
            pipeline_sched = pipeline.PipelineAsync.create(
                barrier_storage=storage.mbar_ptr_sched.data_ptr(),
                num_stages=2,
                producer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread),
                # the MMA warps and the dQ store warp, one arrive each
                consumer_group=cutlass.pipeline.CooperativeGroup(
                    cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE + 1
                ),
                elect_one_commit=True,
                elect_one_release=True,
                defer_sync=False,
            )
            TileSchedulerProd = partial(
                TileScheduler.create, tile_sched_params, ctx=(sWork, pipeline_sched, True)
            )
            TileSchedulerCls = partial(
                TileScheduler.create, tile_sched_params, ctx=(sWork, pipeline_sched, False)
            )

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    pipeline_Q,
                    pipeline_dO,
                    pipeline_KV,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerProd,
                    qhead_per_kvhead_divmod,
                )
            if warp_idx == 1:
                self.dQaccum_store(mdQaccum, sdQaccum, block_info, TileSchedulerCls, SeqlenInfoCls)
        else:
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            mma = partial(
                self.mma,
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                mdQaccum,
                sQ,
                sK,
                sV,
                sdO,
                sdS,
                sdK_stage,
                sdV_stage,
                sLSE,
                sdPsum,
                sdQaccum,
                pipeline_Q,
                pipeline_dO,
                pipeline_KV,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQaccum,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                qhead_per_kvhead_divmod,
                mStats,
                mPart,
                mPartCtr,
                sLast,
            )
            # One copy of the loop per warpgroup: its barrier ids and turn
            # partner are immediates rather than registers.
            if cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4 < 4:
                cute.arch.setmaxregister_increase(self.num_mma_regs)
                mma(warp_group_idx=0)
            else:
                cute.arch.setmaxregister_increase(self.num_mma_regs)
                mma(warp_group_idx=1)

    @cute.jit
    def _head_kv(self, head_idx, qhead_per_kvhead_divmod):
        """The KV head of a work tile's `head` field."""
        if const_expr(self.qhead_per_kvhead == 1):
            return head_idx
        return (head_idx * self.head_loop) // qhead_per_kvhead_divmod

    def _head_loaders(
        self,
        seqlen,
        batch_idx,
        hq,
        mQ,
        mdO,
        mLSE,
        mdPsum,
        sQ,
        sdO,
        sLSE,
        sdPsum,
        tma_atom_Q,
        tma_atom_dO,
        pipeline_Q,
        pipeline_dO,
    ):
        """Query head `hq`'s Q, dO, LSE and dPsum loaders."""
        gQ = cute.local_tile(
            seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, hq],
            (self.tile_m, self.tile_hdim),
            (None, 0),
        )
        gdO = cute.local_tile(
            seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, hq],
            (self.tile_m, self.tile_hdim),
            (None, 0),
        )
        cur = lambda mX: seqlen.offset_batch_Q(mX, batch_idx, dim=2, padded=True)[None, hq]
        vec = lambda mX: cute.local_tile(cur(mX), (self.tile_m,), (None,))
        load_Q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ)
        load_dO, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dO, 0, cute.make_layout(1), gdO, sdO)
        load_LSE = copy_utils.cpasync_bulk_get_copy_fn(vec(mLSE), sLSE)
        load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(vec(mdPsum), sdPsum)
        return (
            copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q),
            copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO),
            copy_utils.tma_producer_copy_fn(load_LSE, pipeline_Q),
            copy_utils.tma_producer_copy_fn(load_dPsum, pipeline_dO),
        )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        pipeline_KV: cutlass.pipeline.PipelineAsync | None,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        qhead_per_kvhead_divmod: FastDivmodDivisorV2 | None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if warp_idx_in_wg == 0:
            producer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.stages
            )
            producer_state_KV = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, 1
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                n_block, head_idx, batch_idx, record_idx = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                head_idx_kv = self._head_kv(head_idx, qhead_per_kvhead_divmod)
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK, single_stage=True
                )
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV, single_stage=True
                )

                m_block_min, m_block_max = self.m_range(block_info, seqlen, n_block, record_idx)
                process_tile = const_expr(not self.varlen) or m_block_min < m_block_max

                # The K/V buffer has to be free before its TMA lands in it.
                # This sits outside `process_tile` because the pipeline pairs
                # one acquire with one release per work tile, and the consumer
                # releases for skipped tiles too.
                if const_expr(self.persistent):
                    pipeline_KV.producer_acquire(producer_state_KV)
                    producer_state_KV.advance()
                if process_tile:
                    for _g in cutlass.range(self.head_loop, unroll=1):
                        lq = self._head_loaders(
                            seqlen,
                            batch_idx,
                            head_idx * self.head_loop + _g,
                            mQ,
                            mdO,
                            mLSE,
                            mdPsum,
                            sQ,
                            sdO,
                            sLSE,
                            sdPsum,
                            tma_atom_Q,
                            tma_atom_dO,
                            pipeline_Q,
                            pipeline_dO,
                        )
                        # K and V ride the tile's first stage only; the later
                        # heads find them resident
                        if _g == 0:
                            pipeline_Q.producer_acquire(
                                producer_state, extra_tx_count=self.tma_copy_bytes["K"]
                            )
                            load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state))
                        else:
                            pipeline_Q.producer_acquire(producer_state)
                        lq[0](m_block_min, producer_state=producer_state)
                        # the preprocess writes LSE and dPsum
                        cute.arch.griddepcontrol_wait()
                        lq[2](m_block_min, producer_state=producer_state)
                        if _g == 0:
                            pipeline_dO.producer_acquire(
                                producer_state, extra_tx_count=self.tma_copy_bytes["V"]
                            )
                            load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state))
                        else:
                            pipeline_dO.producer_acquire(producer_state)
                        lq[1](m_block_min, producer_state=producer_state)
                        lq[3](m_block_min, producer_state=producer_state)
                        producer_state.advance()
                        for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                            pipeline_Q.producer_acquire(producer_state)
                            lq[0](m_block, producer_state=producer_state)
                            lq[2](m_block, producer_state=producer_state)
                            pipeline_dO.producer_acquire(producer_state)
                            lq[1](m_block, producer_state=producer_state)
                            lq[3](m_block, producer_state=producer_state)
                            producer_state.advance()

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            if const_expr(self.persistent):
                tile_scheduler.producer_tail()
            # This CTA has no loads left to issue: the postprocess may launch
            # once every CTA is here, and its griddepcontrol_wait holds it
            # until the MMA warps' last stores have landed.
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdS: cute.Tensor,
        sdK_stage: cute.Tensor | None,
        sdV_stage: cute.Tensor | None,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        pipeline_KV: cutlass.pipeline.PipelineAsync | None,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom | None,
        tma_atom_dV: cute.CopyAtom | None,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        qhead_per_kvhead_divmod: FastDivmodDivisorV2 | None,
        mStats: cute.Tensor,
        mPart: cute.Tensor | None,
        mPartCtr: cute.Tensor | None,
        sLast: cute.Tensor | None,
        warp_group_idx: cutlass.Constexpr[int],
    ):
        # The preprocess writes the grids' maxima, and only the producer's
        # wait covers what the pipeline loads; these warps read them directly.
        cute.arch.griddepcontrol_wait()
        warp_group_thread_layout = cute.make_layout(self.num_wg_mma, stride=128)
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(warp_group_idx))
        # S = Q K^T
        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_S, sQ, sK, swap_AB=True
        )
        mma_qk_fn = partial(
            gemm_zero_init, tiled_mma_SdP, shape_mnk_S[:2], tSrQ, tSrK, swap_AB=True
        )
        # dP = dO V^T
        _, tdPrdO, tdPrV = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_S, sdO, sV, swap_AB=True
        )
        mma_dov_fn = partial(
            gemm_zero_init, tiled_mma_SdP, shape_mnk_S[:2], tdPrdO, tdPrV, swap_AB=True
        )
        # dV += P^T dO
        shape_mnk_dKV = (self.tile_n, self.tile_hdim, self.tile_m)
        acc_dV, _, tdVrdOt = sm90_utils.partition_fragment_ABC(
            wg_mma_dV, shape_mnk_dKV, None, layout_utils.transpose_view(sdO)
        )
        mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)
        # dK += dS^T Q
        acc_dK, _, tdKrQt = sm90_utils.partition_fragment_ABC(
            wg_mma_dK, shape_mnk_dKV, None, layout_utils.transpose_view(sQ)
        )
        mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)
        # dQ = dS K
        shape_mnk_dQ = (self.tile_m, self.tile_hdim, self.tile_n)
        _, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(
            wg_mma_dQ, shape_mnk_dQ, sdS, layout_utils.transpose_view(sK)
        )
        mma_dsk_fn = partial(gemm_zero_init, tiled_mma_dQ, shape_mnk_dQ[:2], tdQrdS, tdQrKt)
        tdQsdQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx).partition_D(sdQaccum)

        copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            layout_utils.transpose_view(sdS),
            tidx,
            transpose=True,
            position_independent=True,
            major_mode_size=self.tile_n // self.num_wg_mma,
        )
        tLSEsLSE = layout_utils.mma_partition_C_vec(
            sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=False
        )
        tLSEsdPsum = layout_utils.mma_partition_C_vec(
            sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=False
        )
        if const_expr(self.shuffle_stats):
            # rows spread across the 8 quads of a warp, 2 values a thread
            shfl_copy = copy_utils.tiled_copy_1d(Float32, num_threads=8, num_copy_elems=2)
            tLSEsLSE, tLSEsdPsum = [
                cute.group_modes(
                    shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(t), 0, 2
                )
                for t in (tLSEsLSE, tLSEsdPsum)
            ]

        one_m_block = partial(
            self.mma_one_m_block,
            warp_group_idx=warp_group_idx,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            mma_dsk_fn=mma_dsk_fn,
            copy_dS_r2s=copy_dS_r2s,
            pipeline_Q=pipeline_Q,
            pipeline_dO=pipeline_dO,
            tLSEsLSE=tLSEsLSE,
            tLSEsdPsum=tLSEsdPsum,
            tdQsdQaccum=tdQsdQaccum,
            softmax_scale_log2=softmax_scale_log2,
        )
        consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.stages
        )
        consumer_state_KV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # blocks this warpgroup has written dS for, which picks the dS slot
        ds_cnt = Int32(0)
        if const_expr(warp_group_idx == 1):
            # warpgroup 0 issues first
            self._turn_arrive(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, record_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mask = AttentionMaskCls(seqlen)
            head_kv = self._head_kv(head_idx, qhead_per_kvhead_divmod)
            # One exponent per (request, KV head): the softmax carries it into
            # P, so dQ arrives on the grid and dK/dV leave scaled by it.
            if const_expr(self.dq_fp32):
                lse_bias = Float32(0.0)
            else:
                lse_bias = grid.log2_pow2(grid.dq_scale(mStats, batch_idx, head_kv, self.root_d))
            m_block_min, m_block_max = self.m_range(block_info, seqlen, n_block, record_idx)
            process_tile = const_expr(not self.varlen) or m_block_min < m_block_max

            if process_tile:
                mask_fn = partial(
                    mask.apply_mask,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    n_block=n_block,
                    thr_mma=thr_mma_SdP,
                    mask_seqlen=True,
                    mask_causal=self.is_causal,
                )
                block = partial(one_m_block, lse_bias=lse_bias, mask_fn=mask_fn)
                # The masked query blocks come first: under causal, those
                # whose first row does not yet cover the key block's last
                # column; without it, every block of a key block that runs
                # past the sequence. Query rows past the sequence need no
                # mask, as their padded statistics zero P.
                if const_expr(self.is_causal):
                    m_free = cute.ceil_div(
                        (n_block + 1) * self.tile_n - 1 + seqlen.seqlen_q - seqlen.seqlen_k,
                        self.tile_m,
                    )
                    m_mask_end = min(max(m_free, m_block_min), m_block_max)
                else:
                    m_mask_end = m_block_min
                    if (n_block + 1) * self.tile_n > seqlen.seqlen_k:
                        m_mask_end = m_block_max
                # The tile's first block has no dQ before it to issue. Its
                # accumulate flags are constants, so dK and dV are never live
                # into the tile.
                consumer_state, ds_cnt = block(
                    m_block_min, consumer_state, ds_cnt, m_mask_end=m_block_max, first=True
                )
                for _g in cutlass.range(self.head_loop, unroll=1):
                    for m_block in cutlass.range(
                        m_block_min + 1 - min(_g, 1), m_block_max, unroll=1
                    ):
                        consumer_state, ds_cnt = block(
                            m_block, consumer_state, ds_cnt, m_mask_end=m_mask_end, first=False
                        )
                self.dQ_last(ds_cnt, warp_group_idx, mma_dsk_fn, tdQsdQaccum)

                # Rebuilt rather than kept: the m-loop then carries neither
                # the sequence's offsets nor the grid's unscale, which are
                # registers the loop's WGMMAs need.
                seqlen = SeqlenInfoCls(batch_idx)
                if const_expr(self.dq_fp32):
                    dkv_unscale = Float32(1.0)
                else:
                    dkv_unscale = grid.inv_pow2(
                        grid.dq_scale(mStats, batch_idx, head_kv, self.root_d)
                    )
                if const_expr(self.dkv_fold):
                    self.epilogue_dKV_fold(
                        acc_dV,
                        mdV,
                        acc_dK,
                        mdK,
                        sdS,
                        seqlen,
                        tidx,
                        n_block,
                        head_kv,
                        batch_idx,
                        pipeline_KV,
                        consumer_state_KV,
                        mStats,
                        dkv_unscale,
                    )
                else:
                    is_last = Boolean(True)
                    if const_expr(self.dkv_combine):
                        is_last, part_base, part_stride = self.combine_dKV(
                            acc_dK,
                            acc_dV,
                            tidx,
                            seqlen,
                            n_block,
                            head_idx,
                            head_kv,
                            batch_idx,
                            mPart,
                            mPartCtr,
                            sLast,
                        )
                    if is_last:
                        if const_expr(self.dkv_combine):
                            # read back in the branch that uses the sum, so the
                            # accumulators never merge across a branch
                            self._combine_read(acc_dK, acc_dV, part_base, part_stride, tidx, sdS)
                        # P carried the grid's `2^s` through the whole m-loop,
                        # so dK and dV leave scaled by the same power of two
                        # and are put back here, exactly.
                        acc_dK.store(acc_dK.load() * (softmax_scale * dkv_unscale))
                        acc_dV.store(acc_dV.load() * dkv_unscale)
                        self.epilogue_dKV(
                            acc_dV,
                            mdV,
                            sV,
                            acc_dK,
                            mdK,
                            sK,
                            seqlen,
                            tma_atom_dK,
                            tma_atom_dV,
                            tiled_mma_dK,
                            tiled_mma_dV,
                            tidx,
                            n_block,
                            head_kv,
                            batch_idx,
                            pipeline_KV,
                            consumer_state_KV,
                            sdK_stage,
                            sdV_stage,
                        )
                    elif const_expr(self.persistent):
                        # the combine's barrier has every MMA done with `sK`/`sV`
                        if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == 4:
                            pipeline_KV.consumer_release(consumer_state_KV)
            elif const_expr(self.persistent):
                # Nothing read the slot, so it is free the moment it was
                # acquired; hand it straight back.
                if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == 4:
                    pipeline_KV.consumer_release(consumer_state_KV)
            if const_expr(self.persistent):
                consumer_state_KV.advance()

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(warp_group_idx == 0):
            # warpgroup 1's last hand-back has no later turn to consume it
            self._turn_sync(0)
        if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == 4:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def _turn_sync(self, wg):
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1) + wg, number_of_threads=2 * 128
        )

    @cute.jit
    def _turn_arrive(self, wg):
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1) + wg, number_of_threads=2 * 128
        )

    @cute.jit
    def mma_one_m_block(
        self,
        m_block,
        cs,
        ds_cnt,
        warp_group_idx: cutlass.Constexpr[int],
        mma_qk_fn,
        mma_dov_fn,
        mma_pdo_fn,
        mma_dsq_fn,
        mma_dsk_fn,
        copy_dS_r2s,
        pipeline_Q,
        pipeline_dO,
        tLSEsLSE,
        tLSEsdPsum,
        tdQsdQaccum,
        softmax_scale_log2,
        lse_bias,
        mask_fn,
        m_mask_end,
        first: cutlass.Constexpr[bool],
    ):
        """One query block, with the warpgroups in ping-pong.

        GEMM groups are issued in strict turn order, warpgroup 0's S/dP, then
        warpgroup 1's, then their dV/dK, so one warpgroup's exponentials, dS
        and quantiser run under the other's GEMMs. The previous block's dQ
        rides in the second group: by then both halves of its dS have been
        stored, which the turn order guarantees (a warpgroup hands over its
        first turn only after its previous dS store and fence), and the
        order-free integer reduce does not care that dQ lands a block late.
        """
        smem_idx = cs.index
        other = 1 - warp_group_idx
        slot = ds_cnt % self.dS_stages
        prev = (ds_cnt + self.dS_stages - 1) % self.dS_stages
        pipeline_Q.consumer_wait(cs, pipeline_Q.consumer_try_wait(cs))
        pipeline_dO.consumer_wait(cs, pipeline_dO.consumer_try_wait(cs))
        # S = Q K^T and dP = dO V^T
        self._turn_sync(warp_group_idx)
        acc_S = mma_qk_fn(A_idx=smem_idx, wg_wait=-1)
        acc_dP = mma_dov_fn(A_idx=smem_idx, wg_wait=-1)
        self._turn_arrive(other)
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx])
        warpgroup.wait_group(1)
        # P = exp(S - LSE), computed as `P 2^s`: the grid's exponent is a
        # constant of the head, so the softmax carries it and the dQ partials
        # leave the GEMM on the grid. A runtime branch rather than a second
        # copy of the loop, which ptxas cannot fit in the register budget.
        if m_block < m_mask_end:
            mask_fn(acc_S, m_block=m_block)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=True)
        lane_idx = cute.arch.lane_idx()
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_stats) - lse_bias
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(
                    acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True
                )
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx])
        warpgroup.wait_group(0)
        # dS = P (dP - dPsum)
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=True)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = self._get_stat(tLSErdPsum, r, lane_idx, shuffle=self.shuffle_stats)
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)
        # P is packed only once dS no longer needs it in fp32, so the two
        # never hold registers together
        tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
        tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)
        copy_dS_r2s(tdKrdS, dst_idx=slot)
        cute.arch.fence_view_async_shared()
        # the previous block's dQ = dS K, dV += P^T dO, dK += dS^T Q
        self._turn_sync(warp_group_idx)
        if const_expr(not first):
            acc_dQ = mma_dsk_fn(A_idx=prev, wg_wait=-1)
        zero_init = const_expr(first)
        mma_pdo_fn(tCrA=tdVrP, B_idx=smem_idx, zero_init=zero_init, wg_wait=-1)
        mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx, zero_init=zero_init, wg_wait=-1)
        self._turn_arrive(other)
        if const_expr(not first):
            warpgroup.wait_group(2)
            self._quantise_dQ(acc_dQ, tdQsdQaccum, warp_group_idx)
        warpgroup.wait_group(1)
        pipeline_dO.consumer_release(cs)
        warpgroup.wait_group(0)
        pipeline_Q.consumer_release(cs)
        cs.advance()
        return cs, ds_cnt + 1

    @cute.jit
    def dQ_last(self, ds_cnt, warp_group_idx: cutlass.Constexpr[int], mma_dsk_fn, tdQsdQaccum):
        """The tile's last dQ, which no later block carries, in a turn of its
        own so the order stays alternating."""
        prev = (ds_cnt + self.dS_stages - 1) % self.dS_stages
        self._turn_sync(warp_group_idx)
        acc_dQ = mma_dsk_fn(A_idx=prev, wg_wait=-1)
        self._turn_arrive(1 - warp_group_idx)
        warpgroup.wait_group(0)
        self._quantise_dQ(acc_dQ, tdQsdQaccum, warp_group_idx)

    @staticmethod
    @cute.jit
    def _get_stat(tSrS: cute.Tensor, row: Int32, lane: Int32, shuffle: bool) -> Float32:
        """The statistic for accumulator row `row`: a register, or a shuffle
        from the thread of the 8 sharing the row that holds it."""
        if const_expr(not shuffle):
            return tSrS[row]
        vecsize = cute.size(tSrS, mode=[0, 0])
        idx0, off, idx1 = cute.idx2crd(row, (vecsize, 8, cute.shape(tSrS, mode=[0, 1])))
        return utils.shuffle_sync(tSrS[idx0 + idx1 * vecsize], offset=off * 4 + (lane % 4))

    @cute.jit
    def _quantise_dQ(self, acc_dQ, tdQsdQaccum, wg: cutlass.Constexpr[int]):
        """Round the grid-scaled dQ partial in place and hand it to the store
        warp. Both views are 32-bit, and element `i` is written only after it
        is read. An fp32 partial's bits go out unchanged through the int32
        view."""
        acc_flat = cute.make_tensor(acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape))
        rdQfix = cute.make_tensor(
            cute.recast_ptr(acc_dQ.iterator, dtype=cutlass.Int32),
            cute.make_layout(cute.size(acc_dQ)),
        )
        if const_expr(not self.dq_fp32):
            for i in cutlass.range_constexpr(cute.size(tdQsdQaccum)):
                rdQfix[i] = cvt_rni_s32_f32(acc_flat[i])
        # wait for the dQ store warp to have read the previous block out
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + wg,
            number_of_threads=128 + cute.arch.WARP_SIZE,
        )
        cute.autovec_copy(rdQfix, tdQsdQaccum)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.dQFullWG0) + wg,
            number_of_threads=128 + cute.arch.WARP_SIZE,
        )

    @cute.jit
    def m_range(self, block_info, seqlen, n_block, record_idx):
        """The query blocks a work tile walks: the key block's causal range,
        or its `record_idx`-th canonical record."""
        m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
        if const_expr(self.record_width > 0):
            lo = m_block_min + record_idx * self.record_width
            hi = min(m_block_max, lo + self.record_width)
            m_block_min, m_block_max = lo, max(lo, hi)
        return m_block_min, m_block_max

    @cute.jit
    def combine_dKV(
        self,
        acc_dK,
        acc_dV,
        tidx,
        seqlen,
        n_block,
        head_idx,
        head_kv,
        batch_idx,
        mPart,
        mPartCtr,
        sLast,
    ):
        """Publish this tile's fp32 dK/dV partial. Returns True in the CTA that
        arrives last at its key block, which then sums every partial.

        Nobody waits and nothing depends on dispatch order: a CTA that is not
        last leaves, and the sum runs in partial-index order, so it is the
        same bits whichever CTA arrives last. A partial is stored as 16-byte
        groups, group-major over the threads, so both the publish and the
        read-back are coalesced.
        """
        epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.Epilogue), num_threads=self.num_mma_threads
        )
        nreg = cute.size(acc_dK)
        regs = [cute.make_tensor(acc.iterator, cute.make_layout(nreg)) for acc in (acc_dK, acc_dV)]
        n_grp = nreg // 4
        nthr = self.num_mma_threads
        sub = head_idx % self.n_sub
        if const_expr(not self.varlen):
            part = mPart[batch_idx, head_kv, n_block, None, 0, None]
            ctr = mPartCtr[batch_idx, head_kv, n_block, None].iterator.toint()
        else:
            slot = seqlen.padded_offset_k // self.tile_n + n_block
            part = mPart[head_kv, slot, None, 0, None]
            ctr = mPartCtr[head_kv, slot, None].iterator.toint()
        stride = 2 * nreg * 4 * nthr
        base = part.iterator.toint() + Int64(tidx) * 16
        for w in cutlass.range_constexpr(2):
            for j in cutlass.range_constexpr(n_grp):
                cute.autovec_copy(
                    cute.make_tensor(regs[w].iterator + 4 * j, cute.make_layout(4)),
                    self._part_grp(base, stride, sub, w * n_grp + j),
                )
        fence_acq_rel_gpu()
        epi_barrier.arrive_and_wait()
        if tidx == 0:
            last = Int32(0)
            # The release publishes this CTA's partial; the acquire makes every
            # earlier arriver's visible to the last one. Counting is all the
            # synchronisation the sum needs: its order is fixed by the index.
            if atomic_add_acq_rel_gpu(ctr, 1) == self.n_sub - 1:
                store_release_gpu(ctr, 0)
                last = Int32(1)
            sLast[0] = last
        epi_barrier.arrive_and_wait()
        is_last = sLast[0] != 0
        # `sLast` is rewritten by the next tile's combine only after its own
        # first barrier, which every thread reaches after this read
        return is_last, base, stride

    def _part_grp(self, base, stride, q, g):
        """Partial `q`'s 16-byte group `g` of this thread."""
        return cute.make_tensor(
            cute.make_ptr(
                Float32,
                base + Int64(q) * stride + g * self.num_mma_threads * 16,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout(4),
        )

    def _stage_grp(self, sbuf, n_buf, tidx, c, j):
        off = (((c % n_buf) * 4 + j) * self.num_mma_threads + tidx) * 16
        return cute.make_tensor(
            cute.make_ptr(Float32, sbuf + off, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout(4),
        )

    @cute.jit
    def _combine_read(self, acc_dK, acc_dV, base, stride, tidx, sdS):
        """Every partial summed in partial order, streamed through the dead
        `sdS` in chunks of 4 groups (16 KB) with `cp.async`, each thread
        reading back only what its own copies wrote. Loaded to registers
        directly, ptxas hoists every load and the kernel's WGMMAs lose their
        registers."""
        regs_dK, regs_dV = [
            cute.make_tensor(acc.iterator, cute.make_layout(cute.size(acc)))
            for acc in (acc_dK, acc_dV)
        ]
        regs = [regs_dK, regs_dV]
        n_grp = cute.size(regs_dK) // 4
        n_chunk = 2 * n_grp // 4
        n_buf = cute.cosize(sdS.layout) * (self.dtype.width // 8) // (4 * self.num_mma_threads * 16)
        ahead = min(n_buf - 1, n_chunk - 1)
        sbuf = sdS.iterator.toint()
        g2s = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
            Float32,
            num_bits_per_copy=128,
        )
        for w in cutlass.range_constexpr(2):
            regs[w].fill(0.0)
        for q in cutlass.range(self.n_sub, unroll=1):
            for c in cutlass.range_constexpr(n_chunk + ahead):
                if const_expr(c < n_chunk):
                    for j in cutlass.range_constexpr(4):
                        cute.copy(
                            g2s,
                            self._part_grp(base, stride, q, c * 4 + j),
                            self._stage_grp(sbuf, n_buf, tidx, c, j),
                        )
                    cute.arch.cp_async_commit_group()
                if const_expr(c >= ahead):
                    cc = c - ahead
                    cute.arch.cp_async_wait_group(min(ahead, n_chunk - 1 - cc))
                    for j in cutlass.range_constexpr(4):
                        g = cc * 4 + j
                        w, jj = g // n_grp, g % n_grp
                        v = cute.make_rmem_tensor(4, Float32)
                        cute.autovec_copy(self._stage_grp(sbuf, n_buf, tidx, cc, j), v)
                        for e in cutlass.range_constexpr(4):
                            regs[w][4 * jj + e] = regs[w][4 * jj + e] + v[e]

    @cute.jit
    def epilogue_dKV_fold(
        self,
        acc_dV,
        mdV,
        acc_dK,
        mdK,
        sdS,
        seqlen,
        tidx,
        n_block,
        head_kv,
        batch_idx,
        pipeline_KV,
        consumer_state_KV,
        mStats,
        dkv_unscale,
    ):
        """One canonical record's dK/dV, rounded onto the integer grids and
        reduced into the accumulators asynchronously; the postprocess converts
        them. Staged on `sdS`, which is dead by the epilogue, never on K/V."""
        epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.Epilogue), num_threads=self.num_mma_threads
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        nk_pass, nv_pass = self.dk_stage_pass, self.dv_stage_pass
        # one warpgroup's slice of one pass
        dk_slice = self.tile_n * self.tile_hdim // (self.num_wg_mma * nk_pass)
        dv_slice = self.tile_n * self.tile_hdim // (self.num_wg_mma * nv_pass)
        gdKaccum = cute.flat_divide(
            cute.local_tile(
                seqlen.offset_batch_K(mdK, batch_idx, dim=2, padded=True, multiple=self.tile_hdim)[
                    None, head_kv
                ],
                (self.tile_n * self.tile_hdim,),
                (n_block,),
            ),
            (dk_slice,),
        )
        gdVaccum = cute.flat_divide(
            cute.local_tile(
                seqlen.offset_batch_K(mdV, batch_idx, dim=2, padded=True, multiple=self.tile_hdim)[
                    None, head_kv
                ],
                (self.tile_n * self.tile_hdim,),
                (n_block,),
            ),
            (dv_slice,),
        )
        # two halves of `sdS` wherever both fit, so dV need not wait for dK's
        # reduce to have read its slice; one shared buffer otherwise
        sdKaccum = cute.make_tensor(
            cute.recast_ptr(sdS.iterator, dtype=self.dk_accum_type),
            cute.make_layout((dk_slice, self.num_wg_mma)),
        )
        sdV_base = sdS.iterator
        if const_expr(self.dv_stage_offset):
            sdV_base = cute.recast_ptr(sdS.iterator, dtype=cutlass.Int8) + self.dv_stage_offset
        sdVaccum = cute.make_tensor(
            cute.recast_ptr(sdV_base, dtype=self.dv_accum_type),
            cute.make_layout((dv_slice, self.num_wg_mma)),
        )

        def _r2s(acc_type):
            return cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), acc_type, num_bits_per_copy=128),
                cute.make_layout((128, self.num_wg_mma)),
                cute.make_layout(128 // acc_type.width),
            ).get_slice(tidx)

        tdKsdKaccum = _r2s(self.dk_accum_type).partition_D(sdKaccum)
        tdVsdVaccum = _r2s(self.dv_accum_type).partition_D(sdVaccum)

        cute.arch.cp_async_bulk_wait_group(0, read=True)
        epi_barrier.arrive_and_wait()
        # this path never reads `sK`/`sV`, so the slot is free once the
        # m-loop's last WGMMA has retired
        if const_expr(self.persistent):
            if warp_idx == 4:
                pipeline_KV.consumer_release(consumer_state_KV)
        st = grid.load_stats(mStats, batch_idx, head_kv)
        # the number of query rows, which bounds the column mass
        rows = Float32(seqlen.seqlen_q * self.qhead_per_kvhead)
        dk_sc = grid.dk_scale(st, rows, self.root_d, self.dk_grid_bits) * dkv_unscale
        dv_sc = grid.dv_scale(st, rows, self.dv_grid_bits) * dkv_unscale

        def _stage_and_reduce(
            acc,
            sdaccum,
            gdaccum,
            tdsdaccum,
            sc,
            nbytes_total,
            p,
            bits: cutlass.Constexpr,
            red_type: cutlass.Constexpr,
            n_pass: cutlass.Constexpr,
        ):
            per_thread = cute.size(tdsdaccum)
            rflat = cute.make_tensor(acc.iterator + p * per_thread, tdsdaccum.shape)
            if const_expr(bits == 64):
                # an int64 value is two fp32 slots, so it cannot overwrite the
                # accumulator in place
                rfix = cute.make_rmem_tensor(tdsdaccum.shape, cutlass.Int64)
                for i in cutlass.range_constexpr(per_thread):
                    rfix[i] = cvt_rni_s64_f32(rflat[i] * sc)
            else:
                rfix = cute.make_tensor(
                    cute.recast_ptr(acc.iterator + p * per_thread, dtype=cutlass.Int32),
                    tdsdaccum.shape,
                )
                for i in cutlass.range_constexpr(per_thread):
                    rfix[i] = cvt_rni_s32_f32(rflat[i] * sc)
            cute.autovec_copy(rfix, tdsdaccum)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_wg_mma):
                        # the tile is warpgroup-major, so warpgroup `w`'s pass
                        # `p` is global slice `w * n_pass + p`
                        copy_utils.cpasync_bulk_s2g(
                            sdaccum[None, wg_idx].iterator,
                            gdaccum[None, wg_idx * n_pass + p].iterator,
                            nbytes_total // (self.num_wg_mma * n_pass),
                            reduction_kind=cpasync.ReductionOp.ADD,
                            dtype=red_type,
                        )
                cute.arch.cp_async_bulk_commit_group()

        for p in cutlass.range_constexpr(nk_pass):
            if const_expr(p > 0):
                # the slice just reduced has to have been read out of `sdS`
                if warp_idx == 4:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                epi_barrier.arrive_and_wait()
            _stage_and_reduce(
                acc_dK,
                sdKaccum,
                gdKaccum,
                tdKsdKaccum,
                dk_sc,
                self.tma_copy_bytes["dKacc"],
                p,
                self.dk_accum_bits,
                self.dk_reduce_type,
                nk_pass,
            )
        epi_barrier.arrive_and_wait()
        for p in cutlass.range_constexpr(nv_pass):
            if const_expr(p > 0 or not self.dkv_split_halves):
                if warp_idx == 4:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                epi_barrier.arrive_and_wait()
            _stage_and_reduce(
                acc_dV,
                sdVaccum,
                gdVaccum,
                tdVsdVaccum,
                dv_sc,
                self.tma_copy_bytes["dVacc"],
                p,
                self.dv_accum_bits,
                self.dv_reduce_type,
                nv_pass,
            )
        # `sdS` is the next tile's dS buffer as soon as this loop turns, so both
        # reduces have to have read it first
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        epi_barrier.arrive_and_wait()

    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_kv: Int32,
        batch_idx: Int32,
        pipeline_KV: cutlass.pipeline.PipelineAsync | None,
        consumer_state_KV,
        sdK_stage: cute.Tensor | None,
        sdV_stage: cute.Tensor | None,
    ):
        epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.Epilogue), num_threads=self.num_mma_threads
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3, ragged=self.varlen)[
            None, None, head_kv
        ]
        mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3, ragged=self.varlen)[
            None, None, head_kv
        ]
        gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
        gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
        # `sK`/`sV` when the CTA owns them for its lifetime, the staging
        # buffers when a persistent producer may already be refilling them
        bufK = sdK_stage if const_expr(self.dkv_stage_on_sdS) else sK
        bufV = sdV_stage if const_expr(self.dkv_stage_on_sdS) else sV
        store_dK, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_dK, 0, cute.make_layout(1), bufK, gdK, single_stage=True
        )
        store_dV, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_dV, 0, cute.make_layout(1), bufV, gdV, single_stage=True
        )
        copy_dV_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_dV, bufV, tidx, transpose=False, position_independent=True
        )
        copy_dK_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_dK, bufK, tidx, transpose=False, position_independent=True
        )
        # stops the epilogue trampling a store the previous tile issued out of
        # the same buffer
        if const_expr(not self.dkv_stage_on_sdS):
            cute.arch.cp_async_bulk_wait_group(1, read=True)
        epi_barrier.arrive_and_wait()
        # Staging never touches `sK`/`sV`, so the slot is free once
        # the m-loop's last WGMMA has retired, which the barrier above
        # establishes; releasing here gives the producer the whole epilogue to
        # fetch the next tile's K/V.
        if const_expr(self.dkv_stage_on_sdS):
            if warp_idx == 4:
                pipeline_KV.consumer_release(consumer_state_KV)
        if const_expr(self.dkv_stage_apart):
            copy_dV_r2s(acc_dV)
            copy_dK_r2s(acc_dK)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dV()
                store_dK()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            # `sdS` is the next tile's dS buffer as soon as this loop turns
            epi_barrier.arrive_and_wait()
        elif const_expr(self.dkv_stage_on_sdS):
            copy_dV_r2s(acc_dV)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dV()
                cute.arch.cp_async_bulk_commit_group()
                # dK reuses the buffer dV was just stored out of
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            epi_barrier.arrive_and_wait()
            copy_dK_r2s(acc_dK)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dK()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            # `sdS` is the next tile's dS buffer as soon as this loop turns
            epi_barrier.arrive_and_wait()
        else:
            # a one-tile CTA stages on the `sK`/`sV` it owns
            copy_dV_r2s(acc_dV)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dV()
                cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            epi_barrier.arrive_and_wait()
            copy_dK_r2s(acc_dK)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dK()
                cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def dQaccum_store(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        block_info: BlockInfo,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
    ):
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, record_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = self.m_range(block_info, seqlen, n_block, record_idx)
            for _g in cutlass.range(self.head_loop, unroll=1):
                head_q = head_idx * self.head_loop + _g
                if const_expr(not self.varlen):
                    mdQaccum_cur = mdQaccum[None, head_q, batch_idx]
                else:
                    mdQaccum_cur = cute.domain_offset(
                        (seqlen.padded_offset_q * self.tile_hdim,), mdQaccum[None, head_q]
                    )
                # ((M * K / 2, 2), num_m_blocks), a half per warpgroup
                gdQaccum = cute.local_tile(
                    mdQaccum_cur,
                    (
                        cute.make_layout(
                            (self.tile_m * self.tile_hdim // self.num_wg_mma, self.num_wg_mma)
                        ),
                    ),
                    (None,),
                )
                for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                    for wg in cutlass.range_constexpr(self.num_wg_mma):
                        cute.arch.cp_async_bulk_wait_group(self.num_wg_mma - 1 - wg, read=True)
                        cute.arch.barrier_arrive(
                            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + wg,
                            number_of_threads=128 + cute.arch.WARP_SIZE,
                        )
                    for wg in cutlass.range_constexpr(self.num_wg_mma):
                        cute.arch.barrier(
                            barrier_id=int(NamedBarrierBwd.dQFullWG0) + wg,
                            number_of_threads=128 + cute.arch.WARP_SIZE,
                        )
                        with cute.arch.elect_one():
                            # an integer add is associative, so the order the
                            # key blocks land in cannot change the sum
                            copy_utils.cpasync_bulk_s2g(
                                sdQaccum[None, wg].iterator,
                                gdQaccum[(None, wg), m_block].iterator,
                                self.tma_copy_bytes["dQ"],
                                reduction_kind=cpasync.ReductionOp.ADD,
                                dtype=self.dq_reduce_type,
                            )
                        cute.arch.cp_async_bulk_commit_group()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        cute.arch.cp_async_bulk_wait_group(0, read=True)
