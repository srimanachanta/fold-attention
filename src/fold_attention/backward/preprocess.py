"""The backward's preprocess: Delta, the grids' maxima, and the dQ clear.

Adapted from FlashAttention's backward preprocess (BSD-3-Clause). Beside
`Delta_i = dO_i . O_i` and the base-2 LSE it takes the maxima the integer
grids need (`grid.py`) with `red.max` over float bits, which lands the same in
any order, and clears the int32 dQ accumulator.
"""

import math
import operator
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
from cutlass import Float32, Int32, const_expr, cute
from flash_attn.cute import utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.seqlen_info import SeqlenInfo
from flash_attn.cute.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from quack import copy_utils, layout_utils
from quack.cute_dsl_utils import ParamsBase

from .grid import ST_DELTA, ST_DO, ST_DOC, ST_K, ST_Q, ST_V

_N_MAXIMA = 6


class FoldBackwardPreprocess:
    def __init__(
        self,
        dtype,
        head_dim: int,
        tile_m: int,
        kv_group: int,
        record_stats: bool,
        num_threads: int = 256,
    ):
        self.dtype = dtype
        self.tile_m = tile_m
        self.head_dim = head_dim
        self.kv_group = kv_group
        # max |dO_id| and max |Q| feed only the canonical records' dK/dV grids
        self.record_stats = bool(record_stats)
        self.num_threads = num_threads
        assert num_threads >= tile_m

    def _setup_attributes(self):
        gmem_k_block_size = (
            128 if self.head_dim % 128 == 0 else 64 if self.head_dim % 64 == 0 else 32
        )
        num_copy_elems = 128 // self.dtype.width
        self.gmem_tiled_copy_O = copy_utils.tiled_copy_2d(
            self.dtype, gmem_k_block_size // num_copy_elems, self.num_threads, num_copy_elems
        )
        # one int32 block of dQ in whole 128-bit vectors per thread, which every
        # supported tile divides exactly, so the clear never leaves its block
        self.zero_elems = self.tile_m * self.head_dim
        assert self.zero_elems % (self.num_threads * 4) == 0
        self.gmem_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(cutlass.Int32, self.num_threads, 4)

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,  # (batch, seqlen, nheads, head_dim) or (total_q, nheads, head_dim)
        mdO: cute.Tensor,
        mPdPsum: cute.Tensor,  # (batch, nheads, seqlen_padded) or (nheads, total_q_padded)
        mLSE: cute.Tensor,  # (batch, nheads, seqlen) or (nheads, total_q), nats
        mLSElog2: cute.Tensor,  # same shape as mPdPsum
        mdQaccum: cute.Tensor,
        mStats: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mCuSeqlens: cute.Tensor | None = None,
        mCuTotalMBlocks: cute.Tensor | None = None,
        mBlocksToBatch: cute.Tensor | None = None,
        stream: cuda.CUstream = None,
    ):
        self._setup_attributes()
        # Compiled against real tensors, the strides stay symbolic, and a
        # vectorised load cannot be proven legal without these.
        mO, mdO, mPdPsum, mLSElog2, mdQaccum, mQ, mK, mV = [
            assume_tensor_aligned(t) if t is not None else None
            for t in (mO, mdO, mPdPsum, mLSElog2, mdQaccum, mQ, mK, mV)
        ]
        varlen = mCuSeqlens is not None
        qo_transpose = [0, 2, 1] if const_expr(varlen) else [1, 3, 2, 0]
        mO, mdO, mQ, mK, mV = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=qo_transpose))
            for mX in (mO, mdO, mQ, mK, mV)
        ]
        transpose = [1, 0] if const_expr(varlen) else [2, 1, 0]
        mPdPsum, mLSE, mLSElog2, mdQaccum = [
            layout_utils.select(mX, transpose) if mX is not None else None
            for mX in (mPdPsum, mLSE, mLSElog2, mdQaccum)
        ]
        if const_expr(varlen):
            TileScheduler = SingleTileVarlenScheduler
            num_batch = mCuSeqlens.shape[0] - 1
        else:
            TileScheduler = SingleTileScheduler
            num_batch = mO.shape[3]
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mO.shape[0], self.tile_m),
            num_head=mO.shape[2],
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=0,
            headdim_v=mO.shape[1],
            total_q=cute.size(mO.shape[0])
            if const_expr(varlen)
            else cute.size(mO.shape[0]) * cute.size(mO.shape[3]),
            tile_shape_mn=(self.tile_m, 1),
            mCuSeqlensQ=mCuSeqlens,
            qhead_per_kvhead_packgqa=1,
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
            blocks_to_batch_idx_ptr=mBlocksToBatch,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.kernel(
            mO,
            mdO,
            mPdPsum,
            mLSE,
            mLSElog2,
            mdQaccum,
            mCuSeqlens,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_dQaccum,
            tile_sched_params,
            TileScheduler,
            mStats,
            mQ,
            mK,
            mV,
        ).launch(
            grid=TileScheduler.get_grid_shape(tile_sched_params),
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def _load_rows(self, tXgX, tXrX, t0OcO, tOcO, seqlen_limit):
        # t0OcO's entries are compile-time, so compare against the limit minus
        # this thread's row offset
        for m in cutlass.range(cute.size(tXrX.shape[1]), unroll_full=True):
            if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                copy_utils.copy(tXgX[None, m, None], tXrX[None, m, None])

    @cute.jit
    def _block_absmax(
        self, mX, head, seqlen, batch_idx, m_block, gmem_thr_copy_O, t0OcO, tOcO, seqlen_limit
    ):
        """max |X| over this block's rows of head `head`."""
        gX = cute.local_tile(
            seqlen.offset_batch(mX, batch_idx, dim=3)[None, None, head],
            (self.tile_m, self.head_dim),
            (m_block, 0),
        )
        tXgX = gmem_thr_copy_O.partition_S(gX)
        tXrX = cute.make_rmem_tensor_like(tXgX)
        tXrX.fill(0.0)
        self._load_rows(tXgX, tXrX, t0OcO, tOcO, seqlen_limit)
        x = tXrX.load().to(Float32)
        return cute.arch.fmax(
            x.reduce(cute.ReductionOp.MAX, init_val=0.0, reduction_profile=0),
            -(x.reduce(cute.ReductionOp.MIN, init_val=0.0, reduction_profile=0)),
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mPdPsum: cute.Tensor,
        mLSE: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdQaccum: cute.Tensor | None,
        mCuSeqlens: cute.Tensor | None,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        mStats: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        NWP = self.num_threads // cute.arch.WARP_SIZE
        sRed = cutlass.memory.SmemAllocator().allocate_tensor(
            Float32, cute.make_layout((_N_MAXIMA, NWP), stride=(NWP, 1)), 16
        )

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx
        cute.arch.griddepcontrol_wait()

        if work_tile.is_valid_tile:
            seqlen = SeqlenInfo.create(batch_idx, mO.shape[0], mCuSeqlens, None, tile=self.tile_m)
            mO_cur, mdO_cur = [
                seqlen.offset_batch(mX, batch_idx, dim=3)[None, None, head_idx] for mX in (mO, mdO)
            ]
            mPdPsum_cur = seqlen.offset_batch(mPdPsum, batch_idx, dim=2, padded=True)[
                None, head_idx
            ]
            seqlen_limit = seqlen.seqlen - m_block * self.tile_m

            gLSE = cute.local_tile(
                seqlen.offset_batch(mLSE, batch_idx, dim=2)[None, head_idx],
                (self.tile_m,),
                (m_block,),
            )
            lse = Float32.inf
            if tidx < seqlen_limit:
                lse = gLSE[tidx]

            blk_shape = (self.tile_m, self.head_dim)
            gO = cute.local_tile(mO_cur, blk_shape, (m_block, 0))
            gdO = cute.local_tile(mdO_cur, blk_shape, (m_block, 0))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOgO = gmem_thr_copy_O.partition_S(gO)
            tOgdO = gmem_thr_copy_O.partition_S(gdO)
            cO = cute.make_identity_tensor(blk_shape)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cO)
            tOrO = cute.make_rmem_tensor_like(tOgO)
            tOrdO = cute.make_rmem_tensor_like(tOgdO)

            self._load_rows(tOgO, tOrO, t0OcO, tOcO, seqlen_limit)
            self._load_rows(tOgdO, tOrdO, t0OcO, tOcO, seqlen_limit)
            # the main kernel's griddepcontrol_wait covers everything written below
            cute.arch.griddepcontrol_launch_dependents()
            threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]

            def row_reduce(x, op, init):
                v = x.reduce(op, init_val=init, reduction_profile=(0, None, 1))
                if const_expr(op == cute.ReductionOp.ADD):
                    # a row's sum spans the threads sharing it; a maximum
                    # need not, since the block's is taken below
                    v = utils.warp_reduce(v, operator.add, width=threads_per_row)
                out = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
                out.store(v)
                return out

            o, do = tOrO.load().to(Float32), tOrdO.load().to(Float32)
            PdP_sum = row_reduce(o * do, cute.ReductionOp.ADD, 0.0)
            dO_sq = row_reduce(do * do, cute.ReductionOp.ADD, 0.0)
            dn = Float32(0.0)
            dl = Float32(0.0)
            for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                if tOcO[0, m, 0][0] < seqlen_limit:
                    dn = cute.arch.fmax(dn, cute.math.sqrt(dO_sq[m]))
                    dl = cute.arch.fmax(dl, cute.arch.fmax(PdP_sum[m], -PdP_sum[m]))
            dc = Float32(0.0)
            if const_expr(self.record_stats):
                # the largest element as well as the largest row norm: dV's
                # bound is per output component
                dO_sqmax = row_reduce(do * do, cute.ReductionOp.MAX, 0.0)
                for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                    if tOcO[0, m, 0][0] < seqlen_limit:
                        dc = cute.arch.fmax(dc, cute.math.sqrt(dO_sqmax[m]))

            # max |K| and |V| once per KV head, over this block's rows
            kx = Float32(0.0)
            vx = Float32(0.0)
            qx = Float32(0.0)
            head_kv = head_idx // self.kv_group
            absargs = (seqlen, batch_idx, m_block, gmem_thr_copy_O, t0OcO, tOcO, seqlen_limit)
            if head_idx % self.kv_group == 0:
                kx = self._block_absmax(mK, head_kv, *absargs)
                vx = self._block_absmax(mV, head_kv, *absargs)
            if const_expr(self.record_stats):
                qx = self._block_absmax(mQ, head_idx, *absargs)
            vals = [None] * _N_MAXIMA
            for i, x in (
                (ST_K, kx),
                (ST_V, vx),
                (ST_DO, dn),
                (ST_DELTA, dl),
                (ST_DOC, dc),
                (ST_Q, qx),
            ):
                vals[i] = x
            warp = tidx // cute.arch.WARP_SIZE
            for i in cutlass.range_constexpr(_N_MAXIMA):
                wmax = utils.warp_reduce(vals[i], cute.arch.fmax)
                if cute.arch.lane_idx() == 0:
                    sRed[i, warp] = wmax
            cute.arch.sync_threads()
            if tidx < _N_MAXIMA:
                r = sRed[tidx, 0]
                for wi in cutlass.range_constexpr(1, NWP):
                    r = cute.arch.fmax(r, sRed[tidx, wi])
                # max over non-negative floats is the max over their bits, and
                # a max lands the same in any order
                stat_ptr = mStats.iterator + (
                    (batch_idx * mStats.shape[1] + head_kv) * mStats.shape[2] + tidx
                )
                cute.arch.red(
                    stat_ptr, r.bitcast(Int32), op="max", dtype="u32", sem="relaxed", scope="gpu"
                )

            gPdPsum = cute.local_tile(mPdPsum_cur, (self.tile_m,), (m_block,))
            # the thread holding column 0 of a row writes its Delta
            if tOcO[0, 0, 0][1] == 0:
                for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                    row = tOcO[0, m, 0][0]
                    val = Float32(0.0)
                    if row < seqlen_limit:
                        val = PdP_sum[m]
                    gPdPsum[row] = val

            mdQaccum_cur = seqlen.offset_batch(
                mdQaccum, batch_idx, dim=2, padded=True, multiple=self.head_dim
            )[None, head_idx]
            gdQaccum = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Int32,
                    (mdQaccum_cur.iterator + m_block * self.tile_m * self.head_dim).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(self.zero_elems),
            )
            tdQgdQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx).partition_S(gdQaccum)
            zero = cute.make_rmem_tensor_like(tdQgdQaccum)
            zero.fill(0)
            cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)

            # The forward's LSE is in nats and the mainloop's `exp2` wants base 2.
            lse_log2 = lse * math.log2(math.e) if lse != -Float32.inf else 0.0
            gLSElog2 = cute.local_tile(
                seqlen.offset_batch(mLSElog2, batch_idx, dim=2, padded=True)[None, head_idx],
                (self.tile_m,),
                (m_block,),
            )
            if tidx < cute.round_up(seqlen.seqlen, self.tile_m) - m_block * self.tile_m:
                gLSElog2[tidx] = lse_log2
