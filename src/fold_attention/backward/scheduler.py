"""Persistent dispatch over the dense backward's host-built work list."""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
from cutlass import Int32, const_expr, cute
from flash_attn.cute.tile_scheduler import WorkTileInfo
from quack.cute_dsl_utils import ParamsBase


class ListScheduler:
    """Every CTA claims its next list entry from a counter.

    Greedy dispatch, the same balance the hardware gives one-tile CTAs, with
    the CTA's prologue paid once and the next tile's K/V load overlapping this
    tile's epilogue. Any assignment of tiles to CTAs gives the same bits, which
    is what lets a claim race decide it.

    One producer lane claims; the index reaches the MMA warpgroups and the dQ
    store warp through a two-stage shared-memory pipeline (`ctx`). The counter
    is `(2,)` int32, claims and finished CTAs, zero on entry; the last CTA to
    finish puts both back to zero, so graph replays need no memset.
    """

    @dataclass
    class Params(ParamsBase):
        total: Int32
        mList: cute.Tensor
        mCounter: cute.Tensor

    @classmethod
    def to_underlying_arguments(cls, mList, mCounter):
        return ListScheduler.Params(total=Int32(mList.shape[0]), mList=mList, mCounter=mCounter)

    def __init__(self, params, tile_idx, state, ctx, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._state = state
        self._ctx = ctx

    @classmethod
    @cute.jit
    def create(cls, params, ctx, *, loc=None, ip=None):
        _, _, is_producer = ctx
        kind = (
            cutlass.pipeline.PipelineUserType.Producer
            if is_producer
            else cutlass.pipeline.PipelineUserType.Consumer
        )
        state = cutlass.pipeline.make_pipeline_state(kind, 2)
        return cls(params, Int32(0), state, ctx, loc=loc, ip=ip)

    @classmethod
    def get_grid_shape(cls, params, *, loc=None, ip=None):
        sm_count = cutlass.utils.HardwareInfo().get_device_multiprocessor_count()
        return (cutlass.min(Int32(sm_count), params.total), Int32(1), Int32(1))

    @cute.jit
    def _coords(self, idx: Int32):
        i = cutlass.min(idx, self.params.total - 1)
        row = self.params.mList
        return row[i, 0], row[i, 1], row[i, 2], row[i, 3]

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None):
        return WorkTileInfo(self._coords(self._tile_idx), self._tile_idx < self.params.total)

    @cute.jit
    def _next(self):
        sWork, pipe, is_producer = self._ctx
        if const_expr(is_producer):
            idx = Int32(0)
            if cute.arch.lane_idx() == 0:
                idx = Int32(cute.arch.atomic_add(self.params.mCounter.iterator, Int32(1)))
            idx = cute.arch.shuffle_sync(idx, 0)
            pipe.producer_acquire(self._state)
            if cute.arch.lane_idx() == 0:
                sWork[self._state.index] = idx
            pipe.producer_commit(self._state)
            self._state.advance()
            self._tile_idx = idx
        else:
            pipe.consumer_wait(self._state)
            self._tile_idx = sWork[self._state.index]
            pipe.consumer_release(self._state)
            self._state.advance()

    def initial_work_tile_info(self, *, loc=None, ip=None):
        self._next()
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._next()
        return self.get_current_work()

    @cute.jit
    def producer_tail(self, *, loc=None, ip=None):
        """After the producer has claimed past the end: count this CTA out; the
        last one resets the counter for the next launch."""
        if cute.arch.lane_idx() == 0:
            claims = self.params.mCounter.iterator
            done = Int32(cute.arch.atomic_add(claims + 1, Int32(1)))
            if done == cute.arch.grid_dim()[0] - 1:
                cute.arch.store(claims, Int32(0), sem="release", scope="gpu")
                cute.arch.store(claims + 1, Int32(0), sem="release", scope="gpu")

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx, self._state]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx, self._state], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        params, tile_idx, state = obj_list
        return self.__class__(params, tile_idx, state, self._ctx)
