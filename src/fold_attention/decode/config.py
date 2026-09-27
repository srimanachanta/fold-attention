"""What one decode build is: its compile-time configuration and footprint."""

from __future__ import annotations

from dataclasses import dataclass

# One warpgroup over a 64-key tile: `wgmma`'s M is 64 and its C layout gives
# warp `w` keys [16w, 16w + 16), which every per-key register in the tile loop
# is indexed by.
BN = 64
NT = 128
# The packed kernel's tile (`decode.packed`): two keys in each of the logit's
# 64 rows.
TK = 2 * BN

# The wide kernel's row tile a consumer warpgroup, its consumer warpgroups,
# and the producer warpgroup's registers a thread: it holds a tile's
# addresses and converts a quarter of its K rows.
MR = 64
NWG = 2
PRODUCER_REGS = 64
# The wide kernel's fp16 K and V tiles in flight: even, so that under a key
# split a stage always serves one warpgroup and no waiter runs a phase ahead
# of its parity. Then K's planes in flight, a tile ahead of their conversion.
WIDE_STAGES = 4
WIDE_RAW_STAGES = 3

# The mass reference's stratum samples and how many of the largest it trims.
MASS_STRATA = 128
MASS_TRIM = 4


@dataclass(frozen=True)
class DecodeConfig:
    """Every compile-time choice of one decode build: the kernel's one
    `Constexpr` and its compile-cache key."""

    n_groups: int  # row groups (request x KV head) in the grid
    group: int  # query rows per row group
    head_dim: int
    split: int  # CTAs per row group
    truncate: bool  # gather only keys some row keeps above its cut
    v8: bool  # V is two e4m3 planes rather than bf16
    v_regs: bool  # the value matmul's A comes from registers (8-bit V only)
    dropped_mass: bool  # re-enter the truncated mass at V's block mean
    row_vmean: bool  # the dropped-mass direction is per query row
    paged: bool
    page_size: int
    kv_heads: int  # KV heads per request under a page table
    z_prepass: bool  # Z and the cut are written by an earlier kernel (the prepass or the front)
    keep_all: bool  # every gathered key enters every row at its true weight
    draft: int  # draft positions per query head under a mask; 0 for none
    slots: int  # partial slots a row group owns across cascade levels
    slot0: int  # the first slot this level writes
    unique_group: int  # the unique level's G, the Q row stride under `shared`
    shared: bool  # this build is a cascade's shared level
    direct: bool  # the single split normalises without a combine launch
    min_blocks: int  # `min_blocks_per_mp`, which caps registers; 0 for none
    tail_blocks: bool = False  # the dropped mass re-enters at its block's V row
    tail_rank: int = 0  # rank of the part of a dropped key's V its K predicts
    weight_terms: int = 1  # bf16 weights as one term, or two: value and rounding error
    front: bool = False  # Z, Q and the new token come from `front.mass_front`
    front_qk: int = 0  # 1: the mass CTA publishes Q and K; 2: mass Q, append K/V
    front_clear: bool = True  # clear the front's words after their last reader
    sound: bool = False  # shift the cut by plane A's own residual bound
    order: bool = False  # grid slots are row groups permuted longest-first
    ek_stride: int = 0  # a contiguous cache's key-scale row stride; 0 when paged
    # (edge0, edge1, rk1, rk2, rv1, rv2): dense gates picked by the row group's
    # own length; empty for the launch's RK/RV alone
    refine_bands: tuple = ()
    # keys per split, fixed, so a request's partials depend on its own length
    # alone; 0 splits each request into `split` equal chunks
    chunk_keys: int = 0
    # the splits' rank sums go to the combine, which expands them once per row
    rank_combine: bool = False
    # shared bytes past the kernel's own, to hold an SM to the grid's share of CTAs
    smem_pad: int = 0
    # keys in each row of the logit's M: 2 is `decode.packed`'s 128-key tile
    pack: int = 1


def mass_tile(draft: int = 0) -> tuple[int, int]:
    """`(rows, window)` of the mass reference's tile: the sink, the most
    recent `window` keys and the strata, in whole 64-row tiles. The window
    takes whatever the tiles leave, at least 31 keys and every draft key,
    which a row sees only through its mask."""
    rows = -(-(1 + MASS_STRATA + max(31, draft)) // BN) * BN
    return rows, rows - 1 - MASS_STRATA


def pack_for(D: int, G: int, v8: bool) -> int:
    """Keys a row of the logit's M holds: 2, the packed kernel, for a group of
    at most four rows on a bf16 V at D = 64. At D = 128 a tile twice as wide
    halves the CTAs a SM, and the loop gains nothing. A cascade's shared
    levels hold one."""
    return 2 if not v8 and G <= 4 and D == 64 else 1


def weights_in_plane_b(D: int, NG: int, w2: bool, v8: bool, pack: int = 1) -> bool:
    """Whether the weight buffer lives in plane B's tile: on a bf16 V, whose
    tile holds no second V plane, where it fits and keeps the tile's 1 KB
    alignment."""
    n = BN * pack
    return not v8 and NG * 8 * n * 2 * (2 if w2 else 1) <= n * D and (n * D) % 1024 == 0


def smem_bytes(
    D: int,
    G: int,
    v8: bool,
    v_regs: bool,
    tail_rank: int = -1,
    weight_terms: int = 1,
    sound: bool = False,
    pack: int | None = None,
) -> int:
    """Shared memory one CTA holds, mirroring the kernel's allocator so the
    split rules can use it before anything is compiled. `tail_rank` is -1
    for no tail, `pack` the kernel (`pack_for`'s by default). It leaves out
    the front's ready word (4 B); the driver's own 1 KB per-block reservation
    is added by the callers."""
    NG = (G + 7) // 8
    NQ = NG * 8
    NW = NT // 32
    reg_v = v_regs and v8
    if (pack_for(D, G, v8) if pack is None else pack) == 2:
        # plane A, plane B (the weights inside), V; the key scales double
        # buffered; one Q block of both planes at G <= 2, else two; U, zeros, U
        by = 4 * TK * D + 2 * TK * 2 + (8 if G <= 2 else 16) * 2 * D + 3 * max(tail_rank, 0) * D
        if not weights_in_plane_b(D, 1, weight_terms == 2, False, 2):
            by += NQ * TK * 2 * weight_terms
        # four query rows' Z, cut, scale (and bound), draft masks and sums;
        # the warps' counts; one page segment; the barrier
        by += (5 if sound else 4) * 4 * 4 + (2 * NW + 1) * 4 * 4 + 2 * NW * 4 + 16 + 8
        return by
    by = 2 * BN * D  # K planes A and B
    by += 2 * BN * 2  # the tiles' key scales, double buffered
    by += BN * D if reg_v else 2 * BN * D  # the V tile, e4m3 or 16-bit
    if v8 and not reg_v:
        by += BN * D  # the e4m3 staging tile
    if not weights_in_plane_b(D, NG, weight_terms == 2, v8):
        by += NQ * BN * 2 * weight_terms  # the weight buffer, and its residual's
    by += (2 * NQ + max(tail_rank, 0)) * D  # both Q planes, then the tail's U
    by += (5 if sound else 4) * NQ * 4  # Z, cut, scale, bound; draft masks
    by += (2 * NW + 1) * NQ * 4 + BN * 4 + 8 + 8
    return by


def wide_slots_per_split(group: int) -> int:
    """Partial slots one split of a wide level writes: a group one warpgroup
    holds is split between both warpgroups by tile, a slot each."""
    return 2 if group <= MR else 1


@dataclass(frozen=True)
class WideConfig:
    """Every compile-time choice of one wide build (`decode.wide`): the
    kernel's one `Constexpr` and its compile-cache key."""

    n_groups: int  # row groups (request x KV head, or a cascade range x KV head)
    group: int  # query rows per row group
    head_dim: int
    split: int
    truncate: bool  # rows keep only keys above their cut
    dropped_mass: bool  # the cut mass re-enters at V's mean
    row_vmean: bool
    paged: bool
    page_size: int
    kv_heads: int
    slots: int  # partial slots a unique row group owns
    slot0: int  # the first slot this level writes
    unique_group: int  # the unique level's rows per row group
    weight_terms: int
    ek_stride: int  # a contiguous cache's key-scale row stride; 0 when paged
    image: bool = False  # K and V arrive as `prefix_image` tiles: no conversion, no gathers
    shared: bool = True  # rows are named by a cascade's row map, else by the row group
    draft: int = 0  # draft positions under a mask; 0 for none
    direct: bool = False  # one split normalises its own rows

    @property
    def key_split(self) -> bool:
        # a group one warpgroup holds: the two take alternate tiles of it,
        # each into a partial slot of its own
        return self.group <= MR

    @property
    def row_tiles(self) -> int:
        return -(-self.group // (MR * (1 if self.key_split else NWG)))

    @property
    def slots_per_split(self) -> int:
        return wide_slots_per_split(self.group)

    @property
    def threads(self) -> int:
        # the consumer warpgroups and a producer warpgroup
        return 128 * NWG + 128

    @property
    def consumer_regs(self) -> int:
        # What the producer gives back, over the consumers. A CTA's pool is
        # its launch count a thread, which ptxas caps to fit one CTA an SM, and
        # a request past the pool spins in `setmaxnreg` for ever.
        launch = 65536 // self.threads // 8 * 8
        pool = launch * self.threads
        return (pool - PRODUCER_REGS * 128) // (128 * NWG) // 8 * 8
