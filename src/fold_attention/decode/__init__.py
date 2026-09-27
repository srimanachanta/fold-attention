"""Decode attention over a two-plane quantised KV cache under a static
reference.

`FoldKVCache` is the serving entry point. This package is the kernel-level
one: quantise a cache (`quantize_q`, `quantize_k`, `quantize_v`,
`pad_scales`, `paged_cache`, `paged_scale`, `write_kv`, `tail_model`), then run `fold_decode` with a given
or estimated reference, a draft tree (`draft_mask`, `pack_rows`) or a shared
prefix (`SharedPrefix`, `cascade_rows`, `prefix_image`).
"""

from .cache import (
    PrefixImage,
    Tail,
    pad_scales,
    paged_cache,
    paged_scale,
    prefix_image,
    quantize_k,
    quantize_kq,
    quantize_q,
    quantize_v,
    tail_model,
    write_kv,
)
from .heuristics import pick_config, pick_split
from .launch import SharedPrefix, capture_decode, fold_decode, prepare_fold_decode
from .rows import cascade_degree, cascade_rows, draft_mask, pack_rows, unpack_rows

__all__ = [
    "PrefixImage",
    "SharedPrefix",
    "Tail",
    "capture_decode",
    "cascade_degree",
    "cascade_rows",
    "draft_mask",
    "fold_decode",
    "pack_rows",
    "pad_scales",
    "paged_cache",
    "paged_scale",
    "pick_config",
    "pick_split",
    "prefix_image",
    "prepare_fold_decode",
    "quantize_k",
    "quantize_kq",
    "quantize_q",
    "quantize_v",
    "tail_model",
    "unpack_rows",
    "write_kv",
]
