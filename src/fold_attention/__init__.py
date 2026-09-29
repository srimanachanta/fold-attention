"""FoldAttention: softmax attention whose cross-CTA reductions do not depend
on the order work runs in.

Training uses FlashAttention-4's forward and an SM90 backward whose dQ, dK and
dV sums are integer or fixed-order reductions, so gradients are the same bits
on every run and for every batching of a request. Serving decodes over a
quantised paged cache against a static per-row reference, so split-KV,
shared-prefix and speculative partials combine by plain addition.
"""

from .interface import fold_attn_func, fold_attn_varlen_func, fold_attn_with_kvcache
from .kv_cache import FoldKVCache

__version__ = "0.2.0"

__all__ = ["FoldKVCache", "fold_attn_func", "fold_attn_varlen_func", "fold_attn_with_kvcache"]
