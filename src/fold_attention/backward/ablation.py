"""The backward with one mechanism changed, for measuring what it buys.

Each variant is the shipped kernel (`prepare_backward`) with one choice
replaced and the rest as it is:

- `fp32_dq=True`: dQ's partials summed as fp32 by `cp.reduce.async.bulk
  .add.f32` in the order they land, as FlashAttention sums them. It skips the
  rounding and is not deterministic. P then carries no grid exponent, whose
  subtraction rounds the softmax's argument, so dK and dV move in their last
  bits too.
- `persistent=False`: a dense call runs one CTA per work tile under
  `SingleTileScheduler` instead of the persistent work list, with no
  canonical dK/dV records.
"""

from __future__ import annotations

from .launch import BackwardLaunch, _prepare


def prepare_backward_variant(
    q, k, v, o, do, lse, *, causal, fp32_dq=False, persistent=True, **kwargs
) -> BackwardLaunch:
    """`prepare_backward` with dQ summed as fp32 (`fp32_dq`) and the dense
    scheduler `persistent`; the other keyword arguments are
    `prepare_backward`'s. The defaults are the shipped kernel."""
    return _prepare(
        q, k, v, o, do, lse, causal=causal, dq_fp32=fp32_dq, persistent=persistent, **kwargs
    )
