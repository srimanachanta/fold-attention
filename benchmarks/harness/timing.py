"""CUDA-event timing under a fixed cache and ordering contract.

Three effects bias a kernel benchmark on a power-limited H100, and each has a
control here:

- The clock drifts. Every arm is timed once per round, the first rounds are
  dropped, and the statistic is the median over rounds. Ratios are taken per
  round (`paired_ratio`), so drift that moves both arms cancels.
- The previous kernel leaves clock, power and cache state behind. The order
  within a round advances by strides coprime to the arm count, so every arm
  follows every other arm over the run. A fixed order once put three
  identical backward arms at 1.133, 1.117 and 1.101 of FA-3.
- L2 is 50 MB and a short KV cache fits in it. `cold=True` (the default)
  evicts L2 before every sample and times one invocation, as FlashInfer's
  harness does. `cold=False` times back-to-back invocations, so the pair can
  be reported.

A sample is one CUDA graph replay where the arm can be captured, so the host's
launch path is not in the measurement. An arm that cannot be captured is
timed eagerly and reported as such.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from math import gcd

import torch

__all__ = ["Arm", "L2Flush", "measure", "paired_ratio"]


class L2Flush:
    """Evicts the device's L2 by reading a buffer several times its size."""

    def __init__(self, device=None, factor: int = 4, min_mib: int = 256):
        dev = device or torch.cuda.current_device()
        l2 = torch.cuda.get_device_properties(dev).L2_cache_size
        self.nbytes = max(int(l2) * factor, min_mib << 20)
        self.buf = torch.empty(self.nbytes // 4, dtype=torch.float32, device="cuda")
        self.buf.normal_()
        self.sink = torch.empty((), dtype=torch.float32, device="cuda")

    def __call__(self):
        torch.sum(self.buf, dim=0, out=self.sink)


def _graph(fn: Callable, warmup: int = 3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


class Arm:
    """One timed callable, replayed from a CUDA graph where it can be captured."""

    def __init__(self, fn: Callable, graphable: bool = True):
        self.fn = fn
        self.graph = None
        if graphable:
            try:
                self.graph = _graph(fn)
            except Exception:  # noqa: BLE001
                self.graph = None
        if self.graph is None:
            for _ in range(3):
                fn()
            torch.cuda.synchronize()

    @property
    def mode(self) -> str:
        return "graph" if self.graph is not None else "eager"

    def once(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self.fn()

    def timed(self, reps: int = 1) -> float:
        """Microseconds per invocation, averaged over `reps` back-to-back."""
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps):
            self.once()
        en.record()
        en.synchronize()
        return st.elapsed_time(en) * 1e3 / reps


def _orders(names, rounds):
    """Round orders whose strides are coprime to the arm count, so every arm
    follows every other arm across the run."""
    n = len(names)
    strides = [s for s in range(1, max(n, 2)) if gcd(s, n) == 1] or [1]
    out = []
    for r in range(rounds):
        off = r % n
        step = strides[(r // n) % len(strides)]
        out.append([names[(off + step * j) % n] for j in range(n)])
    return out


def measure(
    fns: dict,
    *,
    rounds: int | None = None,
    drop: int = 4,
    cold: bool = True,
    graphable: dict | None = None,
    target_us: float = 4000.0,
    flusher: L2Flush | None = None,
) -> dict:
    """`{name: callable}` to `{name: {us, iqr, lo, hi, n, samples, mode, cold, reps}}`.

    `rounds` defaults to `max(21, 2 * len(fns))`: the rotation cancels
    position bias only once every arm has held every slot.
    """
    graphable = graphable or {}
    arms = {n: Arm(f, graphable.get(n, True)) for n, f in fns.items()}
    names = list(arms)
    if rounds is None:
        rounds = max(21, 2 * len(names))
    if cold and flusher is None:
        flusher = L2Flush()

    reps = {}
    for n in names:
        t = arms[n].timed(3)
        reps[n] = 1 if cold else max(3, min(400, int(target_us / max(t, 1.0))))

    dev = {n: [] for n in names}
    for r, order in enumerate(_orders(names, rounds + drop)):
        for n in order:
            if cold:
                assert flusher is not None
                flusher()
            t = arms[n].timed(reps[n])
            if r >= drop:
                dev[n].append(t)

    out = {}
    for n in names:
        s = dev[n]
        q = statistics.quantiles(s, n=4) if len(s) >= 4 else [min(s), s[0], max(s)]
        out[n] = dict(
            us=statistics.median(s),
            iqr=q[2] - q[0],
            lo=min(s),
            hi=max(s),
            n=len(s),
            samples=s,
            mode=arms[n].mode,
            cold=bool(cold),
            reps=reps[n],
        )
    return out


def paired_ratio(results: dict, numerator: str, denominator: str) -> dict:
    """Median and IQR of the per-round ratio `numerator / denominator`.

    Both arms ran in the same round under the same clock, so the per-round
    ratio cancels drift that a ratio of medians keeps.
    """
    return ratio_of(results[numerator]["samples"], results[denominator]["samples"])


def ratio_of(a, b) -> dict:
    rs = [x / y for x, y in zip(a, b, strict=True)]
    q = statistics.quantiles(rs, n=4) if len(rs) >= 4 else [min(rs), rs[0], max(rs)]
    return dict(ratio=statistics.median(rs), iqr=q[2] - q[0], lo=min(rs), hi=max(rs), n=len(rs))
