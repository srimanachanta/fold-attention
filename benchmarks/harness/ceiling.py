"""The device's achievable read bandwidth, the yardstick for a decode's bytes.

A decode at long context reads its cache and little else, so its speed is
best stated as a fraction of what the memory system delivers. A `torch.sum`
reduction under-reports that (four attention kernels beat it on H100), so the
ceiling is a wide, unrolled linear scan with eight blocks per SM.

A gather delivers less than a linear scan, depending on its run length, so a
kernel that gathers rows is not expected to reach this number.
"""

from __future__ import annotations

import statistics

import torch


def _time(fn, rounds=9, drop=3, rep=4):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds + drop):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        fn()
        st.record()
        for _ in range(rep):
            fn()
        en.record()
        en.synchronize()
        ts.append(st.elapsed_time(en) * 1e3 / rep)
    return statistics.median(ts[drop:])


def _triton_scan():
    import triton
    import triton.language as tl

    @triton.jit
    def _scan(SRC, OUT, N, BLOCK: tl.constexpr, STEPS: tl.constexpr):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
        base = pid * BLOCK
        stride = nprog * BLOCK
        for _ in range(STEPS):
            o = base + tl.arange(0, BLOCK)
            acc += tl.load(SRC + o, mask=o < N, other=0)
            base += stride
        tl.store(OUT + pid, tl.sum(acc))

    return _scan


def read_ceiling(gib=2.0, block=4096, blocks_per_sm=8) -> dict:
    """`{gbs, us, bytes, method}` of a linear scan over `gib` GiB.

    Without triton it falls back to a torch reduction and says so in
    `method`: that number is a floor, not a ceiling.
    """
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    nprog = sms * blocks_per_sm
    try:
        scan = _triton_scan()
    except Exception:  # noqa: BLE001
        n = int(gib * 2**30) // 2
        big = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
        us = _time(lambda: big.sum(dtype=torch.float32))
        del big
        torch.cuda.empty_cache()
        nb = n * 2
        return dict(gbs=nb / (us * 1e-6) / 1e9, us=us, bytes=nb, method="torch reduction (floor)")

    span = block * nprog
    steps = max(1, int(gib * 2**30) // 4 // span)
    n = span * steps
    src = torch.empty(n, dtype=torch.int32, device="cuda")
    src.random_(-(2**30), 2**30)
    out = torch.empty(nprog, dtype=torch.int32, device="cuda")
    us = _time(lambda: scan[(nprog,)](src, out, n, BLOCK=block, STEPS=steps, num_warps=8))
    nb = n * 4
    return dict(gbs=nb / (us * 1e-6) / 1e9, us=us, bytes=nb, method=f"triton scan, {nprog} blocks")
