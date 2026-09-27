"""Host-side helpers shared by the backward and the decode."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cuda.bindings.runtime as cudart
import torch

# Every decode weight is `exp2(s - Z)`, so Q is scaled by `LOG2E / sqrt(D)`
# before it is quantised. Nothing in a kernel can detect an unscaled Q.
LOG2E = 1.4426950408889634

_ROTATIONS: dict = {}
_COMPILED: dict = {}


@dataclass(eq=False)
class Launch:
    """A prepared launch: calling it calls `run`. The kernels take the bound
    tensors by address, so `held` keeps them alive as long as the launch."""

    run: Callable
    held: tuple = ()

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


def compile_cached(key, build: Callable):
    """`build()`'s compiled kernel, built once per `key`. Keys start with the
    name of what they compile, so the kernels share one cache."""
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = build()
    return fn


def rotation(d, device=None, dtype=torch.float32):
    """The orthonormal Sylvester Hadamard matrix of order `d`.

    `q.k == (Rq).(Rk)` for any orthogonal R, so rotating the cache changes no
    logit. It spreads a model's outlier channels over all of them, so one
    integer scale per row fits the rms rather than the largest channel.
    """
    key = (d, str(device), str(dtype))
    if key not in _ROTATIONS:
        if d & (d - 1):
            raise ValueError(f"a Sylvester Hadamard needs a power of two, got {d}")
        h = torch.ones(1, 1, device=device, dtype=dtype)
        while h.shape[0] < d:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _ROTATIONS[key] = h * (1.0 / math.sqrt(d))
    return _ROTATIONS[key]


def hadamard(x):
    """`x @ rotation(d).T` over the last axis, in fp32, as a butterfly.

    A matmul rounds differently depending on how many rows it is given, so a
    key rotated alone and inside a batch would quantise to different planes.
    Each butterfly stage adds a fixed pair, so a row's bits are its own, and
    the device writers run the same stages in the same order.
    """
    d = x.shape[-1]
    if d & (d - 1):
        raise ValueError(f"a Sylvester Hadamard needs a power of two, got {d}")
    y = x.float().reshape(-1, d)
    h = 1
    while h < d:
        y = y.view(-1, d // (2 * h), 2, h)
        a, b = y[:, :, 0], y[:, :, 1]
        y = torch.stack((a + b, a - b), 2)
        h *= 2
    return (y.reshape(x.shape) * (1.0 / math.sqrt(d))).contiguous()


def full_carveout(compiled):
    """Give every kernel of a `cute.compile` result the whole L1 as shared
    memory on the current device, and return it.

    The driver sizes each kernel's carveout from its own occupancy, and an SM
    runs CTAs of one carveout at a time, so a programmatic dependent lands
    only on the SMs its predecessor left empty. `launch(preferred_smem_carveout=)`
    does not reach a kernel loaded through the DSL's CUDA library path, so the
    attribute is set on the loaded kernels here.
    """
    lib = compiled.library
    err, n = cudart.cudaLibraryGetKernelCount(lib)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaLibraryGetKernelCount: {err}")
    err, kernels = cudart.cudaLibraryEnumerateKernels(n, lib)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaLibraryEnumerateKernels: {err}")
    dev = cuda.CUdevice(torch.cuda.current_device())
    attr = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    for k in kernels:
        (err,) = cuda.cuKernelSetAttribute(attr, 100, cuda.CUkernel(int(k)), dev)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuKernelSetAttribute: {err}")
    return compiled


_STREAMS: dict = {}
_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def current_stream(device) -> cuda.CUstream:
    """The current torch stream as a `CUstream`, cached on its raw handle.

    `torch.cuda.current_stream().cuda_stream` costs about 3 us a call and the
    raw getter 0.05 us, which matters at the smallest decode batch. Keying
    the cache on the handle keeps a stream switch correct.
    """
    ix = device.index if device.index is not None else torch.cuda.current_device()
    raw = _RAW_STREAM(ix) if _RAW_STREAM is not None else torch.cuda.current_stream(ix).cuda_stream
    stream = _STREAMS.get(raw)
    if stream is None:
        stream = _STREAMS[raw] = cuda.CUstream(raw)
    return stream
