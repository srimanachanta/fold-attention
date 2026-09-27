"""Post-RoPE Q/K/V captured from real models, and decode shapes cut from them.

Random normal operands understate attention's difficulty: real post-RoPE keys
carry outlier channels and real score rows are far more peaked, and both
change what truncation keeps and what a quantised cache costs. Every decode
accuracy number in the suite comes from a capture.

A capture is a `torch.save` dict with `q` `(1, H, S, D)` and `k`, `v`
`(1, H_KV, S, D)`, one layer over one sequence. The files live outside the
repository; `EFA_CAPTURE_DIR` names their directory. A result file records
each capture's path, size and a hash of its ends.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

CAPDIR = Path(os.environ.get("EFA_CAPTURE_DIR", Path.home() / "efa_captures"))

# name -> file; the name is model, layer and head dim
FILES = {
    "qwen3-30b-l24-d128": Path(os.environ.get("EFA_CAPTURE", CAPDIR / "c2d7013b70c34823.pt")),
    "gptoss-20b-l9-d64": CAPDIR / "e012ec57761bcf01.pt",
    "gptoss-20b-l21-d64": CAPDIR / "5b384d4597d01cea.pt",
    "glm4-9b-l28-d128": CAPDIR / "6d4ca1a2a0e29c58.pt",
    "glm4-9b-l8-d128": CAPDIR / "8e74c9b6689968d6.pt",
}

_CACHE: dict = {}
_HASH: dict = {}


def fingerprint(name: str) -> dict:
    """Path, size and a SHA-256 of the first and last MiB."""
    p = FILES[name]
    if name not in _HASH:
        h = hashlib.sha256()
        size = p.stat().st_size
        with p.open("rb") as f:
            h.update(f.read(1 << 20))
            f.seek(max(0, size - (1 << 20)))
            h.update(f.read(1 << 20))
        _HASH[name] = dict(path=str(p), bytes=size, sha256_ends=h.hexdigest())
    return _HASH[name]


def raw(name: str):
    """`(q, k, v)` on the CPU: `q` is `(H, S, D)`, `k` and `v` `(H_KV, S, D)`."""
    if name not in _CACHE:
        d = torch.load(FILES[name], map_location="cpu", weights_only=False)
        _CACHE[name] = (d["q"][0], d["k"][0], d["v"][0])
    return _CACHE[name]


def heads(name: str) -> tuple[int, int, int, int]:
    """`(H, H_KV, S, D)` of a capture."""
    q, k, _ = raw(name)
    return q.shape[0], k.shape[0], q.shape[1], q.shape[2]


def decode_shape(name, B, S, G=None, HKV=None, lens=None, device="cuda", seed=0):
    """A `(B, H, D)` query and a `(B, H_KV, S, D)` cache cut from a capture.

    Heads are taken, never repeated, when the capture has enough of them: a
    repeated query row makes a group self-similar, which favours any method
    that shares work across the group. When the request needs more heads than
    the capture has, rows are repeated and `proxy` says so.

    Request `b` is a decode step of `lens[b]` keys (default `S`) at its own
    position `p` of the captured sequence: the sink (token 0) and the
    `lens[b] - 1` most recent tokens through `p`, whose query it is, so the
    step attends its own key as a decode does. The sink holds a large share of
    a real row's mass and every serving cache keeps it; a window without it
    understates what a truncation drops. A request as long as the capture is
    the whole sequence, so at that length requests differ only by head.
    """
    q_all, k_all, v_all = raw(name)
    H_cap, S_cap, D = q_all.shape
    HKV_cap = k_all.shape[0]
    G_cap = H_cap // HKV_cap
    HKV = HKV or HKV_cap
    G = G or G_cap
    if S > S_cap:
        raise ValueError(f"{name} has {S_cap} keys, asked for {S}")
    n = [S] * B if lens is None else [int(x) for x in lens]
    if len(n) != B or min(n) < 2 or max(n) > S:
        raise ValueError(f"lens must be {B} lengths in [2, {S}]")
    g = torch.Generator().manual_seed(seed)
    proxy = []
    kv_pick = torch.arange(HKV) % HKV_cap
    if HKV > HKV_cap:
        proxy.append(f"kv heads repeated {HKV}/{HKV_cap}")
    q_pick = None
    if G > G_cap:
        q_pick = kv_pick.repeat_interleave(G) * G_cap + (torch.arange(HKV * G) % G_cap)
        proxy.append(f"query rows repeated {G}/{G_cap}")

    k = torch.zeros(B, HKV, S, D, dtype=torch.bfloat16)
    v = torch.zeros(B, HKV, S, D, dtype=torch.bfloat16)
    q = torch.empty(B, HKV, G, D, dtype=torch.bfloat16)
    u = torch.rand(B, generator=g)
    for b in range(B):
        nb = n[b]
        # the query's position, late enough for its window to follow the sink
        p = nb - 1 + int(u[b] * (S_cap - nb + 1))
        rows = torch.cat([torch.zeros(1, dtype=torch.long), torch.arange(p - nb + 2, p + 1)])
        k[b, :, :nb] = k_all[kv_pick[:, None], rows[None, :]].to(torch.bfloat16)
        v[b, :, :nb] = v_all[kv_pick[:, None], rows[None, :]].to(torch.bfloat16)
        if q_pick is None:
            for h in range(HKV):
                c = int(kv_pick[h]) * G_cap
                q[b, h] = q_all[c : c + G, p].to(torch.bfloat16)
        else:
            q[b] = q_all[q_pick, p].reshape(HKV, G, D).to(torch.bfloat16)
    dev = torch.device(device)
    return dict(
        q=q.reshape(B, HKV * G, D).to(dev),
        k=k.to(dev),
        v=v.to(dev),
        B=B,
        S=S,
        D=D,
        G=G,
        HKV=HKV,
        H=HKV * G,
        capture=name,
        proxy="; ".join(proxy),
        fingerprint=fingerprint(name),
    )


def ragged_lens(B, S, spread=0.5, device="cuda", seed=0, page=128):
    """Page-aligned request lengths in `[S (1 - spread), S]`, one of them `S`."""
    g = torch.Generator().manual_seed(seed)
    lo = max(page, int(S * (1.0 - spread)))
    n = torch.randint(lo // page, S // page + 1, (B,), generator=g) * page
    n[0] = S
    return n.to(torch.int32).to(device)
