"""One result file per benchmark run, with the provenance to interpret it.

Every file carries the device, the library versions, the clock at the start
and end, the source revision, the command line, and each arm's provenance
string naming the call and configuration that was timed. Raw per-round samples
are kept beside the medians.

Nothing that identifies the machine or its users is written: no host name,
GPU UUID or device index, and no absolute path, which `anonymize` reduces to
its last component. `node_id` stands in for the host where two runs must be
told apart.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

SCHEMA = 4
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("EFA_BENCH_OUT", ROOT / "benchmarks" / "out"))
# an absolute path: a `/` that no word character or dot precedes, so ratios
# such as `dQ/dK` and `1/256` are left alone
_ABS_PATH = re.compile(r"(?<![\w.])(?:/[^\s'\"(),:;=\]]+)+")


def node_id() -> str:
    """A stable stand-in for the host name: distinct hosts get distinct ids,
    and the name itself is not recorded."""
    return hashlib.sha256(platform.node().encode()).hexdigest()[:12]


def anonymize(x):
    """`x` with every absolute path in its strings reduced to its last
    component, recursively through lists and dicts."""
    if isinstance(x, str):
        return _ABS_PATH.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1], x)
    if isinstance(x, dict):
        return {k: anonymize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [anonymize(v) for v in x]
    return x


def source_digest(root: Path | None = None) -> str:
    """sha256 over every `.py` under `src/` and `benchmarks/`, path and bytes.

    A `REVISION` stamp names the commit the tree was synced from, and a tree
    synced before its commit carries the previous one marked dirty. The digest
    names the code that actually ran: `python -m benchmarks.harness.report`
    prints it for a local checkout, so a result can be matched to a commit."""
    root = root or ROOT
    h = hashlib.sha256()
    for sub in ("src", "benchmarks"):
        for p in sorted((root / sub).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _revision():
    """The source revision. The GPU node receives the tree without `.git`, so
    `scripts/sync.sh` stamps `REVISION` before every push; git is the local
    fallback."""
    root = ROOT
    digest = source_digest(root)
    stamp = root / "REVISION"
    if stamp.exists():
        lines = stamp.read_text().split()
        return dict(
            commit=lines[0] if lines else "unknown",
            dirty="dirty" in lines,
            source="REVISION",
            source_sha256=digest,
        )
    try:
        c = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        d = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return dict(commit=c, dirty=bool(d), source="git", source_sha256=digest)
    except Exception:  # noqa: BLE001
        return dict(commit="unknown", dirty=None, source="none", source_sha256=digest)


def _script(path):
    """The script's path from the repository root, e.g. `benchmarks/decode.py`."""
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name


def _module_version(name):
    """A library's version, read without importing it: two FA-3 builds share
    one module name, and importing the wrong one first would shadow the other
    for the rest of the process."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def _lib_versions():
    out = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "cudnn": torch.backends.cudnn.version(),
    }
    for dist in (
        "flashinfer-python",
        "nvidia-cudnn-frontend",
        "flash-attn-3",
        "flash-attn-4",
        "nvidia-cutlass-dsl",
        "vllm",
        "sglang-kernel",
        "prefix_attn",
    ):
        out[dist] = _module_version(dist)
    return out


def device_facts(index: int = 0) -> dict:
    p = torch.cuda.get_device_properties(index)
    return dict(
        name=p.name,
        sm=f"{p.major}{p.minor}",
        sms=p.multi_processor_count,
        l2_bytes=p.L2_cache_size,
        mem_bytes=p.total_memory,
    )


def clock_sample(index: int = 0) -> dict:
    """SM clock, temperature, power and throttle reasons of this process's GPU."""
    try:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", str(index))
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    vis.split(",")[0],
                    (
                        "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,"
                        "clocks_throttle_reasons.active"
                    ),
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .split(", ")
        )
        return dict(
            sm_mhz=float(out[0]),
            sm_max_mhz=float(out[1]),
            temp_c=float(out[2]),
            power_w=float(out[3]),
            throttle=out[4],
        )
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e)[:120])


class Report:
    """Collects one run's rows and writes them to `OUT/<name>.json`.

    `flush()` rewrites the file after every row, so a run that dies partway
    keeps what it measured.
    """

    def __init__(self, name: str, **meta):
        self.name = name
        self.rows = []
        self.meta = dict(meta)
        self.started = time.time()
        self.env = dict(
            schema=SCHEMA,
            device=device_facts(),
            libs=_lib_versions(),
            revision=_revision(),
            argv=" ".join([_script(sys.argv[0]), *sys.argv[1:]]),
            node=node_id(),
            clock_start=clock_sample(),
        )

    def add(self, **row):
        self.rows.append(row)
        self.flush()
        return row

    def _doc(self):
        return anonymize(
            dict(
                experiment=self.name,
                meta=self.meta,
                env=dict(self.env, libs=_lib_versions()),
                clock_end=clock_sample(),
                wall_s=round(time.time() - self.started, 1),
                rows=self.rows,
            )
        )

    def flush(self) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{self.name}.json"
        path.write_text(json.dumps(self._doc(), indent=1, default=str))
        return path

    def write(self) -> Path:
        path = self.flush()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with (OUT / "MANIFEST.txt").open("a") as f:
            f.write(f"{digest}  {path.name}  {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        print(f"\nwrote {path}  ({len(self.rows)} rows, sha256 {digest[:16]})", flush=True)
        return path


if __name__ == "__main__":
    print(source_digest())
