"""Tables from the suite's result files, written to `out/RESULTS.md`.

Ratios are per-round paired medians from the result files (`timing.paired_ratio`),
not ratios of medians. A speedup is the baseline's time over ours, so above
1 means FoldAttention is faster.

    python -m benchmarks.summarize
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from benchmarks.harness.report import OUT


def load(stem):
    p = OUT / f"{stem}.json"
    return json.loads(p.read_text()) if p.exists() else None


def files(prefix):
    return sorted(p.stem for p in OUT.glob(f"{prefix}*.json"))


def geomean(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def f(x, fmt=".1f"):
    return "–" if x is None else format(x, fmt)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def env_section(docs):
    lines = ["## Environment\n"]
    seen = set()
    for name, d in docs:
        e = d["env"]
        key = (e["device"]["name"], json.dumps(e["libs"], sort_keys=True), e["revision"]["commit"])
        if key in seen:
            continue
        seen.add(key)
        libs = ", ".join(f"{k} {v}" for k, v in e["libs"].items() if v and not k.endswith("path"))
        rev = e["revision"]
        lines.append(
            f"- `{name}`: {e['device']['name']} ({e['device']['sms']} SMs), {libs}; "
            f"revision {rev['commit'][:12]}{' (dirty)' if rev.get('dirty') else ''}\n"
        )
    return "".join(lines) + "\n"


def _ours(arms, boundary="decode"):
    return [a for a in arms if a.get("ours") and a.get("boundary", "decode") == boundary]


def _budget(r, gate):
    """`band`: the largest BF16 baseline error; `strict`: the smallest."""
    errs = [t["err"] for t in r["tuned"].values() if not t["fp8"]]
    return max(errs) if gate == "band" else min(errs)


def _matched(r, gate="band"):
    """The fastest finite-depth member within the gate."""
    b = _budget(r, gate)
    ok = [
        a
        for a in _ours(r["arms"])
        if a.get("depth") is not None and a.get("cold") and a.get("err", a.get("err_final")) <= b
    ]
    return min(ok, key=lambda a: a["cold"]["us"]) if ok else None


def _matched_cell(m):
    if not m:
        return "none"
    name = f"T={m['depth']:g}" + (" v8" if m.get("v8") else "")
    return (
        f"{name} {f(m['cold']['us'])} / {f(m['err'] * 1e3, '.2f')}, live {f(m['live'], '.2f')} "
        f"({f(m['vs_fastest']['ratio'], '.2f')}x"
        + (f", {f(m['vs_fa4']['ratio'], '.2f')}x FA-4)" if m.get("vs_fa4") else ")")
    )


def _step_of(arms, a):
    if a is None:
        return None
    return next((s for s in arms if s["arm"] == f"{a['arm']} step" and s.get("cold")), None)


def decode_section(stem):
    d = load(stem)
    if not d:
        return ""
    ceiling = d["meta"].get("ceiling", {}).get("gbs")
    out = [
        f"### `{stem}`\n\n",
        (
            f"V {'8-bit' if d['meta'].get('v8') else 'bf16'}; depths {d['meta'].get('depths')}; "
            f"read ceiling {f(ceiling, '.0f')} GB/s. Band = the fastest finite depth at or "
            "below the largest BF16 baseline error; strict = at or below the smallest. "
            "Speedups are over the fastest BF16 baseline (and FA-4).\n\n"
        ),
    ]
    groups = defaultdict(list)
    for r in d["rows"]:
        c = r["cell"]
        groups[(c["D"], c["G"], c.get("row_groups"))].append(r)
    for (D, G, rg), rows in sorted(groups.items(), key=lambda kv: (-kv[0][0], -kv[0][1], kv[0][2])):
        out.append(f"**D={D} G={G}, {rg} row groups**\n\n")
        body = []
        for r in sorted(rows, key=lambda r: (r["cell"]["ragged"], r["cell"]["S"])):
            c = r["cell"]
            kind = "ragged" if c["ragged"] else "uniform"
            if "error" in r:
                body.append([c["S"], kind, f"error: {r['error'][:60]}"] + [""] * 8)
                continue
            arms = r["arms"]
            by = {a["arm"]: a for a in arms}
            fb = by[r["fastest_baseline"]]
            fa4 = next(
                (a for a in arms if a.get("family") == "FA-4" and a["boundary"] == "decode"), None
            )
            dense = next(a for a in _ours(arms) if a.get("depth") is None)
            m, mst = _matched(r, "band"), _matched(r, "strict")
            fp8 = [a for a in arms if a.get("fp8") and a.get("cold")]
            b8 = min(fp8, key=lambda a: a["cold"]["us"]) if fp8 else None
            ds, ms = _step_of(arms, dense), _step_of(arms, m)
            body.append(
                [
                    c["S"],
                    kind,
                    f"{fb['arm']} {f(fb['cold']['us'])} / {f(fb['err'] * 1e3, '.2f')}",
                    f"{f(fa4['cold']['us'])} / {f(fa4['err'] * 1e3, '.2f')}" if fa4 else "–",
                    f"{f(dense['cold']['us'])} / {f(dense['err'] * 1e3, '.2f')}",
                    f"{f(dense['vs_fastest']['ratio'], '.2f')}x"
                    + (f" ({f(dense['vs_fa4']['ratio'], '.2f')}x)" if dense.get("vs_fa4") else ""),
                    _matched_cell(m),
                    _matched_cell(mst),
                    f"{f(ds['cold']['us']) if ds else '–'} / {f(ms['cold']['us']) if ms else '–'}",
                    f"{f(dense.get('pct_ceiling'), '.0f')}%",
                    f"{b8['arm']} {f(b8['cold']['us'])} / {f(b8['err'] * 1e3, '.1f')}"
                    if b8
                    else "–",
                ]
            )
        out.append(
            table(
                [
                    "S",
                    "batch",
                    "fastest BF16 us / err e-3",
                    "FA-4 us / err",
                    "dense us / err",
                    "dense speedup",
                    "band matched (speedup)",
                    "strict matched (speedup)",
                    "step us dense / band",
                    "dense % ceiling",
                    "fastest FP8 us / err e-3",
                ],
                body,
            )
            + "\n"
        )
    return "".join(out)


FIXED = (
    "Fold dense",
    "Fold dense v8",
    "Fold T=16",
    "Fold T=14",
    "Fold capacity",
    "Fold capacity v8",
)


def _gate(err, lo, hi):
    return "S" if err <= lo else "B" if err <= hi else "over"


def _fixed_cell(by, name, lo, hi, errkey="err"):
    a = by.get(f"{name} step")
    d = by.get(name)
    if not a or not d or not a.get("vs_fastest_step"):
        return "–", None
    e = d.get(errkey, d.get("err"))
    r = a["vs_fastest_step"]["ratio"]
    return f"{f(r, '.2f')}x / {f(e * 1e3, '.2f')} {_gate(e, lo, hi)}", r


def headline_section():
    """Fixed members at the step boundary: each member's step (front, decode,
    combine) against the fastest BF16 baseline's step (its fastest append
    ahead of its tuned decode). S: error at or below the most accurate BF16
    baseline's; B: within the BF16 baselines' range."""
    out = [
        "## Headline: fixed members at the step boundary\n\n",
        (headline_section.__doc__ or "").split("\n\n")[0].replace("\n", " ").strip() + "\n\n",
    ]
    rng = defaultdict(list)
    d = load("generate")
    if d:
        body = []
        for r in d["rows"]:
            if "error" in r:
                continue
            by = {a["arm"]: a for a in r["arms"]}
            errs = [t["err"] for t in r["tuned"].values() if not t["fp8"]]
            lo, hi = min(errs), max(errs)
            fbs = by.get(r.get("fastest_baseline_step"))
            cells = []
            for n in FIXED:
                c, x = _fixed_cell(by, n, lo, hi, "err_final")
                cells.append(c)
                if x:
                    e = by[n]["err_final"]
                    rng[(n, _gate(e, lo, hi))].append(x)
                    rng[(n, "all")].append(x)
            body.append(
                [
                    r["case"],
                    f"{fbs['arm'] if fbs else '–'} {f(fbs['cold']['us']) if fbs else ''}",
                    f"{f(lo * 1e3, '.2f')}-{f(hi * 1e3, '.2f')}",
                    *cells,
                ]
            )
        out.append("### Generations (`generate`)\n\n")
        out.append(table(["case", "fastest BF16 step us", "BF16 err e-3", *FIXED], body) + "\n")
        lines = []
        for n in FIXED:
            xs = rng.get((n, "all"))
            if not xs:
                continue
            nb = len(rng.get((n, "S"), [])) + len(rng.get((n, "B"), []))
            lines.append(
                f"- {n}: {f(min(xs), '.2f')}-{f(max(xs), '.2f')}x (geomean "
                f"{f(geomean(xs), '.2f')}x); strict {len(rng.get((n, 'S'), []))}/{len(xs)}, "
                f"within band {nb}/{len(xs)}\n"
            )
        out.append("".join(lines) + "\n")
    for stem in files("decode_"):
        dd = load(stem)
        if not dd:
            continue
        acc = defaultdict(list)
        for r in dd["rows"]:
            if "error" in r:
                continue
            by = {a["arm"]: a for a in r["arms"]}
            errs = [t["err"] for t in r["tuned"].values() if not t["fp8"]]
            lo, hi = min(errs), max(errs)
            for n in FIXED:
                c, x = _fixed_cell(by, n, lo, hi)
                if x:
                    acc[n].append((x, _gate(by[n]["err"], lo, hi), r["cell"]))
        if not acc:
            continue
        out.append(f"### `{stem}` step boundary\n\n")
        body = []
        for n, xs in acc.items():
            v = [x for x, _, _ in xs]
            best = max(xs, key=lambda t: t[0])
            c = best[2]
            body.append(
                [
                    n,
                    len(v),
                    f(min(v), ".2f"),
                    f(geomean(v), ".2f"),
                    f(max(v), ".2f"),
                    f"D{c['D']} G{c['G']} S{c['S']} {'ragged' if c['ragged'] else 'uniform'}",
                    sum(g == "S" for _, g, _ in xs),
                    sum(g in ("S", "B") for _, g, _ in xs),
                    sum(x < 1.0 for x in v),
                ]
            )
        out.append(
            table(
                ["member", "cells", "min", "geomean", "max", "max at", "strict", "band", "< 1"],
                body,
            )
            + "\n"
        )
    return "".join(out)


def chunk_section(stem="generate"):
    """What a fixed chunk of keys per split costs the step: each chunk
    member's step time over the same depth's default step, per case."""
    d = load(stem)
    if not d:
        return ""
    chunks = d["meta"].get("chunks") or []
    names = [f"Fold dense chunk{c}" for c in chunks]
    if chunks:
        names.append(f"Fold T=16 chunk{max(chunks)}")
    body, best = [], []
    for r in d["rows"]:
        if "error" in r:
            continue
        by = {a["arm"]: a for a in r["arms"] if a.get("cold")}
        cells, dense = [], []
        for n in names:
            base = n.split(" chunk")[0]
            a, b = by.get(f"{n} step"), by.get(f"{base} step")
            x = a["cold"]["us"] / b["cold"]["us"] if a and b else None
            cells.append(f(x, ".3f"))
            if x and base == "Fold dense":
                dense.append(x)
        best.append(min(dense) if dense else None)
        body.append([r["case"], *cells, f(best[-1], ".3f")])
    if not body:
        return ""
    return (
        f"### fixed chunk (`{stem}`)\n\nStep time with a fixed number of keys per split "
        "over the default split's, same depth (below 1 is faster).\n\n"
        + table(["case", *names, "best dense chunk"], body)
        + "\n"
    )


def generate_section(stem):
    d = load(stem)
    if not d:
        return ""
    out = [f"### `{stem}`\n\n", f"{d['meta']['steps']} steps per case.\n\n"]
    for r in d["rows"]:
        if "error" in r:
            out.append(f"**{r['case']}**: error {r['error'][:200]}\n\n")
            continue
        lens = r["lens"]
        out.append(
            f"**{r['case']}** ({r['capture']}, H={r['H']}/{r['HKV']}, D={r['D']}, B={r['B']}, "
            f"prompts {min(lens)}..{max(lens)}). FA-4 error per step: "
            + ", ".join(f(x * 1e3, ".2f") for x in r["fa4_trace"])
            + " (e-3). Speedup over "
            + r["fastest_baseline"]
            + ".\n\n"
        )
        body = []
        for a in r["arms"]:
            body.append(
                [
                    a["arm"],
                    a["boundary"],
                    f(a["cold"]["us"]),
                    f"{f(a['vs_fastest']['ratio'], '.2f')}x" if a.get("vs_fastest") else "–",
                    f(a["err_final"] * 1e3, ".2f"),
                    f(a.get("err_median", 0) * 1e3, ".2f") if a.get("ours") else "–",
                    f(a.get("err_worst", 0) * 1e3, ".2f") if a.get("ours") else "–",
                    f(a.get("live"), ".3f"),
                    *(
                        ("yes" if a["err_final"] <= _budget(r, g) else "no")
                        if a.get("err_final") is not None
                        else "–"
                        for g in ("band", "strict")
                    ),
                ]
            )
        out.append(
            table(
                [
                    "arm",
                    "boundary",
                    "us",
                    "speedup",
                    "final err e-3",
                    "median err",
                    "worst err",
                    "live",
                    "<= band",
                    "<= strict",
                ],
                body,
            )
            + "\n"
        )
    return "".join(out)


PREFIX_FAMILIES = ("FlashInfer cascade", "PAT", "vLLM cascade", "FastTree")


def cascade_summary(stem):
    """Per cascade and prefix-tree cell, FoldAttention's dense cascade against
    each prefix-sharing kernel (FlashInfer's cascade, PAT, vLLM's cascade
    path, FastTree; each in its fastest configuration) and against the
    fastest baseline of any kind, at the decode boundary and with the flat
    member's front (its step minus its decode) charged."""
    d = load(stem)
    if not d:
        return ""
    out, body = [], []
    acc = defaultdict(list)
    for r in d["rows"]:
        if r.get("kind") not in ("cascade", "tree") or "error" in r:
            continue
        by = {a["arm"]: a for a in r["arms"] if a.get("cold")}
        ours = by.get("Fold dense cascade")
        fb = by.get(r["fastest_baseline"])
        step, dec = by.get("Fold dense serving step"), by.get("Fold dense serving")
        if not ours or not fb:
            continue
        us = ours["cold"]["us"]
        row = [r["tag"], f(us)]
        for fam in PREFIX_FAMILIES:
            won = r["tuned"].get(fam)
            a = by.get(won["arm"]) if won else None
            if a:
                x = a["cold"]["us"] / us
                acc[(r["kind"], fam)].append(x)
                row.append(f"{f(a['cold']['us'])} ({f(x, '.2f')}x)")
            else:
                row.append("–")
        charged = us + step["cold"]["us"] - dec["cold"]["us"] if step and dec else None
        x_fb = fb["cold"]["us"] / us
        x_front = fb["cold"]["us"] / charged if charged else None
        acc[(r["kind"], "fb")].append(x_fb)
        if x_front:
            acc[(r["kind"], "front")].append(x_front)
        row += [
            f"{fb['arm']} {f(fb['cold']['us'])} ({f(x_fb, '.2f')}x)",
            f(x_front, ".2f") + "x" if x_front else "–",
        ]
        body.append(row)
    if not body:
        return ""
    out.append(
        "### cascades and prefix trees, dense\n\n"
        + (cascade_summary.__doc__ or "").replace("\n    ", " ").strip()
        + "\n\n"
    )
    out.append(
        table(
            [
                "cell",
                "Fold cascade us",
                *(f"{fam} us (speedup)" for fam in PREFIX_FAMILIES),
                "fastest baseline us (speedup)",
                "speedup, front charged",
            ],
            body,
        )
        + "\n"
    )
    for kind in ("cascade", "tree"):
        parts = []
        for k, label in (
            *((fam, f"over {fam}") for fam in PREFIX_FAMILIES),
            ("fb", "over the fastest baseline"),
            ("front", "with the front charged"),
        ):
            xs = acc.get((kind, k))
            if xs:
                parts.append(
                    f"{label} {f(min(xs), '.2f')}-{f(max(xs), '.2f')}x "
                    f"(geomean {f(geomean(xs), '.2f')}x)"
                )
        if parts:
            out.append(f"- {kind}: " + "; ".join(parts) + "\n")
    return "".join(out) + "\n"


def draft_summary(stem):
    """Per draft shape, FoldAttention dense against the fastest baseline and
    against each verification kernel that runs it (SGLang's FA-3 tree
    verification, XQA, FA-3, FlashInfer; each in its fastest configuration),
    over captures, batches and contexts."""
    d = load(stem)
    if not d:
        return ""
    fams = ("SGLang", "XQA", "FA-3", "FlashInfer")
    acc = defaultdict(list)
    drafts = []
    for r in d["rows"]:
        if r.get("kind") != "draft" or "error" in r:
            continue
        by = {a["arm"]: a for a in r["arms"] if a.get("cold")}
        ours = by.get("Fold dense")
        fb = by.get(r["fastest_baseline"])
        if not ours or not fb:
            continue
        us = ours["cold"]["us"]
        if r["draft"] not in drafts:
            drafts.append(r["draft"])
        acc[(r["draft"], "fb")].append(fb["cold"]["us"] / us)
        for fam in fams:
            won = r["tuned"].get(fam)
            a = by.get(won["arm"]) if won else None
            if a:
                acc[(r["draft"], fam)].append(a["cold"]["us"] / us)
    if not drafts:
        return ""

    def cell(xs):
        if not xs:
            return "–"
        return f"{f(min(xs), '.2f')}-{f(max(xs), '.2f')}x ({f(geomean(xs), '.2f')}x)"

    body = [[dn, cell(acc[(dn, "fb")]), *(cell(acc[(dn, fam)]) for fam in fams)] for dn in drafts]
    return (
        "### draft verification, dense\n\n"
        + (draft_summary.__doc__ or "").replace("\n    ", " ").strip()
        + " Each cell is min-max (geomean) of the baseline's time over Fold's.\n\n"
        + table(["draft", "over the fastest baseline", *(f"over {fam}" for fam in fams)], body)
        + "\n\n"
    )


def compose_section(stem):
    d = load(stem)
    if not d:
        return ""
    out = [f"### `{stem}`\n\n"]
    for r in d["rows"]:
        if "error" in r:
            out.append(f"**{r.get('kind')} {r.get('params')}**: error {r['error'][:200]}\n\n")
            continue
        out.append(f"**{r['tag']}** (speedup over {r['fastest_baseline']})\n\n")
        body = [
            [
                a["arm"],
                a.get("boundary", "decode"),
                f(a["cold"]["us"]),
                f"{f(a['vs_fastest']['ratio'], '.2f')}x" if a.get("vs_fastest") else "–",
                f(a["err"] * 1e3, ".2f"),
                a.get("reference", ""),
            ]
            for a in r["arms"]
        ]
        out.append(table(["arm", "boundary", "us", "speedup", "err e-3", "note"], body) + "\n")
    return "".join(out)


SANITY = 20.0


def _wrong_arms(stems):
    """`{(label, arm)}` whose gradients are over `SANITY` times the best arm's
    error at that shape, from every backward-mode file."""
    wrong = set()
    for stem in stems:
        d = load(stem)
        for r in (d or {}).get("rows", []):
            worst = {
                a["arm"]: max(a["accuracy"].values())
                for a in r.get("arms", [])
                if a.get("accuracy")
            }
            if worst:
                best = min(worst.values())
                wrong |= {(r["label"], n) for n, e in worst.items() if e > SANITY * best}
    return wrong


def backward_section(stem, dash_stem=None):
    d = load(stem)
    if not d:
        return ""
    wrong = _wrong_arms(["backward_fa3", "backward_dash"])
    dash = load(dash_stem) if dash_stem else None
    dash_by = {}
    if dash:
        for r in dash["rows"]:
            if "error" not in r:
                dash_by[r["label"]] = {a["arm"]: a for a in r["arms"]}
    out = [f"### `{stem}` ({d['meta']['mode']})\n\n"]
    if dash:
        out.append(
            "DASH: commit d87bcc9 with `benchmarks/dash_d87bcc9_causal_mask.patch`; "
            "dense batches only, since DASH has no varlen schedule.\n\n"
        )
    out.append(
        "Each ratio is the baseline's time over FoldAttention's, paired per round; DASH's comes "
        "from its own process against that process's FoldAttention. `DASH vs FA-3 det` joins "
        "the two processes through FoldAttention: FA-3 deterministic's time over DASH's, the "
        "comparison DASH's paper reports. `FA-4 drift` is FA-4's ratio in the DASH process "
        "over its ratio here, a check that the join holds. `wrong` marks an arm whose "
        "gradients are over 20 times the best arm's FP32-relative error at that shape; it is "
        "left out of every ratio and geomean.\n\n"
    )
    sets = defaultdict(list)
    for r in d["rows"]:
        sets[r.get("set", "?")].append(r)
    baselines = ["FA-3", "FA-3 det", "FA-4", "FA-4 det", "cuDNN", "DASH"]
    acc = defaultdict(list)
    rep = defaultdict(lambda: [0, 0])
    for s, rows in sets.items():
        body, cols = [], defaultdict(list)
        best_det, best_nd, dash_gain, drift = [], [], [], []
        for r in rows:
            if "error" in r:
                body.append([str(r.get("shape")), "error"] + [""] * len(baselines))
                continue
            by = {a["arm"]: a for a in r["arms"]}
            other = dash_by.get(r["label"], {})
            if other.get("DASH"):
                by["DASH"] = other["DASH"]
            ours = by.get("FoldAttention")
            if not ours:
                continue
            line = [r["label"], f(ours["cold"]["us"])]
            for b in baselines:
                a = by.get(b)
                bad = (r["label"], b) in wrong
                x = a["over_ours"]["ratio"] if a and a.get("over_ours") and not bad else None
                cols[b].append(x)
                line.append("wrong" if bad else f(x, ".3f"))
            det = [cols[b][-1] for b in ("FA-3 det", "FA-4 det") if cols[b][-1]]
            nd = [cols[b][-1] for b in ("FA-3", "FA-4") if cols[b][-1]]
            best_det.append(min(det) if det else None)
            best_nd.append(min(nd) if nd else None)
            x, y = cols["FA-3 det"][-1], cols["DASH"][-1]
            dash_gain.append(x / y if x and y else None)
            a4, b4 = by.get("FA-4"), other.get("FA-4")
            drift.append(
                b4["over_ours"]["ratio"] / a4["over_ours"]["ratio"]
                if a4 and b4 and a4.get("over_ours") and b4.get("over_ours")
                else None
            )
            line += [
                f(best_det[-1], ".3f"),
                f(best_nd[-1], ".3f"),
                f(dash_gain[-1], ".3f"),
                f(drift[-1], ".3f"),
            ]
            body.append(line)
            for n, a in by.items():
                if a.get("accuracy"):
                    acc[n].append(max(a["accuracy"].values()))
                if a.get("repeatable") is not None:
                    rep[n][0] += bool(a["repeatable"])
                    rep[n][1] += 1
        body.append(
            ["**geomean**", ""]
            + [f(geomean(cols[b]), ".3f") for b in baselines]
            + [
                f(geomean(best_det), ".3f"),
                f(geomean(best_nd), ".3f"),
                f(geomean(dash_gain), ".3f"),
                f(geomean(drift), ".3f"),
            ]
        )
        out.append(f"**{s}**\n\n")
        out.append(
            table(
                [
                    "shape",
                    "ours us",
                    *baselines,
                    "fastest det",
                    "fastest nondet",
                    "DASH vs FA-3 det",
                    "FA-4 drift",
                ],
                body,
            )
            + "\n"
        )
    if acc:
        body = [
            [
                n,
                f(max(v) * 1e3, ".3f"),
                f(sorted(v)[len(v) // 2] * 1e3, ".3f"),
                f"{rep[n][0]}/{rep[n][1]}",
            ]
            for n, v in sorted(acc.items())
        ]
        out.append("**accuracy and repeatability** (worst of dQ/dK/dV rel. L2, e-3)\n\n")
        out.append(table(["arm", "worst", "median", "shapes bitwise repeatable"], body) + "\n")
    return "".join(out)


def prefill_section(stem):
    d = load(stem)
    if not d:
        return ""
    out = [f"### `{stem}`\n\n"]
    body = []
    for r in d["rows"]:
        if "error" in r:
            body.append([r.get("model"), r.get("B"), r.get("P"), "error", "", "", ""])
            continue
        for a in r["arms"]:
            body.append(
                [
                    r["model"],
                    r["B"],
                    r["P"],
                    a["arm"],
                    f(a["cold"]["us"]),
                    f(a.get("tflops"), ".0f"),
                    f"{f(a['over_fa4']['ratio'] * 100, '.1f')}%",
                ]
            )
    out.append(table(["model", "B", "P", "arm", "us", "TFLOP/s", "of FA-4 attention"], body))
    return "".join(out) + "\n"


def ablation_section(stem):
    d = load(stem)
    if not d:
        return ""
    out = [
        (
            f"### `{stem}`\n\nSpeed over the fastest BF16 baseline, bytes read per key, "
            "and error (e-3), adding one mechanism at a time.\n\n"
        )
    ]
    for r in d["rows"]:
        if "error" in r:
            continue
        fb = r["fastest_baseline"]
        base = next(a for a in r["arms"] if a["arm"] == fb)["cold"]["us"]
        rows = [
            (
                a["arm"],
                f(a["cold"]["us"]),
                f(base / a["cold"]["us"], ".2f") + "x",
                f(a["bytes_per_key"], ".0f"),
                f(a["err"] * 1e3, ".2f"),
            )
            for a in r["arms"]
        ]
        out.append(f"**{r['tag']}** (fastest {fb})\n\n")
        out.append(table(["arm", "us", "speedup", "B/key", "err"], rows) + "\n")
    return "".join(out)


def bwd_ablation_section(stem):
    d = load(stem)
    if not d:
        return ""
    base = d["meta"]["base"]
    rows, ratios = [], defaultdict(list)
    for r in d["rows"]:
        if "error" in r:
            continue
        for a in r["arms"]:
            ratio = a["over_base"]["ratio"] if a.get("over_base") else None
            ratios[a["arm"]].append(ratio)
            rows.append(
                (
                    r["label"],
                    a["arm"],
                    f(a["cold"]["us"]),
                    f(ratio, ".3f"),
                    f(a["accuracy"]["dq"] * 1e3, ".2f"),
                    "same" if a["repeatable"] else f"{a['max_repeat_diff']['dq']:.1e}",
                    {True: "yes", False: "no"}.get(a.get("dkv_same_as_base"), "–"),
                )
            )
    geo = [(n, f(geomean(v), ".3f"), len(v)) for n, v in ratios.items()]
    return (
        f"### `{stem}`\n\nThe shipped backward (`{base}`) against `backward.ablation`'s "
        "variants in the same kernel: fp32 dQ atomics instead of the integer fold, one CTA "
        "per tile instead of the persistent work list, and one dQ grid per (request, KV "
        "head) instead of one per query row; each arm's time over the shipped kernel's "
        "(above 1 is slower).\n\n"
        + table(["arm", "geomean over base", "shapes"], geo)
        + "\n"
        + table(
            ["shape", "arm", "us", "over base", "dq err e-3", "repeat |d|", "dK/dV bits as base"],
            rows,
        )
        + "\n"
    )


def train_curve_section(stem):
    d = load(stem)
    if not d:
        return ""
    runs = [r for r in d["rows"] if r.get("kind") == "run" and "loss" in r]
    body = [
        (
            f"{r['arm']}#{r['run']}",
            f(sum(r["loss"][-100:]) / len(r["loss"][-100:]), ".4f"),
            f(r["val"][-1]["loss"], ".4f"),
            f(r["step_ms_median"], ".0f"),
            r["weights_sha256"][:12],
        )
        for r in runs
    ]
    cmp_rows = [
        (
            " vs ".join(c["runs"]),
            "bit-identical" if c["bit_identical"] else f"step {c['first_differing_step']}",
            f"{c['max_abs_loss_diff']:.2e}",
            f"{c['mean_abs_loss_diff_last100']:.2e}",
        )
        for c in d["rows"]
        if c.get("kind") == "compare"
    ]
    m = d["meta"]
    return (
        f"### `{stem}`\n\n{m['model']} from scratch on WikiText-103, {m['steps']} steps of "
        f"{m['micro_batches']} x {m['seq']} tokens. Train loss over the last 100 steps, "
        "final validation loss, and where two runs' curves part.\n\n"
        + table(["run", "train loss", "val loss", "ms/step", "weights"], body)
        + "\n"
        + table(["runs", "first difference", "max abs dloss", "last-100 mean"], cmp_rows)
        + "\n"
    )


def grad_elements_section(stem):
    d = load(stem)
    if not d:
        return ""
    out = [
        (
            f"### `{stem}`\n\nPer-element error of dQ/dK/dV on training-state operands "
            "against FP64: |err|/|ref| percentiles and the fraction of elements off by more "
            "than half and one bf16 ulp; then dQ's median relative error by binade below "
            "its (request, KV head) maximum.\n\n"
        )
    ]
    for r in d["rows"]:
        if "error" in r:
            continue
        rows = []
        for n, s in r["arms"].items():
            for w in ("dq", "dk", "dv"):
                x = s[w]
                rows.append(
                    (
                        n,
                        w,
                        *(f"{x['rel'][k]:.1e}" for k in ("p50", "p99", "p99.99", "p100")),
                        f(x["frac_gt_half_ulp"], ".4f"),
                        f(x["frac_gt_1_ulp"], ".4f"),
                    )
                )
        out.append(f"**{r['tag']} layer {r['layer']}**\n\n")
        out.append(
            table(["arm", "grad", "p50", "p99", "p99.99", "max", ">0.5 ulp", ">1 ulp"], rows) + "\n"
        )
        names = list(r["arms"])
        brows = []
        for i, b0 in enumerate(r["arms"][names[0]]["dq"]["by_binade"]):
            if not b0["n"]:
                continue
            brows.append(
                (
                    b0["binade"],
                    f"{b0['frac']:.1e}",
                    *(f"{r['arms'][n]['dq']['by_binade'][i]['rel_p50']:.1e}" for n in names),
                )
            )
        out.append(table(["binade", "share", *names], brows) + "\n")
        pairs = r.get("dq_pairs") or {}
        if pairs:
            prow = []
            for k, v in pairs.items():
                deep = [x for x in v["by_binade"] if x["n"]][-1]
                prow.append(
                    (
                        k,
                        f(v["frac_differ"], ".3f"),
                        f"{v['rel_p50']:.1e}",
                        f"{v['rel_p99']:.1e}",
                        f(v["frac_gt_half_ulp"], ".4f"),
                        f"{deep['rel_p99']:.1e} (binade {deep['binade']})",
                    )
                )
            out.append(
                table(
                    [
                        "dQ pair",
                        "differ",
                        "p50 |a-b|/|ref|",
                        "p99",
                        ">0.5 ulp",
                        "p99, deepest binade",
                    ],
                    prow,
                )
                + "\n"
            )
    return "".join(out)


def oracle_section(stem):
    d = load(stem)
    if not d:
        return ""
    out = [
        (
            f"### `{stem}`\n\nThe mass reference against the exact log-sum-exp in the same "
            "kernel: bytes per key, time and error (e-3) per depth, and what the oracle needs "
            "for the mass point's error.\n\n"
        )
    ]
    for r in d["rows"]:
        if "error" in r:
            continue
        rows = [
            (
                a["arm"],
                f(a["us"]),
                f(a["bytes_per_key"], ".0f"),
                f(a["err"] * 1e3, ".2f"),
                f(a.get("bytes_ratio"), ".3f"),
                f(a.get("us_ratio"), ".3f"),
            )
            for a in r["arms"]
        ]
        out.append(f"**{r['tag']}**\n\n")
        out.append(table(["arm", "us", "B/key", "err", "oracle bytes", "oracle time"], rows) + "\n")
    return "".join(out)


def invariance_section(stems):
    rows = []
    for stem in stems:
        d = load(stem)
        for r in (d or {}).get("rows", []):
            if "arm" not in r:
                continue
            rows.append(
                (
                    r["kind"],
                    r["shape"],
                    r["arm"],
                    *(
                        ("-" if r.get(k) is None else f"{r[k]:.1e}")
                        for k in ("repeat", "batch", "packing", "split")
                    ),
                )
            )
    if not rows:
        return ""
    return (
        "### invariance\n\nLargest absolute difference from the first result "
        "(0 = every bit matched).\n\n"
        + table(["kind", "shape", "arm", "repeat", "batch", "packing", "split"], rows)
        + "\n"
    )


def train_e2e_section(stems):
    rows = []
    for stem in stems:
        d = load(stem)
        for r in (d or {}).get("rows", []):
            for a in r.get("arms", []):
                ours = next(x for x in r["arms"] if x["arm"] == "FoldAttention")["cold"]["us"]
                rows.append(
                    (
                        stem,
                        r["label"],
                        a["arm"],
                        f(a["cold"]["us"] / 1e3, ".1f"),
                        f(a["cold"]["us"] / ours, ".3f"),
                    )
                )
    if not rows:
        return ""
    return (
        "### model training steps\n\nForward, loss and backward; time over "
        "FoldAttention's.\n\n" + table(["file", "model", "arm", "ms", "over ours"], rows) + "\n"
    )


def serve_section(stem):
    d = load(stem)
    if not d:
        return ""
    speed = [r for r in d["rows"] if r.get("kind") == "speed"]
    if not speed:
        return ""
    rows = []
    for r in speed:
        base = min(v["us"] for k, v in r["arms"].items() if not k.startswith("Fold") and "us" in v)
        for k, v in r["arms"].items():
            if "us" in v:
                rows.append(
                    (r["label"], k, f(v["us"] / 1e3, ".2f"), f(base / v["us"], ".2f") + "x")
                )
    return (
        f"### `{stem}`\n\nWhole-model decode step, over the fastest baseline.\n\n"
        + table(["batch", "arm", "ms/step", "speedup"], rows)
        + "\n"
    )


def quality_section(stem):
    """The real-decoding quality modes: teacher-forced likelihood, greedy
    divergence and RULER-style retrieval, each arm against FA-3."""
    d = load(stem)
    if not d:
        return ""
    out = [f"### `{stem}`\n\n"]
    for r in d["rows"]:
        kind = r.get("kind")
        if kind == "nll":
            body = [
                (
                    n,
                    f(a.get("nll"), ".4f"),
                    f"{a['nll_delta']:+.1e}" if a.get("nll_delta") is not None else "–",
                    f"{a['kl']:.2e}" if a.get("kl") is not None else "–",
                    f(a.get("top1_agree"), ".4f"),
                )
                for n, a in r["arms"].items()
            ]
            out.append(
                f"**teacher-forced, S={r['S']}** ({r['docs']} documents, {r['tokens']} tokens)\n\n"
            )
            out.append(
                table(["arm", "NLL", "vs FA-3", "KL from FA-3", "top-1 agreement"], body) + "\n"
            )
        elif kind == "diverge":
            body = [
                (
                    n,
                    f(a.get("first_divergence_median"), ".0f"),
                    f(a.get("never_diverged"), ".2f"),
                    f(a.get("match_fraction"), ".3f"),
                )
                for n, a in r["arms"].items()
                if "error" not in a
            ]
            out.append(
                f"**greedy divergence from FA-3** ({r['prompts']} prompts, {r['gen_tokens']} tokens)\n\n"
            )
            out.append(
                table(["arm", "first divergence (median)", "never", "matching tokens"], body) + "\n"
            )
        elif kind == "ruler":
            body = [
                (n, f(a["accuracy"], ".3f"), f(a["recall"], ".3f"), f(a["token_agreement"], ".3f"))
                for n, a in r["arms"].items()
            ]
            out.append(f"**RULER {r['task']}, S={r['S']}** ({r['samples']} prompts)\n\n")
            out.append(table(["arm", "exact", "recall", "tokens as FA-3"], body) + "\n")
        if r.get("headroom"):
            out.append(_headroom_table(r["headroom"]))
    return "".join(out)


def _headroom_table(head):
    body = [
        (
            n,
            f(h["lse_minus_z_min"], ".1f"),
            f(h["lse_minus_z_max"], ".1f"),
            str(h["outside_window"]),
            str(h["rows"]),
        )
        for n, h in head.items()
    ]
    return (
        "Reference headroom (log-sum-exp over Z, binades) and rows outside the range "
        "certificate's window:\n\n"
        + table(["arm", "min", "max", "outside window", "rows"], body)
        + "\n"
    )


def _merge_headroom(a, h):
    return dict(
        lse_minus_z_min=min(a["lse_minus_z_min"], h["lse_minus_z_min"]),
        lse_minus_z_max=max(a["lse_minus_z_max"], h["lse_minus_z_max"]),
        outside_window=a["outside_window"] + h["outside_window"],
        rows=a["rows"] + h["rows"],
    )


def longbench_section(stems):
    """LongBench v1 (English): each task's score per arm, the category means
    and the mean over tasks, from every `serve_longbench*` file."""
    # files split tasks, arms, or a task's samples (`--lb-offset`); a slice's
    # arms merge across files, first run kept, and slices merge weighted by n
    slices = {}
    for stem in stems:
        d = load(stem)
        offset = (((d or {}).get("meta") or {}).get("args") or {}).get("lb_offset", 0)
        for r in (d or {}).get("rows", []):
            if r.get("kind") != "longbench":
                continue
            t = slices.setdefault((r["task"], offset), dict(r, arms={}, headroom={}))
            for a, v in r["arms"].items():
                t["arms"].setdefault(a, v)
            for a, v in (r.get("headroom") or {}).items():
                t["headroom"].setdefault(a, v)
    by_task = {}
    for (task, _), s in sorted(slices.items(), key=lambda kv: kv[0][1]):
        t = by_task.get(task)
        if t is None:
            by_task[task] = s
            continue
        t["samples"] += s["samples"]
        arms = {}
        for a in t["arms"].keys() & s["arms"].keys():
            x, y = t["arms"][a], s["arms"][a]
            n = x["n"] + y["n"]
            arms[a] = dict(score=(x["score"] * x["n"] + y["score"] * y["n"]) / max(n, 1), n=n)
        t["arms"] = arms
        t["headroom"] = {
            a: _merge_headroom(t["headroom"][a], s["headroom"][a])
            for a in t["headroom"].keys() & s["headroom"].keys()
        }
    rows = list(by_task.values())
    if not rows:
        return ""
    arms = list(dict.fromkeys(a for r in rows for a in r["arms"]))

    def score(r, a):
        v = r["arms"].get(a)
        return v["score"] if v else None

    def mean(rs, a):
        xs = [score(r, a) for r in rs]
        return None if any(x is None for x in xs) else sum(xs) / len(xs)

    body = [
        (r["task"], r["category"], str(r["samples"]), *(f(score(r, a), ".2f") for a in arms))
        for r in rows
    ]
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    for c, rs in cats.items():
        body.append(
            (
                "mean",
                c,
                str(sum(r["samples"] for r in rs)),
                *(f(mean(rs, a), ".2f") for a in arms),
            )
        )
    body.append(
        (
            "mean",
            "all tasks",
            str(sum(r["samples"] for r in rows)),
            *(f(mean(rows, a), ".2f") for a in arms),
        )
    )
    head = {}
    for r in rows:
        for n, h in (r.get("headroom") or {}).items():
            head[n] = h if n not in head else _merge_headroom(head[n], h)
    return (
        "### LongBench v1, English (`serve_longbench*`)\n\n"
        + table(["task", "category", "samples", *arms], body)
        + "\n"
        + (_headroom_table(head) if head else "")
    )


def methods_section(stem):
    """Sparse and quantised decode against FoldAttention's members at one
    byte model, and the running-maximum gate against the declared reference."""
    d = load(stem)
    if not d:
        return ""
    out = [
        (
            f"### `{stem}`\n\nPer method, the fewest bytes per key (metadata included) at "
            "which its error is within the BF16 kernels' range (band) and at or below the "
            "most accurate one (strict), or `never`.\n\n"
        )
    ]
    for r in d["rows"]:
        if "error" in r or not r.get("pareto"):
            continue
        body = [
            (
                m,
                f(x["band"]["bytes_per_key"], ".0f") if x["band"] else "never",
                f(x["strict"]["bytes_per_key"], ".0f") if x["strict"] else "never",
                f(x["best_err"] * 1e3, ".2f"),
            )
            for m, x in r["pareto"].items()
        ]
        band = r["bf16_band"]
        out.append(
            f"**{r['case']}** (BF16 error {f(band['lo'] * 1e3, '.2f')}-{f(band['hi'] * 1e3, '.2f')} e-3)\n\n"
        )
        out.append(table(["method", "band B/key", "strict B/key", "best err e-3"], body) + "\n")
        grows = []
        for m, g in r["gate"].items():
            if "running_max" not in g:
                continue
            rm, r1 = g["running_max"], g["running_max_split1"]
            grows.append(
                (
                    m,
                    f(g["kernel"]["refined"], ".3f"),
                    f(rm["refined"], ".3f"),
                    f(r1["refined"], ".3f"),
                    f(g["kernel"]["live"], ".3f"),
                    f(rm["live"], ".3f"),
                    f(g["z_bytes_per_key"], ".0f"),
                    f(rm["bytes_per_key"], ".0f"),
                    f(r1["bytes_per_key"], ".0f"),
                )
            )
        if grows:
            out.append(
                table(
                    [
                        "member",
                        "refined (Z)",
                        "refined (running max)",
                        "(unsplit)",
                        "live (Z)",
                        "live (running max)",
                        "B/key Z",
                        "B/key running max",
                        "(unsplit)",
                    ],
                    grows,
                )
                + "\n"
            )
    return "".join(out)


def main():
    docs = [(p.stem, json.loads(p.read_text())) for p in sorted(OUT.glob("*.json"))]
    parts = ["# Benchmark results\n\n", env_section(docs)]
    parts.append(headline_section())
    parts.append("## Decode\n\n")
    order = [
        "decode_context_bf16",
        "decode_context_v8",
        "decode_batch_bf16",
        "decode_batch1024_bf16",
    ]
    stems = files("decode_")
    parts += [
        decode_section(s)
        for s in [*[o for o in order if o in stems], *[s for s in stems if s not in order]]
    ]
    parts.append("## Generation\n\n")
    parts += [generate_section(s) for s in files("generate")]
    parts.append(chunk_section())
    parts.append("## Method classes\n\n")
    parts += [methods_section(s) for s in files("methods")]
    parts.append("## Cascades, prefix trees and drafts\n\n")
    parts += [cascade_summary(s) for s in files("compose")]
    parts += [draft_summary(s) for s in files("compose")]
    parts += [compose_section(s) for s in files("compose")]
    parts.append("## Backward\n\n")
    parts.append(backward_section("backward_fa3", "backward_dash"))
    parts.append(backward_section("train_fa3", "train_dash"))
    parts.append(train_e2e_section(files("train_e2e")))
    parts.append("## Ablation and determinism\n\n")
    parts += [ablation_section(s) for s in files("ablation")]
    parts += [bwd_ablation_section(s) for s in files("bwd_ablation")]
    parts += [train_curve_section(s) for s in files("train_curve")]
    parts += [grad_elements_section(s) for s in files("grad_elements")]
    parts += [oracle_section(s) for s in files("oracle")]
    parts.append(invariance_section(files("invariance")))
    parts.append("## End to end\n\n")
    parts.append(serve_section("serve"))
    parts += [
        quality_section(s)
        for s in ("serve_nll", "serve_diverge", "serve_ruler", "serve_nll_long", "serve_ruler_long")
    ]
    parts.append(longbench_section(files("serve_longbench")))
    parts.append("## Prefill\n\n")
    parts += [prefill_section(s) for s in files("prefill")]
    path = Path(OUT) / "RESULTS.md"
    path.write_text("".join(parts))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
