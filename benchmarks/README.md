# Benchmarks

The suite behind the paper's measurements. Every script uses the harness in
`harness/`, so every number is taken under the same controls:

- **Tuned baselines.** Each library is timed in every configuration it offers
  at the shape (paging, `pack_gqa`, split count, backend) and the fastest one
  is compared against. The result file names that configuration.
- **One reference.** Every arm is scored against one FP32 computation in one
  row order. An arm whose error is 20 times the best arm of its precision is
  dropped as computing a different function.
- **Timing.** CUDA-graph replay, L2 evicted before every sample, a rotating
  arm order with strides coprime to the arm count, and ratios taken per round.
- **Call boundaries.** A decode ratio divides decode-only times, and a step
  ratio step times: our front (Q's planes, the append and the reference)
  ahead of the decode and combine, against each baseline's fastest append of
  the token (its own fused one where it has one) ahead of its tuned decode.
- **Provenance.** Each result file records the device, library versions,
  clocks, the source revision and its digest, the command line and every raw
  sample. It records nothing that identifies the machine or its users: no
  host name, GPU UUID or absolute path (`harness/report.py`).

## Scripts

| script | measures |
|---|---|
| `decode.py` | Decode latency and accuracy per shape: head dim 64/128, query group 1/4/8/16, context 1K to 32K, uniform and ragged batches, bf16 or 8-bit V, truncation depths, row-group (batch) sweeps. Against FA-3, FA-4, FlashInfer, XQA and cuDNN, plus their FP8 variants. Also bytes read and fraction of the read ceiling. |
| `generate.py` | Multi-step generation on real captures through `FoldKVCache`: per-step error, live fraction, then decode and step latency on the final state against every baseline, each baseline's step being its fastest append ahead of its tuned decode. Also the capacity members, which never read a second plane, and fixed chunks of 512, 1024 and 2048 keys per split. |
| `methods.py` | Sparse and quantized decode on the generation states, emulated in exact arithmetic (Quest, Faster Flash Decoding, KIVI, INT8, FP8) beside FoldAttention's kernel members: error against bytes read per key, a running-maximum gate against the declared reference, and the cut mass each finite depth models. |
| `ablation.py` | Decode mechanisms added one at a time (both key planes, gated plane B, one or two weight terms, 8-bit V, finite depth). |
| `invariance.py` | What stays bit-identical under repeat, batch, packing and split, for the backward and for decode (default split, fixed split, fixed chunk). |
| `compose.py` | Shared-prefix cascades and prefix trees (a system prompt for the batch, a document per group of requests), against FlashInfer's multi-level cascade, PAT, vLLM's cascade path, FastTree and the paged baselines over shared pages; draft verification, chains and trees of 2 to 16 nodes, against FA-3, FA-4, XQA, FlashInfer and SGLang's FA-3 tree verification. A level or draft past 64 rows runs on the wide kernel (`decode/wide.py`). The prior-art arms are in `harness/prior.py`. |
| `backward.py` | The training backward against FA-3, FA-4 (deterministic and not), cuDNN and DASH: latency, FP32-relative accuracy of dQ/dK/dV, and bitwise repeatability. `--mode train` times forward plus backward. |
| `bwd_ablation.py` | The shipped backward against `fold_attention.backward.ablation`'s variants in the same kernel: fp32 dQ atomics instead of the integer fold and one CTA per tile instead of the persistent work list; beside FA-3 and FA-4. |
| `train_curve.py` | Loss curves of a 1B Llama trained from scratch on WikiText-103 with each library's attention, same data order and initialisation; repeat runs test bitwise reproducibility of the whole run. `--capture` saves attention operands from real training states. |
| `grad_elements.py` | Per-element dQ/dK/dV error on those captured operands against FP64: relative-error percentiles, bf16-ulp fractions, and error by binade below each (request, KV head) maximum, with the fp32 dQ variant beside the libraries. |
| `oracle.py` | The same decode driven by each row's exact log-sum-exp instead of the estimated reference: bytes and error at matched error. |
| `serve.py` | Qwen3-8B with the whole decode step in one CUDA graph: step latency per attention arm (`--parts speed`), teacher-forced likelihood and KL (`nll`), greedy divergence (`diverge`), RULER-style retrieval (`ruler`), and LongBench v1's English tasks with its prompts and metrics (`longbench`, `harness/longbench.py`); `--yarn` extends the context to 128K. |
| `train_e2e.py` | Whole-model training steps (forward, loss, backward) of Llama-shaped models with each library's attention. |
| `prefill.py` | Prompt attention (FA-4, FA-3, FlashInfer) and the cost of writing the quantised cache against a bf16 page append. |
| `capture.py` | Writes the post-RoPE Q/K/V captures the decode benchmarks read, from one prefill of a public model. |
| `summarize.py` | Writes `out/RESULTS.md` from the result files. |

## Running

```bash
EFA_CAPTURE_DIR=/path/to/captures \
EFA_FA3_BWD=/path/to/fa3-with-backward \
EFA_DASH=/path/to/dash \
EFA_PAT=/path/to/PAT \
benchmarks/run_all.sh            # or: benchmarks/run_all.sh decode backward
```

`compose.py`'s prior art needs vLLM 0.30.0 and sglang-kernel 0.4.7, both
installed with `--no-deps` beside torch 2.13 and FlashInfer 0.7.0, and PAT:
a checkout of github.com/flashserve/PAT at 8cb067f, installed from that
directory with `CUTLASS_ROOT` at CUTLASS v4.8.0 and `TORCH_CUDA_ARCH_LIST=9.0`.
`EFA_PAT` points at the checkout, whose `benchmark/FastTree.py` supplies
FastTree. An arm whose library is missing prints as unavailable.

Each script also runs alone (`python -m benchmarks.decode --help`). Results go
to `benchmarks/out/` (`EFA_BENCH_OUT` overrides it), one JSON file per run,
rewritten after every row so an interrupted run keeps what it measured.

Decode accuracy needs post-RoPE captures: a `torch.save` dict with `q`
`(1, H, S, D)` and `k`, `v` `(1, H_KV, S, D)` for one layer over one sequence.
`harness/captures.py` lists the files the suite expects, and `capture.py`
writes them from the public model weights into `EFA_CAPTURE_DIR` (its
docstring has the three commands). The backward and
prefill use random operands.

FA-3 and DASH both ship the extension `flash_attn_3._C`, which registers the
same `torch.ops.flash_attn_3` operators, so one process can load only one of
them. The backward runs them in separate processes, and both processes also
time FA-4 and FoldAttention, which join the two runs.

DASH is measured only where its schedules apply, dense batches; it has no
varlen schedule. It is built from its last commit, d87bcc9, with
`dash_d87bcc9_causal_mask.patch`: DASH's reversed m-loop left the causal
diagonal block unmasked at head dim 64, and its D=64 causal gradients were
15-60 times off FP32. The patch restores upstream FA-3's masking bound and
changes nothing else. To build it, apply the patch to a d87bcc9 checkout and
build `hopper/` as FA-3 is built.

## Requirements

An SM90 GPU, `flash-attn-4`, FlashInfer (for FlashInfer and XQA), FA-3 for its
arms, and triton for the bandwidth ceiling. A library that is missing or fails
at a shape is reported as absent with its error, and the rest of the run
continues.
