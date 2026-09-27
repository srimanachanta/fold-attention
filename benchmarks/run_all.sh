#!/bin/bash
# Run the whole suite on one H100, then write out/RESULTS.md.
#
#   EFA_FA3_BWD=<dir> EFA_DASH=<dir> EFA_PAT=<dir> EFA_CAPTURE_DIR=<dir> benchmarks/run_all.sh
#
# EFA_FA3_BWD and EFA_DASH are directories holding an FA-3 build with a
# backward and DASH (d87bcc9 + dash_d87bcc9_causal_mask.patch), each as
# `flash_attn_interface`; without them the backward runs whatever
# `flash_attn_interface` imports and skips DASH.
# EFA_CAPTURE_DIR holds the post-RoPE captures (benchmarks/harness/captures.py).
# EFA_PAT is a PAT checkout (README), for compose's PAT and FastTree arms.
# Pass section names to run a subset, e.g. `benchmarks/run_all.sh decode backward`.
set -u
cd "$(dirname "$0")/.."
P="python -m benchmarks"
SECTIONS="${*:-decode generate methods compose ablation backward invariance prefill serve training}"

for s in $SECTIONS; do
  case $s in
    decode)
      $P.decode --out decode_context_bf16
      $P.decode --v8 --out decode_context_v8
      # batch size at fixed context, fewer depths so the widest batch fits
      $P.decode --row-groups 32 64 128 512 --contexts 4096 16384 --depths 12 14 16 \
        --out decode_batch_bf16
      $P.decode --row-groups 1024 --contexts 4096 --depths 12 14 16 --out decode_batch1024_bf16
      ;;
    generate) $P.generate --out generate ;;
    compose) $P.compose --out compose ;;
    backward)
      $P.backward --sets headline small gqa dash-paper dash-paper-gqa varlen --out backward_fa3
      if [ -n "${EFA_DASH:-}" ]; then
        # DASH schedules dense batches only, so it takes no varlen set
        $P.backward --impl dash --sets headline small gqa dash-paper dash-paper-gqa --out backward_dash
      fi
      $P.bwd_ablation --sets headline dash-paper-gqa --out bwd_ablation
      $P.backward --mode train --sets headline small --out train_fa3
      if [ -n "${EFA_DASH:-}" ]; then
        $P.backward --impl dash --mode train --sets headline small --out train_dash
      fi
      ;;
    prefill) $P.prefill --out prefill ;;
    methods) $P.methods --out methods ;;
    ablation) $P.ablation --out ablation ;;
    invariance)
      $P.invariance --impl fa3 --no-decode --out invariance_fa3
      [ -n "${EFA_DASH:-}" ] && $P.invariance --impl dash --no-decode --out invariance_dash
      $P.invariance --no-backward --out invariance_decode
      ;;
    serve)
      $P.serve --parts speed --out serve
      $P.serve --parts nll --batch-tokens 262144 --out serve_nll
      $P.serve --parts diverge --out serve_diverge
      $P.serve --parts ruler --batch-tokens 262144 --out serve_ruler
      $P.serve --parts longbench --arms FA-3 FlashInfer "FlashInfer FP8" "Fold dense" "Fold T=16" "Fold T=14" --out serve_longbench
      $P.serve --yarn --parts nll --contexts 65536 131072 --docs 16 --arms FA-3 FlashInfer "FlashInfer FP8" "Fold dense" "Fold T=16" "Fold T=14" --out serve_nll_long
      $P.serve --yarn --parts ruler --contexts 65536 131072 --samples 32 --arms FA-3 FlashInfer "FlashInfer FP8" "Fold dense" "Fold T=16" "Fold T=14" --out serve_ruler_long
      ;;
    training)
      $P.train_e2e --impl fa3 --out train_e2e_fa3
      [ -n "${EFA_DASH:-}" ] && $P.train_e2e --impl dash --out train_e2e_dash
      # the curves' first FA-3 det run saves the operands grad_elements scores
      $P.train_curve --steps 2000 --out train_curve
      $P.grad_elements --out grad_elements
      ;;
    *) echo "unknown section $s" >&2; exit 2 ;;
  esac
done
$P.summarize
