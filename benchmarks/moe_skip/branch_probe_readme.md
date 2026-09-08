# MoE-Skip Top-1 / Top-2 branch probe

This diagnostic tests whether changing a low-margin draft token changes the
remaining draft continuation. It does not measure Target acceptance or speedup.
It requires this checkout's existing local MoE-Skip implementation and model/data
paths from `run_performance.py`; these benchmark files alone do not add MoE-Skip
support to upstream vLLM.

## Contract

- Qwen3.6 and Gemma4, one GPU per model; TP=1 and B=1.
- First four prompts from each previous interleaved dataset: HumanEval, Alpaca,
  GSM8K and UltraFeedback. Raw text, no chat template.
- Exactly 256 output tokens per prompt, greedy, ignore EOS, seed=0.
- MoE-Skip top-h=4, D=16, eager execution, prefix caching disabled.
- Probe every position with raw logit margin strictly below 1 and a nonempty
  suffix within the request's output budget. Include all proposal positions,
  not just actual rejection positions. Do not recursively branch the alternative.

## Semantics and checks

For each normal proposal, retain the Top-1 draft. For every eligible position,
replay the proposal from the same canonical Target prefix, force the highest
scoring alternative token at that position, and continue draft greedy decoding.
Each replay invokes the existing scratch-state preparation. Compare token IDs
at equal offsets strictly after the branching token.

The actual greedy argmax defines Top-1. If `topk` orders tied maxima differently,
use its first token as the alternative and assert the margin is zero. The
alternative is forced only at the selected step. Its logit is raised inside a
benchmark-only hook; original proposal margins are saved before intervention.

Assertions check the identical shared prefix and forced alternative for every
event. A final unforced replay must exactly reproduce the original full draft;
it restores draft buffers and pending trace records before normal verification.
Only the original proposal is submitted to Target. Independent control runs
omit the probe and compare full output sequences.

Suffix outcomes are entire suffix equal, no equal aligned tokens, or partial
agreement. Continuous agreement from the start is reported separately from
individual-position agreement. Offset curves condition on the suffix reaching
that offset; their event counts vary, so they need not be monotonic.

Draft agreement is a proxy for sensitivity. It neither proves Target acceptance
when branches agree nor proves zero reusable tokens when they disagree.

## Reproduction

Run from `/home/fanya/vllm` in the existing environment. Use a fresh output root.
No dependency installation is required.

```bash
export PATH=/home/fanya/vllm/.venv/bin:$PATH
export PYTHONPATH=/home/fanya/vllm/benchmarks/moe_skip:/home/fanya/vllm
export VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_OFFLINE=1
mkdir -p benchmark_results/branch_probe_new
CUDA_VISIBLE_DEVICES=1 \
VLLM_MOE_SKIP_TRACE_DIR=$PWD/benchmark_results/branch_probe_new/qwen_trace \
.venv/bin/python benchmarks/moe_skip/run_branch_probe.py \
  --model qwen36 --output benchmark_results/branch_probe_new/qwen36
CUDA_VISIBLE_DEVICES=0 \
VLLM_MOE_SKIP_TRACE_DIR=$PWD/benchmark_results/branch_probe_new/gemma_trace \
.venv/bin/python benchmarks/moe_skip/run_branch_probe.py \
  --model gemma4 --output benchmark_results/branch_probe_new/gemma4
.venv/bin/python benchmarks/moe_skip/analyze_branch_probe.py \
  benchmark_results/branch_probe_new
```

The two model commands can run concurrently on their respective devices.
For a control run, use a fresh output and trace directory and add `--control`.
Each completed cell has `RUN_COMPLETE`, `audit.json`, `contract.json`, four
outputs and every branch's token IDs, margin, matching mask and decoded text.
The analyzer writes PNG/PDF plots, summary/offset CSVs and representative examples.
