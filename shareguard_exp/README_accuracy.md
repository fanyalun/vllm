# Algorithm Accuracy Scan

The scanner evaluates fixed-ratio branch dropping with `baseline`,
`min_weight`, `renorm`, and `shareguard` on GSM8K, HumanEval/MBPP, MMLU, and
scorable LiveBench samples. It reads datasets and writes results in the
companion SharedGuard data checkout.

```bash
cd /home/fanya/sharedguard

# Validate all local inputs without loading the model.
VALIDATE_ONLY=1 bash code/judge/run_strategy_scan.sh

# Run a four-domain smoke test in an isolated output directory.
LIMIT=16 MAX_NEW_TOKENS=32 \
  OUT_DIR=/home/fanya/sharedguard/results/strategy_scan_smoke \
  bash code/judge/run_strategy_scan.sh

# Run or resume the complete 13-configuration scan.
bash code/judge/run_strategy_scan.sh
```

The reportable default is `EVAL_BATCH_SIZE=1`; padded branches otherwise
participate in fixed-ratio selection. Each configuration writes generation
JSONL and a summary atomically. Re-running the same command resumes incomplete
records. Use a different `OUT_DIR` for smoke and full runs.
