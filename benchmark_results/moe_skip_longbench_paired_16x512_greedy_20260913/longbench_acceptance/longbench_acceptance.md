# LongBench greedy acceptance pilot

Qwen3.6-35B-A3B, MoE-Skip top-h=4 versus native MTP, D=4/8/16/32.
Data: sfc-gh-goliaro/longbench-longctx, pinned revision in contract.json.
Selection: all 16 source rows from nominal buckets >=32K, in source order.
Each source sample produces two inputs of exactly 16,384 and 32,768 Qwen tokens.
Only the document is prefix-truncated; instruction, question, choices and
default chat template are identical between each pair. Tokenized prefix,
document prefix and question/chat suffix are concatenated to fit each budget.
There is no thinking override. The template opens a think block.
Each request generates exactly 512 tokens with temperature=0, top_p=1,
ignore_eos=True. First EOS positions are preserved in per-request artifacts.
TP=1, B=1, CUDA graphs, A100 80GB, prefix caching disabled, text only.

Mean acceptance length = 1 + total accepted draft tokens / total spec steps.
Aggregation is weighted by verification steps. The +1 is conventional and
can differ from exact emitted yield at the generation boundary. CSV files
also report accepted-only means, acceptance fractions and emitted-token ratios.
Panels show paired 16K/32K contexts; each point aggregates 16 requests.
This controls task identity, but shortening a document may remove answer evidence.
No throughput, LongBench answer-accuracy score or confidence interval is claimed.

AR output consistency: **failed**, 10/256
speculative requests match all 512 AR tokens. Detailed first divergences are
in output_consistency.json; a failed check is retained in gates_failed.json.
Acceptance-count audit and output-equivalence checks are separate.

At matching D, MoE-Skip versus MTP output consistency:
**passed**,
128/128 pairs match
all 512 tokens. See cross_method_consistency.json. This checks the two methods'
generated trajectories separately from their equivalence to AR.

Artifacts in the run directory: acceptance_summary.csv, acceptance_by_request.csv,
sample_manifest.csv, cells/*/result.json, contract.json and audit.json.
AI assistance was used to prepare the benchmark and report.

Reproduce from the repository root, using the included pinned source.parquet:

```bash
.venv/bin/python benchmarks/moe_skip/run_longbench_acceptance.py \
  --run-dir RUN_DIR --source SOURCE_PARQUET
MPLCONFIGDIR=/tmp/longbench_mpl .venv/bin/python \
  benchmarks/moe_skip/analyze_longbench_acceptance.py --run-dir RUN_DIR
```
