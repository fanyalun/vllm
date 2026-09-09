# Three-level speculative decoding (experimental)

This implementation adds `method="hierarchical"` to the V2 GPU runner. It is
an experimental implementation with failing strict model equivalence gates,
not a validated lossless or performance release. See [results.md](results.md).

## Configuration

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/data1/fanya/Qwen/Qwen3.6-35B-A3B",
    max_model_len=1024,
    max_num_seqs=1,
    enable_prefix_caching=False,
    async_scheduling=False,
    limit_mm_per_prompt={"image": 0, "video": 0},
    speculative_config={
        "method": "hierarchical",
        "inner_method": "mtp",
        "inner_num_speculative_tokens": 4,
        "inner_num_rounds": 4,
        "moe_skip_top_h": 4,
        "draft_sample_method": "greedy",
    },
)
outputs = llm.generate("Write a Python quicksort function.", SamplingParams(
    temperature=0, max_tokens=256,
))
```

For DSpark, set `inner_method="dspark"` and `model` inside the speculative
configuration to the DSpark checkpoint. The pre-verifier always reuses the
Target model instance and parameters; there is no second MoE checkpoint load.

Supported scope: Qwen3.6 MoE, TP1/PP1/DP1, one active text request, standard
rejection sampling, greedy small drafter, no prefix caching, LoRA, structured
outputs, RecoverSSM, expert parallelism, or asynchronous scheduling. Target
sampling can be greedy or stochastic. Unsupported combinations fail closed.

## Execution and ownership

Each outer cycle starts from the Target's committed prefix and its newly
sampled anchor. The small drafter proposes up to `D` tokens. The shared
MoE-Skip pre-verifier evaluates `[anchor, draft...]` at top-h, retains the
continuous matching prefix, and appends its recovery or bonus token. Its
accepted GDN state and hidden features feed the next inner round. After `N`
rounds the Target verifies the resulting candidate sequence once at top-k.

The maximum capacity is `N * (D + 1)`, including each inner recovery/bonus;
for D=4/N=4 it is 20, not 16. The actual candidate length is sent to the
scheduler, and unused capacity is never presented as real candidates.

GDN convolution and recurrent candidates are separate private tensors. Block
zero remains reserved for the GDN null-block sentinel. At each outer boundary,
the accepted Target convolution history and recurrent state are copied into
private row one. Inner acceptance advances both states to the same position.
Private GDN cache bindings are scoped to pre-verifier execution and restored
even when a forward raises. Target state remains authoritative.

Attention uses the request's reserved, uncommitted suffix slots. This first
implementation relies on B=1, disabled prefix sharing, causal attention, and
starting strictly after the computed prefix; it does not implement a general
copy-on-write attention page allocator. The next Target verification rewrites
those suffix slots with Target values. This restriction must remain in place
until broader ownership/isolation tests exist.

Target rejection receives `draft_logits=None`: the final deterministic
candidate generator is a point-mass proposal. Passing the pre-verifier softmax
would describe a different proposal distribution. GPU distribution tests cover
lengths 1/4/20 and proposals outside the Target probability support. They do
not certify numerical equivalence of the complete model execution.

Target and MTP prefill graphs include variable candidate widths. The
pre-verifier has independent metadata builders, captures graphs per width,
and refreshes captured metadata tensor contents before replay. Small-model
graph managers are reused. Pre-verifier graphs are captured lazily; warmup is
required before timing. The current inner loop still synchronizes to read
acceptance counts on the CPU. Eliminating those synchronizations remains work.

## Reproduction

Run from the repository root using the existing environment; no installation
is performed by these scripts. `run_cell.py` defaults to the local four-prompt
dataset and validates every prompt hash. Paths can be overridden explicitly.

```bash
.venv/bin/python benchmarks/hierarchical/run_cell.py \
  --method hierarchical --inner-method mtp --eager --check-preverify \
  --num-samples 1 --max-tokens 64 --device 0 \
  --output /tmp/hierarchical_mtp_checked.json \
  --log /tmp/hierarchical_mtp_checked.log \
  --trace-dir /tmp/hierarchical_mtp_checked_trace

.venv/bin/python benchmarks/hierarchical/run_cell.py \
  --method hierarchical --inner-method dspark --device 0 \
  --output /tmp/hierarchical_dspark_graph.json \
  --log /tmp/hierarchical_dspark_graph.log \
  --trace-dir /tmp/hierarchical_dspark_graph_trace

.venv/bin/python -m pytest tests/config/test_hierarchical_config.py \
  tests/v1/worker/gpu/spec_decode/test_hierarchical.py -q

.venv/bin/python -m pytest tests/v1/spec_decode/test_rejection_sampler_utils.py \
  -k deterministic_cascade -q
```

`--check-preverify` runs a same-root sequential oracle, compares the retained
prefix, reports per-layer differences, and raises on the first mismatch. It
disables pre-verifier graph replay and is unsuitable for throughput timing.
`--temperature 1` selects Target temperature 1/top-p 0.95. Identical seeds do
not imply identical stochastic token trajectories between methods.

`audit.py --reference AR.json --cells CELL.json ... --output AUDIT_DIR`
checks greedy token equality and writes JSON/CSV. It exits nonzero on failure
and never creates `RUN_COMPLETE`. Raw e2e throughput includes prefill and
orchestration. Optional engine timing fields remain null if the engine did
not collect them. Three interleaved repetitions of the full six-method,
two-temperature performance matrix remain pending the correctness gates.

AI assistance was used for this implementation and its validation scripts.
