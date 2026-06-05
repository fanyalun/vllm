# Qwen3.6 MTP Large-Batch KV/State-Cache Investigation

This note records the debugging path for the large-batch Qwen3.6 MTP speculative decoding experiment on 4x RTX 5090 GPUs.

Experiment directory:

```text
examples/features/speculative_decoding/results/full_4x5090_modelscope_local_v8_mlen768_ffnfix_20260603
```

## Initial Symptom

The submitted global batch size increased up to 256, but larger speculative draft lengths produced suspicious results. The effective number of tokens observed in valid scheduler steps stopped scaling with submitted batch size.

For `batch_size=256, draft_length=6`, the median effective step token count was about `140`, which corresponds to:

```text
140 / (6 + 1) = 20 active requests globally
```

With 4 DP/EP ranks, that is roughly:

```text
20 / 4 = 5 active requests per GPU
```

The runtime log matched this interpretation:

```text
Running: 5 reqs, Waiting: 59 reqs, GPU KV cache usage: 99.1%
```

So the submitted batch was 256 globally, but the real verification batch at `draft_length=6` was often only about 20 requests globally.

## KV Offloading Was Not Enabled

The run metadata and engine logs did not contain `kv_offloading_size` or `kv_offloading_backend`. The `LLM(...)` construction in the experiment runtime also did not pass KV offloading arguments.

Therefore, the waiting requests were not stored as CPU KV cache. In normal non-offloading scheduling:

- `WAITING` requests usually have no KV cache yet; they only have CPU-side request/token metadata.
- `RUNNING` requests own GPU KV/state-cache blocks and participate in the current step.
- preempted requests free their GPU blocks and later recompute or resume through normal scheduling, unless a KV connector/offloading path is enabled.

This explains why the run can finish even when the full submitted batch does not fit in GPU KV cache: vLLM queues requests and admits only a smaller active set.

## Why `draft_length=6` Drops Concurrency

The first rough hypothesis was that MTP `d=6` simply makes every request need `1 + 6 = 7` positions in the current speculative verification step. That is partly true for scheduler slot reservation, but it is not the whole story.

For ordinary full-attention KV, `draft_length=6` does not turn a 768-token sequence into `768 * 7` persistent KV tokens. Verification computes target KV for positions such as:

```text
L, L+1, L+2, L+3, L+4, L+5, L+6
```

Rejected draft KV is not kept as future context. If only two draft tokens are accepted, only the accepted prefix state should remain.

The main issue for Qwen3.6 is that the model is hybrid:

```text
40 total layers
10 full_attention layers
30 linear_attention layers
```

The linear-attention layers use recurrent state-cache, implemented through vLLM's Mamba-style cache spec. This state is not stored once per prompt token after prefill; prefill mainly leaves the final recurrent state plus a small convolution window. However, speculative decoding needs multiple candidate state versions corresponding to accepting `0..d` draft tokens.

For this model's gated-delta-net state:

```text
linear_num_key_heads = 16
linear_num_value_heads = 32
linear_key_head_dim = 128
linear_value_head_dim = 128
linear_conv_kernel_dim = 4
TP = 1
```

The state shapes are approximately:

```text
conv_dim = 128 * 16 * 2 + 128 * 32 = 8192
conv_state ~= 8192 * (conv_kernel - 1 + num_spec)
temporal_state = 32 * 128 * 128 = 524,288 elements
```

With BF16 cache state, one linear-attention layer is roughly:

```text
d=0:
  (8192 * 3 + 524288) * 2 bytes ~= 1.05 MiB/request/layer

d=6:
  page size grows slightly from the longer conv state, then vLLM reserves
  (1 + num_speculative_blocks) pages:
  ((8192 * 9 + 524288) * 2) * 7 ~= 7.98 MiB/request/layer
```

Across about 30 linear-attention layers, this becomes roughly:

```text
d=0: 31 MiB/request
d=6: 239 MiB/request
```

This per-request recurrent state-cache expansion is large enough to dominate the active-request capacity.

## Log Evidence

The vLLM KV capacity report for normal decode showed:

```text
GPU KV cache size: 28,992 tokens
Maximum concurrency for 768 tokens per request: 37.75x
```

For `draft_length=6`, the report showed:

```text
GPU KV cache size: 3,909 tokens
Maximum concurrency for 768 tokens per request: 5.09x
```

The ratio is:

```text
37.75 / 5.09 = 7.42
```

This is close to `1 + draft_length = 7`, with extra loss from padding and speculative-state overhead.

## Final Conclusion

For Qwen3.6, large-batch MTP speculative decoding is not limited only by ordinary full-attention KV length. The main capacity limiter is the linear-attention recurrent state cache.

Linear attention is still friendly to long-prompt prefill in the usual algorithmic sense: it avoids full `L x L` attention work and does not retain a full recurrent state for every prompt position. The problem appears during speculative decoding, especially at large draft lengths, because vLLM must reserve multiple candidate recurrent states for draft verification.

Practical implications:

- Submitted batch size is not the same as effective active batch size.
- Analyze speculative runs using effective active requests, for example `step_total_tokens / (draft_length + 1)`, not only submitted batch size.
- For Qwen3.6 MTP, `draft_length=2` is more likely to remain useful at larger batches; `draft_length=4` and `draft_length=6` can lose too much concurrency.
- KV cache offloading does not solve this active-step concurrency limit because requests participating in the current step still need GPU-resident KV/state-cache.
