# Qwen GDN mean pre-verification

This experimental option combines shared-weight top-8 to top-4 MoE-Skip with
one recurrent state update per GDN layer and actual Pre-Verify block. It is
restricted to Qwen3.6 MoE, CUDA, TP=1, unquantized weights, FP32 recurrent state,
B=1 and inner D=4. Existing hierarchical restrictions remain in force.

Set `preverify_gdn_mode` in the hierarchical speculative configuration to
`ssm_mean` or `input_mean`. The default `none` retains the existing implementation.
Both MTP and DSpark inner proposers use this same option.

## Pooling and outputs

Pool the entire actual block, including its anchor. The usual D=4 block has
five positions; a short tail has one to five. Padding is never averaged.

For `ssm_mean`, run input projections and causal convolution at all positions.
Average post-convolution keys and values, and average pre-activation a/b gates.
Normalize the pooled key after averaging. For each value head:

```text
k = normalize(mean(k_i))
v = mean(v_i)
g = -exp(A_log) * softplus(mean(a_i) + dt_bias)
beta = sigmoid(mean(b_i))
S_decay = exp(g) * S
S_new = S_decay + beta * (v - S_decay @ k) outer k
y_i = S_new @ normalize(q_i) / sqrt(key_dim)
```

Each position retains its own q, output gate z, gated normalization, and output
projection. The recurrent tensor is loaded, updated, and stored once; all token
queries read the same updated tensor within the kernel. There are no per-token
recurrent checkpoints. Decay is applied once without multiplying g by block
length. This is an approximation rather than an equivalent parallel recurrence.

For `input_mean`, average GDN input hidden states after the layer's input norm
and before GDN input projections. Run the complete GDN branch on this single
pooled vector, including one causal convolution step and one recurrent update.
Broadcast the projected branch output to every original position. Residual
streams, full-attention layers, and MoE processing retain their original widths.

## State ownership

Pre-Verify allocates exactly one private recurrent tensor per layer. Inner
partial or zero acceptance leaves the updated approximate recurrent state in
place, including information from rejected suffixes. Each new outer cycle
initializes it from the Target's accepted state. Private buffers never replace
canonical Target buffers outside the scoped Pre-Verify forward, including when
an exception occurs.

In `ssm_mean`, convolution remains per-token and its short history shifts by
inner accepted position. In `input_mean`, convolution advances by one pooled
step each inner round and is retained along with the approximate SSM. Both
convolution histories are reset from Target at the next outer cycle.

The mode dispatch lives inside an opaque runtime operator so that reuse of a
compiled model cannot freeze the Target/Pre-Verify decision. When an approximation
is enabled, Target uses the existing exact GDN implementation inside that
operator. This changes the compilation boundary and adds an output copy, which
must be included in end-to-end comparisons. With mode `none`, the original
compilation path remains intact.

## Evaluation contract

Pooling a whole candidate block permits suffix information to influence earlier
Pre-Verify predictions. Inner acceptance alone is therefore insufficient.
Compare final Target acceptance, emitted tokens per complete cycle, and
uninstrumented request throughput. Compare output token IDs against an AR
control separately; numerical kernel agreement is not a lossless decoder proof.
Existing hierarchy sequential-equivalence failures also require explicit
accounting when interpreting any output mismatch.

AI assistance was used for the implementation, tests, and analysis.
