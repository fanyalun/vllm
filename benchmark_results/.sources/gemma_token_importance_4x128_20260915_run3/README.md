# Gemma token-importance expert-pool pilot

Gemma4, TP=1, B=1, greedy, four prompts x 128 output tokens. MTP D=4, four inner rounds, outer capacity=20. Pre-Verify and Target share the same model instance and weights. Each configuration has four excluded full-request warmups. All configurations are measured afresh on the same GPU.

Attention60 scores each expert using the sum of normalized native-top8 gate probability times mean-head attention from the last draft position to earlier draft positions. Attention uses the actual paged KV cache and full-context softmax, including GQA, shared KV, sliding masks, scaling and soft-cap. The anchor and last draft have zero scoring weight. Routing60 uses unit weights on the same scoring positions. The candidate expert union includes all four drafts. Retain ceil(0.6 * union size) per layer; break score ties by ascending expert ID. Apply the selected pool to all five Pre-Verify rows, including the anchor and last draft. Renormalize retained gate weights and preserve expert scales. Rows without surviving experts have zero routed contribution; there is no budget-expanding fallback.

This pool is specific to the current Pre-Verify candidate block. Its dependence on the last position makes it a candidate-dependent approximate filter, not a causal AR model. Target routing is untouched.

ms/output and ms/accepted both use complete llm.generate wall time, including scoring, ranking, MTP, Pre-Verify, final Target, API and metric collection. ms/accepted divides by final emitted accepted draft tokens after last-step clipping. Inner metrics and budgets include every proposal initiated by measured requests, including the final unused proposal. No kernel-only timings are presented as end-to-end speedups.

| Method | Mean h | Inner accepted/round | Outer accepted/step | ms/output | ms/accepted | AR speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention60 | 5.648 | 2.725 | 10.596 | 8.351 | 9.195 | 0.952x |
| ar_start | 0.000 | - | - | 7.963 | - | 0.998x |
| routing60 | 5.636 | 2.560 | 10.217 | 8.069 | 8.885 | 0.985x |
| h4 | 4.000 | 2.635 | 10.804 | 7.147 | 7.852 | 1.112x |
| h6 | 6.000 | 2.848 | 12.238 | 7.299 | 7.951 | 1.089x |
| h8 | 8.000 | 2.312 | 12.375 | 6.499 | 7.050 | 1.223x |
| ar_end | 0.000 | - | - | 7.939 | - | 1.002x |

AR repeat exact-output agreement: 0/4. Speculative exact matches against ar_start: attention60=0/4, routing60=0/4, h4=0/4, h6=0/4, h8=0/4. The speedup column is an observed wall-time ratio, not a validated same-output acceleration. Different generated continuations can also change acceptance and expert routing.

This is a small exploratory sample without confidence intervals. The existing Gemma/hierarchical path has unresolved exact-output parity limitations. See output_parity.csv; no lossless or production readiness claim is made. The new scoring and expert masking primitives are checked against GPU references in gpu_reference.log. Aligned assignments keep original token slots and mask expert blocks after assignment, avoiding the earlier invalid-slot limitation.

Reproduce with `.venv/bin/python benchmarks/hierarchical/run_token_importance.py --run-dir <fresh_directory> --gpu 1`, followed by `.venv/bin/python benchmarks/hierarchical/analyze_token_importance.py <fresh_directory>`. Source snapshots and hashes are in contract.json. AI assistance was used.
