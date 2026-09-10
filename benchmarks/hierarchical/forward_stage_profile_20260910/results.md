# D=4 Full / MoE-Skip 前向阶段调查

本次实际测量了 Qwen3.6-35B-A3B 与 Gemma4-26B-A4B-it，在同一个模型实例、相同输入 token、位置和前缀状态下比较 routed top-8 与 top-4。主要结果采用启用 torch.compile 后的 CUDA Graph。测量没有修改推理实现。

## 口径与边界

- B=1、TP=1、BF16、A100 80GB PCIe。Qwen 使用 GPU 0，Gemma 使用 GPU 1；没有与其他任务共享同一 GPU。
- D=4 是候选数量。实际验证块包含 anchor 加 4 个候选，共 5 个 query token；另外单独测了严格的 4-token 前向，二者不能混用。
- Full 与 Skip 都走相同的 private preverify metadata 路径。Full 是完整 top-8 backbone，但这里的时间不是原生 Target `execute_model()` 时间。
- 前向总时间包含 backbone、LM head、argmax；不含 metadata 构造、GDN 状态复制、scheduler、采样 bookkeeping、MTP proposal 或下一次 Target verification。
- 每个模式用 3 个前缀，每个前缀先预热，再按 AB/BA 顺序交替重放 20 次。Qwen 前缀长度为 123/7/38，Gemma 为 131/7/37，属于短前缀诊断，不能直接外推长上下文。
- 阶段表使用第一个前缀上的一条 warmed compiled graph CUPTI trace；不是 60 次平均占比。阶段归因结合参考路径的 CPU launch correlation、kernel 名称和顺序；无唯一归属的融合算子归入 Other。RMS fusion 包含 residual 等操作。Gemma dense MLP 块使用四个相邻 kernel 的名称断言核对。
- Qwen shared/routed 并行区间只计一次。表中的 shared-only/routed-only 是不与另一分支重叠的时间；不能把 inclusive kernel work 直接当作串行耗时相加。

## 无阶段插桩的编译后总时间

每项是 3 个前缀共 60 次 CUDA Graph 重放的均值；不含 warmup 与 compilation。

| 模型 | Query token 数 | Full top-8 (ms) | Skip top-4 (ms) | 时间减少 | Full / Skip |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 | 4 | 10.380 | 8.129 | 21.69% | 1.277x |
| Qwen3.6 | 5，D=4 验证块 | 11.671 | 8.707 | 25.40% | 1.340x |
| Gemma4 | 4 | 12.289 | 8.892 | 27.64% | 1.382x |
| Gemma4 | 5，D=4 验证块 | 11.073 | 9.675 | 12.63% | 1.144x |

分位数见 `compiled_totals.csv`。CUDA event 总时间与单条 trace 的跨度有所不同，因为后者只有一个前缀，并受到 profiler 和运行时波动影响；没有按比例强行把阶段时间缩放到均值。

## 5-token / D=4 阶段表

下面每格是毫秒及其占该条 trace 总跨度的比例，所有行相加为 100%。Attention 包含可归因的投影、attention kernel 和输出投影；无法拆分的跨算子 fusion 保留在 Other。

### Qwen3.6

| 阶段 | Full top-8 | Skip top-4 |
| --- | ---: | ---: |
| GDN | 2.632 / 22.95% | 2.632 / 31.19% |
| Full attention | 0.664 / 5.79% | 0.670 / 7.93% |
| Routed experts，非重叠部分 | 4.974 / 43.38% | 2.454 / 29.09% |
| Shared 与 routed/router 重叠 | 1.756 / 15.32% | 1.318 / 15.62% |
| Shared experts，非重叠部分 | 0.304 / 2.65% | 0.230 / 2.72% |
| Router，非重叠部分 | 0.128 / 1.12% | 0.126 / 1.49% |
| Norm / residual fusion | 0.294 / 2.56% | 0.292 / 3.46% |
| LM head | 0.618 / 5.39% | 0.616 / 7.30% |
| Other / unmatched fusion / gaps | 0.098 / 0.85% | 0.100 / 1.19% |
| Trace 总跨度 | 11.468 | 8.438 |

Qwen 有 30 层 GDN 和 10 层 full attention。减少 routed experts 后，GDN 加 full attention 仍占约 39.1%，而且这部分基本没有变快。Shared expert 的独占区间很小，说明删除它的潜在收益不能按 inclusive shared 时间估算；资源竞争与重叠关系会随配置变化。

### Gemma4

| 阶段 | Full top-8 | Skip top-4 |
| --- | ---: | ---: |
| Attention | 2.444 / 21.96% | 2.430 / 25.07% |
| Routed experts | 5.416 / 48.67% | 4.007 / 41.34% |
| 始终执行的 dense MLP | 1.168 / 10.49% | 1.160 / 11.97% |
| Router | 0.316 / 2.84% | 0.319 / 3.29% |
| Norm / residual fusion | 0.544 / 4.89% | 0.546 / 5.64% |
| LM head | 0.919 / 8.26% | 0.909 / 9.38% |
| Other / unmatched fusion / gaps | 0.322 / 2.89% | 0.321 / 3.31% |
| Trace 总跨度 | 11.129 | 9.693 |

Gemma 的 dense MLP 不是 vLLM FusedMoE 中的 shared-expert 对象，但它是每层无条件执行的固定成本。本次模型没有 Qwen 那种单独 shared expert 分支。精确数值见 `compiled_stages.csv`。

## 为什么 Gemma 的 5-token Full 反而比 4-token Full 快

`vllm/model_executor/layers/fused_moe/fused_moe.py::_prepare_expert_assignment()` 使用 `num_tokens * top_k * 4 <= global_num_experts` 选择 naive assignment，否则使用排序/对齐路径。

Gemma 有 128 个 routed experts：4x8x4=128，使用 naive；5x8x4=160，转向 aligned。top-4 的两种宽度仍走 naive。CUPTI trace 中，只有 5-token/top-8 出现每层的 `moe_align_block_size_kernel` 与 `count_and_sort_expert_tokens_kernel`，同时 fused expert GEMM 的耗时下降。Qwen 的 256 experts 使本次四种组合都满足 naive 条件。

这是代码分支与 trace 一致的解释，但没有强制路径的消融实验，因此不能断言仅改这个阈值就一定加速。两模型的日志也都报告缺少该 A100 形状的专用 tuned MoE config，使用默认配置。此次没有测硬件带宽/SM utilization，不能把所有固定成本都称为 memory-bound。

## 对 Pre-Verify 设计的含义

1. Qwen 应优先调查 GDN 的可复用计算或近似：它在 Skip 中占约 31%，加 attention 约 39%。只进一步减少 routed experts 会越来越受这些固定成本限制。优化 GDN 仍须保留 acceptance-aware 的临时状态语义。
2. Gemma 同时受 routed experts 与固定 backbone 成本限制。Skip 中 attention+dense MLP+LM head 约占 46.4%。应对 dense 分支近似、attention 投影/状态复用或较浅预验证器分别做质量实验；计时本身不能证明这些近似保留接受率。
3. 在改模型结构前，先做小 query-width 的 MoE 路径与配置消融。Gemma 的 4/5-token 差异说明，token 数和 top-k 会改变 kernel 调度收益，不能只按 expert FLOPs 推算。
4. D=4/5-token 下，Skip 仍需 Full 的约 74.6%（Qwen）与 87.4%（Gemma）时间，而且还没计入私有状态准备。它作为额外 Pre-Verify 的成本偏高；最终必须用完整 proposal→Pre-Verify→Target 周期及实际 emitted token 数判断是否值得。

## 验证与产物

- 编译模式与未编译诊断模式各 6 个请求，插桩/诊断开启与关闭的 token IDs 一致；共 12/12。每个图重放的预测也保持一致。
- 这验证的是测量扰动控制，不是已有 hierarchical decoder 的 lossless correctness 认证。
- 原始 eager/graph trace、编译后 trace、每次 event 时间、实际输入 token 和预测保留在原始归档中。
- `compiled_stages.csv` 是主要阶段表；`kernel_stages.csv` 是未编译参考路径的 inclusive kernel work；`stages.csv` 是有明显 Event 扰动的辅助表，不能当作生产占比。
- 逐模块 Event 在初始诊断中带来了约 15%–36% 的扰动，主要图已改用没有逐模块 Event 的 compiled graph trace。
- 单图输出位于 `compiled_forward_comparison/`；辅助插桩图位于 `forward_stage_comparison/`。

本次工作由 Codex AI 辅助完成，只增加测量、分析及相关测试，没有修改 runtime 模型或 decoder。
