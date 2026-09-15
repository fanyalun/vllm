# 非投机 GDN：ReplaySSM 不写回完整 State 的延迟

单张 A100 80GB，Qwen3.6 GDN 维度：HQ=16、HV=32、K=V=128。单层、单个当前位置，每个请求的完整 FP32 State 为 2 MiB。输入 BF16，历史 d/k FP16，g FP32。现有生产 kernel 和默认 launch 配置，未修改精度、计算逻辑或刷新策略。

Baseline：fused_recurrent_gated_delta_rule_packed_decode，读取上一位置 State，计算当前一个 token 的输出与 State，写回完整 State。

ReplaySSM：fused_recurrent_gated_delta_rule_replayssm，实际分配 h 个缓存位置，write_pos=h-1；基准副本将 b_is_flush 固定为 False。从 checkpoint 重建前 h-1 个历史位置，再处理同一个当前 token，计算当前输出并追加 d/k/g，不写回完整 State。checkpoint 到当前结果恰好跨 h 个位置。

固定当前位置为 33，baseline 初始 State 为 S32；ReplaySSM checkpoint 为 S(33-h)，缓存位置为 33-h 到 31（零基更新编号）。例如 h=8：从 S25 开始，重放得到 S32，再计算 S33。这不是从 S24 重放 8 条历史后再计算 S33 的 h+1 路径。

本次不使用投机解码，不使用合成写入或独立重建核。两个 kernel 都包含 当前 token 的 gate、Q/K 归一化、GDN 更新和输出计算；不包含线性投影、Conv、普通 Attention、MoE 和端到端调度。仅 baseline 写回完整 State。ReplaySSM 保留原生非 flush 的 d/k/g 写入。ReplaySSM 不在全局内存物化当前完整 State，而是计算其对当前输出的作用。本轮因此不是两边都输出完整 State tensor 的 materialization 比较。

batch=1/4/8/16/32，buffer=4/8/16/32；每点7轮，seed=0。ReplaySSM 曲线是7轮中位数；不依赖 buffer 的 baseline 合并四种 buffer 对照的28轮。误差线为 min/max，不是置信区间。纵轴为整个 batch、单个 GDN 层的耗时。

每轮先恢复原始 State。冷条件随后清扫256 MiB 无关 buffer；热条件在清扫后执行3次待测 graph，再次恢复 State。状态恢复、清扫、预热和编译均在计时外。没有硬件计数器证明 冷条件的全部访问都来自 HBM，也不保证大 batch 的热数据全部驻留 L2。

全部20组的当前输出通过 baseline 检查 (rtol=0.02, atol=0.002)。计时外利用更新后的 d/k/g 重建 State，与 baseline 完整 State 对照也通过。ReplaySSM 的整个 checkpoint 逐元素完全不变。关闭边界处的自动 flush 仅发生在基准副本；生产文件未修改。每点测一次，不推进游标或运行满周期，不能据此关闭生产中的刷新。

数据来源：../raw.json 与 ../summary.csv。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/replayssm/qwen36_ar_buffer_cost.py --output benchmark_results/qwen36_a100_ar_no_state_write_20260915
.venv/bin/python benchmarks/replayssm/plot_qwen36_ar_buffer_cost.py --output benchmark_results/qwen36_a100_ar_no_state_write_20260915
```
