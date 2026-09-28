# Qwen3.6 GDN：真实 K=h tile 的 State 重算时间

## 实现与计时边界

本轮在一张 NVIDIA A100 80GB PCIe 上测单个 GDN 层的整个 batch，bs=1/8/64，h=1/4/8/16。每请求的 FP32 State 是 32×128×128（2 MiB）；历史 d/k 为 FP16，g 为 FP32。每个 bs×h×cache 条件，四项操作随机交错计时各 21 次，以下为 CUDA event 中位数，单位 µs/单层/整批。

- **State store**：由寄存器生成非零值并写满 State 大小的 buffer，是纯写近似，非生产 flush kernel。
- **Fixed K=16 dot**：上一轮的 TF32x3 Tensor Core 矩阵重建；h<16 被掩码。
- **h-step recurrence**：恰好 h 次 `State ← exp(g)·State + d⊗k` 逐步更新。
- **K=h software tile**：加载 32×h 的 d tile 与 h×32 的 k tile，以及 h 个 gate；用恰好 h 个外积累加闭式重建结果。h 方向没有填充，仍在同一个 32×32 State tile 中完成。这不是 Tensor Core `tl.dot`。

为何使用软件 tile：在本机 Triton 3.6.0/A100 上，FP32 `tl.dot(tf32x3)` 直接设置 K=1/4/8 会因 **K≥16** 的下限编译失败，K=16 才能运行。见 [探针日志](dot_k_probe.log)和[探针源码](dot_k_probe_source.py.txt)。因此 K=h 软件 tile 与固定 K=16 Tensor Core dot 的延迟差异同时包含 K 宽度和执行路径的差异。

两种 K=h 重算及固定 K=16 对照都在寄存器构造起点 State；计时**不读取起点 State，也不写出完整终点 State**。历史 d/k/g 读取、重算、每个 32×32 tile 的一个校验值写入、kernel 启动均计入。warm 是清扫 256 MiB 无关 buffer 后额外执行 3 次待测 graph，再计时；evicted 是清扫后直接计时。清扫与额外预热不计时；未用硬件计数器验证 L2 命中率。

## Warm 结果

| bs | h | State store | Fixed K=16 | h-step recurrence | K=h software tile |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 5.12 | 7.17 | 5.12 | 5.12 |
| 1 | 4 | 5.12 | 7.17 | 6.14 | 7.17 |
| 1 | 8 | 5.12 | 8.19 | 8.19 | 7.17 |
| 1 | 16 | 5.12 | 8.19 | 11.26 | 10.24 |
| 8 | 1 | 11.26 | 23.55 | 9.22 | 10.24 |
| 8 | 4 | 11.26 | 23.55 | 14.34 | 15.36 |
| 8 | 8 | 11.26 | 23.55 | 20.48 | 21.50 |
| 8 | 16 | 11.26 | 23.55 | 35.84 | 43.01 |
| 64 | 1 | 75.78 | 150.53 | 39.94 | 40.96 |
| 64 | 4 | 75.78 | 151.55 | 71.68 | 81.92 |
| 64 | 8 | 75.78 | 152.58 | 120.83 | 135.17 |
| 64 | 16 | 74.75 | 152.58 | 223.23 | 306.18 |

## Evicted 结果

| bs | h | State store | Fixed K=16 | h-step recurrence | K=h software tile |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 5.12 | 8.19 | 6.14 | 6.14 |
| 1 | 4 | 5.12 | 8.19 | 7.17 | 7.17 |
| 1 | 8 | 5.12 | 8.19 | 10.24 | 8.19 |
| 1 | 16 | 5.12 | 9.22 | 16.38 | 11.26 |
| 8 | 1 | 13.31 | 24.58 | 9.22 | 9.22 |
| 8 | 4 | 13.31 | 24.58 | 15.36 | 15.36 |
| 8 | 8 | 13.31 | 24.58 | 25.60 | 22.53 |
| 8 | 16 | 13.31 | 24.58 | 46.08 | 44.03 |
| 64 | 1 | 76.80 | 151.55 | 39.94 | 40.96 |
| 64 | 4 | 74.75 | 153.60 | 75.78 | 82.94 |
| 64 | 8 | 74.75 | 154.62 | 138.24 | 137.22 |
| 64 | 16 | 74.75 | 155.65 | 268.29 | 309.25 |

**K=h 软件 tile 的时间随 h 明显增加。** Warm 条件下，bs=64 从 h=1 的 40.96 µs 增至 h=16 的 306.18 µs；纯写约 75 µs，h=4 的软件 tile 已略慢于纯写。逐步递推在同一条件下为 39.94→223.23 µs，说明“真实 K=h tile 的代价”取决于具体计算组织形式。固定 K=16 的曲线基本持平，反映 Tensor Core dot 的 K 下限及固定工作形状。bs=1 的小幅差别接近 CUDA event 约 1 µs 的读数阶梯，不能过度解释。

## 正确性与产物

完整终点 State 仅在计时外写出。K=h software tile、h-step recurrence、Fixed K=16 都与独立 FP64 重建公式逐元素比对；12 个 bs×h 组合均通过 `rtol=0.002, atol=0.0002`。软件 tile 的最大绝对误差约 1.76e-8。正式运行覆盖 24 个 bs×h×cache 条件、2016 个事件计时值。最初的 32×h×32 三维广播归约实现未通过数值预检，未产生正式计时，本地保留失败记录；本报告仅使用修正后通过校验的实现。

[原始计时](raw.json) · [逐点中位数](summary.csv) · [环境与源码哈希](environment.json) · [完成标记](measurement_complete.json) · [校验](validation.json)

[曲线 PNG](h_tile_latency/h_tile_latency.png) · [PDF](h_tile_latency/h_tile_latency.pdf) · [图注](h_tile_latency/h_tile_latency.md)

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/replayssm/qwen36_state_recompute_exact_h.py --output benchmark_results/qwen36_state_recompute_h_tile_20260928
.venv/bin/python benchmarks/replayssm/plot_qwen36_state_recompute_exact_h.py --output benchmark_results/qwen36_state_recompute_h_tile_20260928
```

计时源码与固定 K=16 依赖的运行快照分别见 [source.py.txt](source.py.txt) 和 [fixed16_source.py.txt](fixed16_source.py.txt)。这些是孤立 kernel 延迟，不包含历史生成、真实 checkpoint State 读取、终点 State 完整物化或完整生成；不能直接推断服务吞吐。代码与报告由 AI 辅助生成，未提交上游 PR。
