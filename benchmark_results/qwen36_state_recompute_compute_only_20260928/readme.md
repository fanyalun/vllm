# Qwen3.6 GDN：State 写回与从 h 步前重算

## 测量范围

在一张 NVIDIA A100 80GB PCIe 上测单个 GDN 层、整个 batch 的 CUDA Graph kernel 延迟。每个请求的 FP32 State 为 32×128×128，即 2 MiB；bs=1/8/64 分别对应 2/16/128 MiB。h=1/4/8/16。每个条件的两项操作各交错测量 21 次，表格为中位数，单位 µs。

- **State 写回**：`state_store` 由寄存器生成非零 FP32 数据并写满 State 大小的 buffer，不读取源 State。包含少量数据生成与 kernel 启动开销；这是纯写成本的近似，不是生产 GDN flush kernel。
- **重算**：使用 GDN 历史 `d/k/g` 和前缀衰减的矩阵式重建公式，`d/k` 为 FP16、`g` 为 FP32，`tl.dot` 使用 TF32x3。起点 State 在寄存器构造，**计时不读取起点 State，也不写出完整终点 State**；历史 `d/k/g` 读取、矩阵计算、每个 32×32 tile 的一个校验值写入，以及 kernel 启动都计入。因此它是“无 State I/O 的重算 kernel 延迟”，不是纯算术指令耗时。
- **warm/evicted**：每次计时前清扫 256 MiB 无关 buffer；warm 额外执行 3 次待测 graph，evicted 直接测量。清扫和额外预热均在计时外；未用硬件计数器验证 L2 命中率。

所有 h 的历史 tile 固定为 16，较短历史被掩码。因此 h≤16 时，矩阵乘的工作形状相同。State 写回与 h 无关；同一 bs 内的微小中位数变化是测量波动。bs=1 的 CUDA event 读数以约 1.024 µs 为阶梯，不能据此推断小于这一量级的趋势。

## 结果

### Warm，µs / 单层 / 整批

| bs | 写回，约 | h=1 重算 | h=4 重算 | h=8 重算 | h=16 重算 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.12 | 8.19 | 7.17 | 8.19 | 8.19 |
| 8 | 11.26 | 23.55 | 23.55 | 23.55 | 24.58 |
| 64 | 76.80 | 150.53 | 151.55 | 151.55 | 151.55 |

### Evicted，µs / 单层 / 整批

| bs | 写回，约 | h=1 重算 | h=4 重算 | h=8 重算 | h=16 重算 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.12 | 8.19 | 8.19 | 8.19 | 8.19 |
| 8 | 13.31 | 24.58 | 24.58 | 24.58 | 24.58 |
| 64 | 75.78 | 151.55 | 151.55 | 152.58 | 156.67 |

同口径下，warm 重算/写回约为 bs=1 的 1.4–1.6×、bs=8 的 2.1–2.2×、bs=64 的 2.0×。这些是**孤立 kernel** 的比较；没有计入产生历史、读取 checkpoint State、终点 State 完整物化、其他层或完整生成过程，不能解释为 ReplaySSM 或服务端吞吐的加速比。

## 正确性与来源

完整输出仅在计时外生成，并与独立 FP64 GDN 重建公式逐元素比较；全部 12 个 bs×h 组合通过 `rtol=0.002, atol=0.0002`，最大绝对误差约 9.6e-9。正式运行覆盖 24 个 bs×h×cache 条件、1008 个事件计时值。运行环境、Git HEAD 和源码 SHA-256 见 [environment.json](environment.json)；原始计时见 [raw.json](raw.json)，逐点中位数见 [summary.csv](summary.csv)，[完成标记](measurement_complete.json)记录覆盖。计时源码保存在 [source.py.txt](source.py.txt)。

[曲线 PNG](state_recompute_cost/state_recompute_cost.png) · [曲线 PDF](state_recompute_cost/state_recompute_cost.pdf) · [图注](state_recompute_cost/state_recompute_cost.md)

复现：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/replayssm/qwen36_state_recompute_compute_only.py --output benchmark_results/qwen36_state_recompute_compute_only_20260928
.venv/bin/python benchmarks/replayssm/plot_qwen36_state_recompute_compute_only.py --output benchmark_results/qwen36_state_recompute_compute_only_20260928
```

本机另保留两轮先前的独立运行 `qwen36_state_recompute_compute_only_20260928_initial/` 和 `qwen36_state_recompute_compute_only_20260928_pre_lint/`，均未并入正式统计，也未上传。代码与报告由 AI 辅助生成；本次没有提交上游 PR。
