# D=3：Dual-checkpoint 与 ReplaySSM 的 GDN 内核延迟

测量日期：2026-09-28。单张 NVIDIA A100 80GB PCIe（GPU 0）；基线代码
`03e0065f73ce00b06f1df4a2aaec6bd93e249416`。这是单个 Qwen3.6 形状的
GDN 层、整个 batch 的 CUDA Graph / CUDA event 时间，单位为 µs；不是完整模型或
生成吞吐。

## 接受长度与状态路径

草稿长度为 3，verify 窗口宽度为 4（前置 token + 3 个草稿）。内核参数
`num_accepted` 包含前置 token，因此接受 3 个草稿时传 4，最后一个草稿失败、
接受 2 个草稿时传 3。两种模式使用完全相同的输入。输入 QKV 为 BF16，SSM State
为 FP32，d/k 历史为 FP16；H=16、HV=32、K=V=128。ReplaySSM 的逻辑历史容量
为 20、物理 ring 为 32；Dual-checkpoint 的 hard cap 为 16、物理 ring 为 16。
这些分别是当前实现的原生容量配置，不是相同内存占用的比较。

单次切换测量先执行上一轮 verify 以实际生成历史和 Dual 的候选 State，然后
执行对应的 commit，最后执行下一轮 verify。该短切换没有触发 flush。全接受时
Dual 将候选 State 提升为 head，并将历史长度清零；只接受两个草稿时 Dual 不提升
head，和 ReplaySSM 一样提交接受的输入。两轮 verify 的输出已比较；每种配置
也连续运行 32 轮，逐轮检查两种模式的输出（`atol=0.01, rtol=0.04`）。六组
连续轨迹均通过，最大绝对差为 0.0001221。

## 单次切换

下表是 `pair`：上一轮 verify + commit + 下一轮 verify 的连续 GPU 时间。
每点 21 次 CUDA Graph 计时，显示中位数。括号是 Dual 相对 ReplaySSM 的延迟变化，
正值表示更慢。`pair` 覆盖候选 State 的生成与写入，也覆盖下一轮状态读取和计算。

| bs | 接受草稿 | ReplaySSM | Dual | Dual 变化 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 29.696 | 28.672 | -3.4% |
| 1 | 2 | 29.696 | 32.768 | +10.3% |
| 8 | 3 | 55.296 | 89.088 | +61.1% |
| 8 | 2 | 54.272 | 102.400 | +88.7% |
| 64 | 3 | 314.368 | 507.904 | +61.6% |
| 64 | 2 | 314.368 | 571.392 | +81.8% |

为观察后一轮验证核本身，另计时 `verify_kernel`，即 commit 后只运行直接的 GDN
verify kernel；它不含 commit、条件 flush kernel 的启动，也不含上一轮写候选 State：

| bs | 接受草稿 | ReplaySSM verify | Dual verify | Dual 变化 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 14.336 | 13.312 | -7.1% |
| 1 | 2 | 14.336 | 16.384 | +14.3% |
| 8 | 3 | 26.624 | 43.008 | +61.5% |
| 8 | 2 | 26.624 | 55.296 | +107.7% |
| 64 | 3 | 152.576 | 245.760 | +61.1% |
| 64 | 2 | 152.576 | 310.272 | +103.4% |

`summary.csv` 还给出 `previous_decode`、`commit`、`next_decode` 的各自中位数。
其中 `next_decode` 是 **commit + 完整 decode 路径**，以保持测量前的 State
一致；不可把它当成纯 verify kernel。各组件各自计时，中位数不可相加来替代
`pair` 的实测时间。

## 连续周期

另以同一接受结果连续运行，预热 32 周期后捕获 32 周期 CUDA Graph；每次测量
连续 replay 20 次，得到 640 周期平均时间，重复 3 次并取中位数。周期包括
commit、条件 flush 启动和 verify。这验证全接受时 ReplaySSM 的定期 flush
开销，以及最后一个失败时两个方法的 flush 开销。

| bs | 接受草稿 | ReplaySSM µs/周期 | Dual µs/周期 | Dual 变化 | ReplaySSM / Dual flush 率 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 14.203 | 12.941 | -8.9% | 25.0% / 0% |
| 1 | 2 | 14.043 | 15.387 | +9.6% | 20.3% / 20.3% |
| 8 | 3 | 22.877 | 40.341 | +76.3% | 25.0% / 0% |
| 8 | 2 | 22.507 | 54.662 | +142.9% | 20.3% / 20.3% |
| 64 | 3 | 179.606 | 255.499 | +42.3% | 25.0% / 0% |
| 64 | 2 | 175.885 | 342.837 | +94.9% | 20.3% / 20.3% |

flush 率来自每种轨迹随后 64 周期的每请求计数。Dual 全接受的提升率为
100%；只接受两个草稿时为 0%。连续 32 轮的输出检查记录在
[`steady_correctness.json`](steady_correctness.json)。

## 解释与边界

在 bs=1、全接受时，Dual 通过候选 State 提升省掉了后续重算，实测周期缩短
8.9%。在 bs=8/64，Dual 即使全接受也更慢；最后一个草稿失败时没有候选 State
提升，退化更明显。这里的差异包含写候选 State、状态读写、tile/launch 配置、
ring 容量及 flush 行为，**不能解释为单独的 State 写入时间**。Dual 和原版
verify kernel 使用各自当前的配置；没有强制相同 tile。这组结果也不说明完整
模型的 speculative decoding 延迟或端到端吞吐。

[固定 verify 配置的补充实验](../qwen36_dual_vs_replayssm_acceptance_d3_config_probe_20260928/readme.md)
将两种方法都设为 `block_v=64, nk=2, num_stages=2, num_warps=4`；全接受
时 bs=8/64 的 Dual 仍分别多用 16.384/95.232 µs，说明原生 warp 配置
并非这两组退化的主因。

## 文件与复现

- [延迟图 PNG](acceptance_latency/acceptance_latency.png)、[PDF](acceptance_latency/acceptance_latency.pdf)、[图说明](acceptance_latency/acceptance_latency.md)
- [单次切换原始 21 次计时](raw.json)、[汇总](summary.csv)、[环境](environment.json)、[完成标记](measurement_complete.json)
- [连续周期原始 3 次计时](steady_raw.json)、[汇总](steady_summary.csv)、[环境](steady_environment.json)、[完成标记](steady_complete.json)
- [结果校验](validation.json)；`source.py.txt`、`steady_source.py.txt`、`steady_correctness_source.py.txt` 为运行时代码快照

从仓库根目录运行（需要已有的 vLLM GPU 环境）：

```bash
.venv/bin/python benchmarks/replayssm/dual_checkpoint_acceptance_d3.py \
  --output benchmark_results/qwen36_dual_vs_replayssm_acceptance_d3_20260928
.venv/bin/python benchmarks/replayssm/dual_checkpoint_steady_correctness_d3.py \
  --output benchmark_results/qwen36_dual_vs_replayssm_acceptance_d3_20260928/steady_correctness.json
.venv/bin/python benchmarks/replayssm/plot_dual_checkpoint_acceptance_d3.py \
  --output benchmark_results/qwen36_dual_vs_replayssm_acceptance_d3_20260928
```

连续周期原始计时由 `benchmarks/replayssm/dual_checkpoint_kernel.py` 的 `measure`
函数生成，轨迹为 `all` 与 `penultimate`，对应内核接受计数 4 与 3。原始
记录和代码快照足以复算 `steady_summary.csv`。
