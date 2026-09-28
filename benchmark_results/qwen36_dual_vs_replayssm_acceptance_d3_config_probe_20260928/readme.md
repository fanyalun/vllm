# D=3 的 Dual-checkpoint verify 配置控制实验

此实验解释[主实验](../qwen36_dual_vs_replayssm_acceptance_d3_20260928/readme.md)
中 bs=8/64 的 Dual-checkpoint 延迟。测量单个 Qwen3.6 形状的 GDN verify
kernel；先执行上一轮 verify 与 commit，以生成对应的 State 和历史，再只计下一轮
直接 verify kernel。草稿长度 3，接受 3/2 个草稿对应内核接受计数 4/3。

两种方法都固定 `block_v=64, nk=2, num_stages=2`，分别测试 1、4 warps。
本机 A100 的原生配置为 ReplaySSM 1 warp、Dual 4 warps。每点 21 次 CUDA
Graph / CUDA event 计时，表中为中位数，单位 µs/单层/整批。两种 warp 配置
的输出逐元素比对通过（`atol=0.01, rtol=0.04`），最大绝对差为
`3.05e-5`。

| bs | 接受草稿 | ReplaySSM 1 warp | ReplaySSM 4 warps | Dual 1 warp | Dual 4 warps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 14.336 | 10.240 | 21.504 | 13.312 |
| 1 | 2 | 13.312 | 10.240 | 25.600 | 15.360 |
| 8 | 3 | 26.624 | 26.624 | 45.056 | 43.008 |
| 8 | 2 | 26.624 | 26.624 | 53.248 | 54.272 |
| 64 | 3 | 152.576 | 150.528 | 315.392 | 245.760 |
| 64 | 2 | 152.576 | 150.528 | 347.136 | 310.272 |

相同的 4-warp 配置下，全接受的 bs=8/64 仍分别多用 16.384/95.232 µs。
在 bs=64，把 Dual 改为 1 warp 会使全接受延迟进一步从 245.760 增至
315.392 µs。因此原生 warp 配置没有造成 Dual 的主要退化，反而减轻了它。

当前代码在每次 verify 中都生成并写入完整候选 State；接受结果尚未确定，
所以最后一个草稿失败时也支付这部分成本。每请求 State 为
`32 × 128 × 128 × 4 = 2 MiB`，bs=8/64 的候选 State 写量分别为 16/128 MiB。
候选分支还再次加载起点 State，并做窗口尾部的矩阵计算。相比之下，D=3 的
ReplaySSM 每次窗口写入的 d/k/g 历史约为 48.5 KiB/请求；其历史 replay
也需要计算，但不会在每次 verify 物化一整份 FP32 State。两种路径同时变化，
本探针只排除了 warp 配置这个混杂因素，**未把候选写入的成本单独拆出**。

[原始 504 次计时](raw.json) · [中位数](summary.csv) ·
[环境及源码哈希](environment.json) · [源码快照](source.py.txt) ·
[完成标记](measurement_complete.json) · [校验](validation.json)

```bash
.venv/bin/python benchmarks/replayssm/dual_checkpoint_config_probe_d3.py \
  --output benchmark_results/qwen36_dual_vs_replayssm_acceptance_d3_config_probe_20260928
```

这只是单层内核诊断，不是完整生成吞吐，也没有提供硬件级读写字节计数。
