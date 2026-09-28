# D=3：生产 GDN verify 的 ReplaySSM 历史长度扫描

目标是检验“多提交 h 个位置的历史，会让 ReplaySSM 每轮多花一次完整 State
重建时间”这一假设。单张 A100 80GB PCIe，单个 Qwen3.6 形状 GDN 层，
bs=1/8/64、草稿长度 3、verify 窗口宽度 4；State FP32、QKV BF16、
d/k 历史 FP16。每点 21 次 CUDA Graph / CUDA event 计时，取中位数，单位
µs/层/整个 batch。

ReplaySSM 的每个 h 都从同一起点 State 初始化，依次运行 h/4 个完整的前置
verify + commit，真实填入历史并检查 `write_pos=h`、无 flush；随后**只计**
下一轮生产 verify kernel。Dual 则运行一轮全部接受的 verify + commit，检查
候选 State 已提升、`write_pos=0`，同样只计下一轮 verify kernel。准备过程均
不在计时内。不同 h 的起点 State/输入相同，但 GPU cache 驻留可受准备过程影响；
此扫描说明实际路径的延迟趋势，不是隔离出的纯历史计算成本。

| bs | ReplaySSM h=0 | h=4 | h=8 | h=12 | Dual 全接受后 h=0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.336 | 14.336 | 14.336 | 14.336 | 13.312 |
| 8 | 29.696 | 25.600 | 25.600 | 25.600 | 44.032 |
| 64 | 154.624 | 153.600 | 154.624 | 156.672 | 245.760 |

bs=64 时，ReplaySSM 从 h=4 增到 h=12 只多约 3.072 µs；Dual 全接受后
即使没有已提交历史，单个 verify 仍比 ReplaySSM h=4 多约 92.160 µs。
bs=8 的 h=0 点比 h≥4 稍慢，说明这些点也受准备后的缓存/调度条件影响，
不能把单点差值解释为精确的历史计算开销。h=16 会进入提前 flush 边界，
因此没有纳入这组无 flush 扫描。

当前 ReplaySSM verify 用 checkpoint 对当前 q/k 做投影，并把历史 d/k 对输出
的作用融合进同一 kernel；它在普通 verify 中不写出一整份重建后的 State。
历史 tile 的编译宽度固定为 16，因此 h=4/8/12 改变有效加载/掩码，并不
逐步扩大 tensor-core dot 的 tile。Dual 每轮另外形成并写出完整 FP32
候选 State，以便全接受时提升；预测失败也先支付这笔成本。参见
[相同 launch 配置的诊断](../qwen36_dual_vs_replayssm_acceptance_d3_config_probe_20260928/readme.md)。

[先前 K=h State 重建微基准](../qwen36_state_recompute_h_tile_20260928/readme.md)
测量独立的逐步递推/软件 tile 与纯写：重建计时不读取起点 State，也不写出
完整终点 State；纯写由寄存器构造值。该实验说明孤立重建对 h 的敏感性，
但两项时间差**不是**当前融合 verify 的 `Dual − ReplaySSM` 预测值。

[原始 315 次计时](raw.json) · [中位数](summary.csv) ·
[环境与源码哈希](environment.json) · [源码快照](source.py.txt) ·
[完成标记](measurement_complete.json) · [校验](validation.json)

```bash
.venv/bin/python benchmarks/replayssm/dual_checkpoint_history_cost_d3.py \
  --output benchmark_results/qwen36_dual_vs_replayssm_d3_history_sweep_20260928
```

这些数字为单个 kernel 的延迟，不是完整模型或生成吞吐；没有做硬件级读写
字节归因。
