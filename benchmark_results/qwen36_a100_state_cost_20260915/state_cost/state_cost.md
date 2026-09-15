# SSM State 写回与重建的小样本比较

Qwen3.6 GDN 维度；单张 A100 80GB；单层，每个请求的 State 为 32×128×128 FP32，即 2 MiB。横轴为 batch size，纵轴为整个 batch 处理一层的耗时，不是每个请求的耗时。

Store only：由寄存器中的简单整数运算生成非零 FP32 模式并写入 State 大小的 buffer，不读取源 State。它是纯 store 成本的近似，包含生成模式的 少量计算和 kernel 启动成本，不是实际模型 flush kernel。

Copy：读取已经物化的当前 State，写入另一个 buffer，包含完整的读和写。

Reconstruct：读取 h 个位置之前的 checkpoint 和 h 条 d/k/g 历史更新，重建并写出当前完整 State。计时包含 checkpoint/历史读取、重建计算和结果写入；不包含投影、草稿验证、历史生成、接受/拒绝和端到端调度。

所有 h 固定同一个当前位置 32，分别从位置 32-h 的 checkpoint 开始。State FP32，d/k 缓存 FP16，g FP32。矩阵乘使用 TF32x3；历史 tile 为 max(16,next_pow2(h))，不是上一轮的固定 64 tile。

每个 batch/h/cache 组合 7 轮，随机交错三种方法；seed=0。重建曲线取 7 轮中位数；与 h 无关的 store/copy 合并四个 h 的共 28 轮。误差线为 min/max，不是置信区间。

每次测量前写入 256 MiB 无关 buffer，操作位于计时外。左图随后先执行 3 次待测 kernel，右图直接测量。清扫操作未通过硬件计数器验证 L2 命中率；热工作集也不保证大 batch 的全部数据能够驻留 L2。CUDA event 包围一次 CUDA graph 中的 kernel，预热、编译、输入构造不计时。

图中重建时间已经包含一次完整 State 写出，因此不能把它全部解释为纯算术成本，也不能把两条曲线相减当作严格隔离的重计算时间。

数据来源：../raw.json、../summary.csv。复现：

```bash
.venv/bin/python benchmarks/replayssm/plot_qwen36_state_cost.py --output benchmark_results/qwen36_a100_state_cost_20260915
```
