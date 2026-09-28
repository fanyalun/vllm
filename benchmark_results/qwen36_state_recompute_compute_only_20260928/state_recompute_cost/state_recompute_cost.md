# SSM State 写回与重计算

数据：`../raw.json`，汇总：`../summary.csv`。三幅面板分别为 bs=1、8、64；横轴为从 h 个位置前重算，纵轴是整批单个 GDN 层的 CUDA event 延迟。点为 21 轮中位数，水平线为同一批量的完整 State 纯写入中位数。

Qwen3.6 GDN 形状：每请求 32×128×128 FP32 State (2 MiB)；历史 d/k 为 FP16、g 为 FP32，矩阵乘为 TF32x3。重算不读取起点 State，也不写出完整终点 State；起点在寄存器构造，历史读取和每 tile 一个校验值的写入计入时间。两种 cache 条件分别是 3 次预热后测量和 256 MiB 清扫后测量，清扫不计时。

复现：`.venv/bin/python benchmarks/replayssm/plot_qwen36_state_recompute_compute_only.py --output benchmark_results/qwen36_state_recompute_compute_only_20260928`。
