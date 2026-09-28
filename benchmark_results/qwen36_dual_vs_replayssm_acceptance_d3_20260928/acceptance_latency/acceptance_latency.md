# D=3 的 ReplaySSM 与 Dual-checkpoint GDN 延迟

数据：`../raw.json` 与 `../steady_raw.json`；中位数见 `../summary.csv` 和 `../steady_summary.csv`。横轴是接受草稿数，3 为全部接受，2 为最后一个草稿未接受。上排测上一轮 verify、按接受结果 commit 和下一轮 decode 的连续两窗口 GPU 时间：每点 21 次 CUDA Graph 计时的中位数，误差线为 min/max。下排测连续全接受或连续只接受两个草稿的稳态平均周期时间：每点三次独立测量的中位数，误差线为 min/max；每次测量包含 20×32 个连续周期。

单张 A100 80GB PCIe、单个 Qwen3.6 形状 GDN 层，BF16 输入、FP32 State、FP16 d/k 历史。原 ReplaySSM 使用逻辑 cap=20/物理 ring=32，Dual 使用 hard cap=16/物理 ring=16。数值为整个 batch 而非每请求。这些结果不代表端到端吞吐。

复现：`.venv/bin/python benchmarks/replayssm/plot_dual_checkpoint_acceptance_d3.py --output benchmark_results/qwen36_dual_vs_replayssm_acceptance_d3_20260928`。
