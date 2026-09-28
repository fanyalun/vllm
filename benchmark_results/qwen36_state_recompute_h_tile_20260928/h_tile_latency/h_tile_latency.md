# K=h 的 GDN State 重算

数据：`../raw.json`；汇总：`../summary.csv`。六个面板按 batch size (1/8/64) 与缓存条件 (warm/evicted) 排列。每点是 21 轮 CUDA event 测量的中位数，单位为 µs/单层/整批。

K=h software tile 加载 32×h 的 d 与 h×32 的 k，并累加恰好 h 个外积；h-step recurrence 逐步更新 State；两者都无 h 方向填充。Fixed K=16 是上一版 TF32x3 Tensor Core 矩阵重建。State store 是每请求 2 MiB FP32 State 的纯写近似。重算计时不读取 起点 State、不写出完整终点 State，但计入 d/k/g 历史读取和每 tile 一个校验值的写入。warm 在清扫 256 MiB 后额外执行 3 次待测 graph，evicted 则直接测量；清扫不计时。

复现：`.venv/bin/python benchmarks/replayssm/plot_qwen36_state_recompute_exact_h.py --output benchmark_results/qwen36_state_recompute_h_tile_20260928`。
