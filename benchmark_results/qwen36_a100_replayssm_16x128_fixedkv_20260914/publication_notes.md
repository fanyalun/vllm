# 测量源码与发布文件

`source/*.py.txt` 是实际测量脚本的逐字副本，SHA-256 见
`source_manifest.json`。所有 36 个模型配置的 `launch.json` 都记录了相同的
worker 脚本 SHA；原始数据没有用发布后的脚本重新覆盖。

仓库发布版本按已有检查规则，将三处 `torch.cuda.synchronize()` 改为
`torch.accelerator.synchronize()`，单层 kernel 脚本也改用
`torch.Event(device="cuda", enable_timing=True)`。完整差异保留在
`publication_diff.patch`，两种版本的 SHA 见 `publication_source.json`。
这些调整不涉及模型、kernel、输入、接受逻辑或测量循环。

发布版本的 B1/D4 单层 GPU 验证通过，记录在 `publication_smoke.json`；
该记录仅验证 API 与数值检查，不参与正式图表。逐字复测使用冻结版本，
恢复文件名及完整命令见 `readme.md`。

GitHub 包含配置、模型权重指纹、逐轮 token IDs、日志、kernel 汇总、
数值检查及 PNG/PDF 图。36 份体积较大的 CUDA 原始 trace 保留在本地
各配置的 `profile/` 目录，文件名、大小与 SHA-256 见 `trace_manifest.json`。
缺少原始 trace 时，分析脚本使用已保存的 `kernel_summary.json` 重绘图表。

代码与报告使用了 AI 辅助，未提交上游 PR。
