# 实验结果归档

本目录是本地完整归档。新增实验产物默认由 Git 忽略，代码与选定的轻量说明单独提交。仍有复现价值的大型数据集、真实输入张量和正式输出留在本地。

已删除 129 份大型 profiler / Nsight 时间线，保留事件聚合；旧清理清单已无损压缩。详见[原始大文件清理报告](reports/raw_profile_cleanup.md)。逐事件时间线与 correlation 分析不再可恢复。

按“优化类别 → 优化组合 → 模型 → 参数配置 → 运行来源”浏览。所有原实验已迁出根目录；跨配置共享材料集中存放，单配置结果文件实际放在对应运行目录的 `raw/` 中。

本次归档 63 个原实验目录，拆出 618 个配置来源；配置来源数不是独立正式实验数。

| 分类 | 配置来源数 | 入口 |
| --- | --- | --- |
| 基线与方法比较 | 96 | [00_baselines](<00_baselines/readme.md>) |
| MoE 专家优化 | 139 | [01_moe](<01_moe/readme.md>) |
| GDN 与状态优化 | 145 | [02_gdn](<02_gdn/readme.md>) |
| 早停与轮数 | 78 | [03_early_exit](<03_early_exit/readme.md>) |
| 执行与批处理 | 160 | [04_execution](<04_execution/readme.md>) |

四种优化沿用 MoE、GDN、早停、执行优化；AR、方法比较与质量控制另放 `00_baselines`，避免挤入优化分类。多优化组合按原实验研究的主要问题归类。

## 命名规则

`draft_mtp__moe_top_h__gdn_windowed__tail_carry__stop_balanced/qwen36/b4_tp1_d4_r4_h4_win5_a0p95_beta0p36328125_n32_out128_rep2_greedy_graph/`

- 优化组合：`__` 分隔组合中的方法；具体预算与阈值放参数层。
- 参数：`b`=batch，`tp`=张量并行，`d`=Draft 长度，`r`=内循环轮数，`h`=专家数，`p`=专家权重阈值，`top_p`=专家累计质量阈值，`win`=GDN 窗口，`a/beta`=GDN 门限，`n`=样本数，`out`=输出上限，`ctx`=上下文长度，`rep`=重复数，`w`=核函数/固定前向的 token 宽度。
- `run_日期_实验主题_来源_编号` 区分同配置的不同运行；版本、完整 spec、采样参数、状态 dtype 与行筛选见 `config.json` / `provenance.json`。
- `qwen36` / `gemma4` 为导航简称；精确模型路径、checkpoint 与模型规模仍以原始配置为准。

## 报告与查询

- [四类优化结论总报告](<reports/preverify_optimization_summary_20260921/readme.md>)：保留性能、接受率、负面结果与公平性限制。
- [原实验与共用材料索引](<reports/study_index.md>)：覆盖全部 63 个原目录。
- [配置索引 CSV](<reports/configuration_index.csv>)：按模型、组合、参数、测量类型筛选。
- [完整配置与指标 JSON](<.archive/configuration_catalog.json>)。
- [旧路径到新路径及 SHA-256](<.archive/relocation_manifest.json>)；[迁移验证](<.archive/validation.json>)。

## 数据解释

每个运行目录包含配置、指标、来源及原始数据入口。原始多配置 JSON 保留一份，使用 selector 与原始行号定位；没有重新运行 GPU 实验。端到端、kernel、审计与调参记录保持区别。

旧报告中的吞吐统计方式保持不变；新 metrics.json 如有 pooled_tokens_per_second，表示所选 e2e 记录总 token / 总时间，不能直接替代原报告的逐轮中位数。接受率分母以原始字段和报告为准。有效负面结果继续保留。

原始文件内嵌的历史绝对路径保留用于复现，不批量改写证据；使用 relocation_manifest.json 查当前位置。新实验建议直接使用本结构并在 run 目录保存 contract/config 与 completion marker。
