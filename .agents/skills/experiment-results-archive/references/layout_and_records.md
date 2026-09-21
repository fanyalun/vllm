# 目录与记录约定

## 默认结构

```text
benchmark_results/
  readme.md
  00_baselines/
  01_moe/
    readme.md
    draft_mtp__moe_top_h__weights_renormalize__stop_none/
      gemma4/
        b1_tp1_d4_r4_h4_n4_out128_greedy_graph_mem0p70/
          run_20260921_token_importance_h4/
            readme.md
            config.json
            metrics.json
            provenance.json
            raw/
  02_gdn/
  03_early_exit/
  04_execution/
  reports/
    configuration_index.csv
    study_index.md
  .sources/
  .archive/
    configuration_catalog.json
    relocation_manifest.json
    validation.json
```

- `01_moe`：专家选择、固定 top-h、阈值、权重模式、专家池。
- `02_gdn`：GDN 计算近似、状态表示、精度、共享及窗口策略。
- `03_early_exit`：退出判据、固定/自适应轮数、分支与拒绝诊断。
- `04_execution`：执行路径、批处理、图执行和状态维护成本。
- `00_baselines`：AR、未优化方法、跨方法对照。被多个实验引用时保留唯一来源。
- `.sources`：同一研究共享的数据集、完整矩阵、源码快照和契约；单配置原始产物优先实际放在该运行的 `raw/`。
- `.archive`：迁移、来源、清理和校验元数据，不是未分类结果的堆放处。

## 命名

普通文件使用小写与下划线；保留工具已有的完成标记名。组合的不同机制用 `__` 连接；同项目使用稳定顺序，如 draft、moe、weights、gdn、tail、stop、execution。数值阈值和预算放参数层。确属已有算法名称的 `attention60` 等可沿用，但仍在 config 中记录预算含义。

参数缩写：`b` batch、`tp` 张量并行、`d` draft 长度、`r` 内轮数、`h` 专家数、`p` 专家权重阈值、`top_p` 累计专家质量阈值、`win` 窗口、`a/beta` 门限、`n` 样本数、`out` 输出上限、`ctx` 上下文、`rep` 重复编号、`w` 固定 kernel/forward 宽度、`mem` 引擎显存预算。小数可写 `0p70`，完整精确值保存在 config。

只放与该实验有关的参数，不编造默认值。模型目录为导航简称，精确 checkpoint、规模、dtype 和路径写入 config。不同参数的运行不能因目录简称相同而合并；重名时加稳定来源 ID，不依赖随机编号。

## 记录内容

沿用既有 schema；新项目可采用以下字段，不需要为每个来源伪造缺失字段。

| 文件 | 至少记录的内容 |
| --- | --- |
| config.json | 模型/assistant 身份、完整优化组合、实际生效 spec、样本身份/hash、生成与采样参数、warmup、设备/并行、dtype、eager/graph、内存预算、源码 commit 和 dirty/snapshot 边界 |
| metrics.json | measurement scope、指标值/单位/分母/聚合方式、原始整数计数、对照 ID、完成覆盖、质量/parity 状态 |
| provenance.json | study/run ID、旧路径、当前源路径、SHA-256、selector、原始行号及编号基准、完成证据、生成工具/版本 |
| readme.md | 研究问题、实际配置、对照与结果表、限制、证据链接、已有复现命令 |
| configuration_index.csv | ID、类别、组合、模型、配置名、运行路径、测量类型、完成/有效性状态、原研究名 |
| relocation_manifest.json | 迁移时间与范围、old/new、迁移前后 hash/bytes、复用/移动/链接/生成等动作 |

行号需说明零基还是一基、是否含表头；selector 应足以重建所选子集。整份 JSON 混有多配置时只移动一次，各运行引用该共享文件，不能复制后误认为独立实验。

新结果追加后刷新受影响的父级索引及统计。冻结的历史报告保留原时间范围；当前总索引使用当前统计，不用历史数量冒充全量数量。
