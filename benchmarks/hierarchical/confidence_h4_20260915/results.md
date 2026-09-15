# h4 correction 置信度：独立请求验证

## 结论

两个猜想的方向均得到支持，但都不能当作确定规则。

1. 外层首次拒绝高度集中在 correction 上，但并非只有 correction 会失败。新样本的 82 次首次拒绝中有 65 次发生在 correction，另有 17 次发生在已通过内层验证的 Draft token 或 bonus。
2. Pre-Verify margin 越大，correction 通常越容易被 Target 接受，但高置信仍会错、低置信仍会对。固定的 margin≥2 分组接受率为 77.2%，margin<0.5 分组拒绝率为 81.8%。
3. 更有希望的用途是识别继续内循环的低收益时刻。预设的 correction margin<0.25 规则在新样本中触发 25 次，可跳过 55 个内轮，没有截掉原轨迹中已被 Target 接受的后续候选。这是离线观察，不是实际加速或无损保证。

## 实验与验证边界

- 仅 Gemma-4-26B-A4B-it + assistant，Pre-Verify h4，TP1/B1，greedy，MTP D4，四轮内循环，外层容量 20，max model length 1024。
- 保留原来的四条请求作为 pilot，不参与新样本验证；另选 16 条不重叠请求，HumanEval、Alpaca、GSM8K、UltraFeedback 各四条，每条输出 128 token。
- 按既有数据集顺序选择每类最先满足长度条件的四条非 pilot 请求；最长 prompt 为 304 token。沿用 raw prompt 和 `ignore_eos=True`，未应用 chat template。
- 在运行新样本前固定 margin 分组及停止阈值，见证据包 `heldout/hypotheses.json`。低置信为严格 `<0.5`，高置信为 `>=2`；另有预设的 0.25、1、4 分界。
- 正式采集：同一引擎先完成 16 条 warmup、16 条未采集对照，然后通过 RPC 安装仅在 CPU 读取 logits 的 hook，再运行 16 条测量请求。采集不修改 Draft 采样或 CUDA Graph。
- 16/16 请求、2048/2048 输出 token、全部逐步 accepted/scheduled counters 与同引擎对照一致。所有 Pre-Verify 预测与实际候选、Target logits 与实际接受前缀均通过运行时检查。
- 更早两种在初始化前安装的采集方式未通过跨进程输出对照，已排除，见 `heldout/excluded_attempts.json`。不同启动/采集路径的敏感性未作为生产正确性修复宣称。
- 新样本共 176 个匹配外层 cycle、82 次外层首次拒绝、129 个 Target 到达的 correction。160 个未到达的 correction 排除于局部接受率统计。

概率通过 T=1 softmax 计算，实际生成仍是 greedy。margin 是 Pre-Verify 自己的 Top-1 与 Top-2 logit 差；例如 margin=2 对应约 7.39 的 Top-1/Top-2 概率比，并不等价于 99% 正确率。Target logits 仅用于标签和案例解释，没有用于停止规则的输入特征。

## 1. 外层拒绝是否只发生在 correction？

以下比例均条件于 Target 已到达该 token。

| token 来源 | 到达数量 | 接受 | 首次拒绝 | 接受率 |
| --- | ---: | ---: | ---: | ---: |
| Draft 与 Pre-Verify 已一致通过 | 1627 | 1614 | 13 | 99.20% |
| Pre-Verify correction | 129 | 64 | 65 | 49.61% |
| 内层全收后的 Pre-Verify bonus | 375 | 371 | 4 | 98.93% |

correction 占全部外层首次拒绝的 65/82=79.27%，此前 pilot 为 18/20=90%。另外两类合计接受 1985/2002=99.15%。因此“其他情况近似通过”在此样本成立，“只有 correction 导致外层拒绝”不成立；仍需保留完整 Target 验证。

这里统计的是 Target 验证的 token，不等于最终返回 token：最后一个 cycle 可能超过输出上限。停止规则另行裁剪到实际输出预算。

## 2. correction 的置信度能否预测接受？

| Pre-Verify margin | pilot 接受/到达 | 独立新样本接受/到达 | 新样本接受率 |
| --- | ---: | ---: | ---: |
| [0, 0.25) | 0/7 | 1/17 | 5.88% |
| [0.25, 0.5) | 1/4 | 5/16 | 31.25% |
| [0.5, 1) | 4/6 | 9/21 | 42.86% |
| [1, 2) | 5/9 | 5/18 | 27.78% |
| [2, 4) | 2/4 | 23/34 | 67.65% |
| [4, +∞) | 12/12 | 21/23 | 91.30% |

分组并非严格单调，不能用硬阈值宣称确定性。

预设主分组：

- margin<0.5：新样本接受 6/33，拒绝 27/33=81.82%；pilot 拒绝 10/11=90.91%。
- margin≥2：新样本接受 44/57=77.19%；pilot 接受 14/16=87.50%。
- margin≥4：新样本接受 21/23=91.30%，仍有两个反例；Wilson 95% 区间约 73.2%–97.6%，不足以保证总体接受率超过 90%。

按请求簇重采样的描述性 95% 区间：margin<0.5 接受率约 4.8%–30.4%，margin≥2 接受率约 65.6%–88.4%。使用 2000 次请求级 bootstrap，seed=0。事件不独立，不能把每个 token 当成一个独立实验；16 条请求仍是有限样本。

correction 接受预测的 AUC：margin 在 pilot 为 0.846，新样本为 0.807；新样本的 Top-1 概率 AUC 为 0.820，负熵为 0.802。AUC=0.5 表示没有排序区分度，1 表示完全区分。三个指标都有信息，样本不足以证明其中一个显著更优。

### 高置信分组的语料差异

margin≥2 时，HumanEval 为 18/20=90.0%，GSM8K 为 8/9=88.9%，Alpaca 为 4/6=66.7%，UltraFeedback 为 14/22=63.6%。单一阈值的校准会随语料变化。

## 3. 反例说明了什么？

所有行号均指正式证据 `heldout/h4/trace/distributions.jsonl`。

| 案例 | trace 行 | Pre-Verify token | margin | Pre-Verify 概率 | Target 结果 |
| --- | ---: | --- | ---: | ---: | --- |
| 高置信仍拒绝，Alpaca | 10 | `<eos>` | 5.296875 | 98.09% | 选 `...`；候选概率仅 5.49% |
| 高置信仍拒绝，UltraFeedback | 80 | `single` | 4.375 | 97.12% | 选 `a`；`single` 排第二 |
| 低置信仍接受，HumanEval | 50 | 两个换行 | 0.125 | 52.67% | 接受 correction，但之后 3 个候选零接受 |
| 低置信且后缀全收，UltraFeedback | 31 | `videos` | 0.25 | 55.29% | 接受 correction，其后 7/7 候选也接受 |

高置信错误不只出现在特殊 token：`single` 是普通 token。另一些案例包含 EOS、格式 token 或重复段，符合本次 raw-prompt、忽略 EOS 的诊断协议；不应直接外推到所有聊天请求。

第三行尤其重要：预测“当前 correction 被拒绝”会出错，但预测“继续内循环没有新增有效 token”仍然正确。因此用于提前停止的最终标签应该是后续有效 yield，而不是只看当前 correction 接受与否。

反方向也成立：当前 correction 的置信度高，不保证未来内轮的 correction 都正确。例如 margin≥4 的有后缀事件有 20 个，仅 13 个后缀全收。

## 4. 离线提前停止：比接受长度阈值更细吗？

固定规则：前三轮中首次满足条件时，在该轮 correction 已追加后停止，跳过后续内轮。所有特征来自已经完成的内层计算。

| 规则 | 触发 cycle | 可跳过内轮 | 后续原本零接受的触发 | 截掉的原本已接受后缀 token |
| --- | ---: | ---: | ---: | ---: |
| 内层 accepted≤1 | 84 | 219 | 61/84 | 142 |
| 任意 correction | 90 | 251 | 63/90 | 189 |
| correction margin<0.25 | 25 | 55 | 25/25 | 0 |
| correction margin<0.5 | 40 | 94 | 37/40 | 27 |
| correction margin<1 | 61 | 144 | 54/61 | 51 |
| correction margin<2 | 73 | 181 | 62/73 | 83 |

计数基于 176 个完整匹配 cycle，共 704 个内轮。margin<0.25 的 55 轮约占其中 7.8%；这些数字不是端到端加速比。

margin<0.25 的 25 次触发来自 8 条请求，触发于第 1/2/3 轮分别为 11/8/6 次。其中 10 次的 Target 已经在更早位置拒绝：信号到达较晚，但仍可识别原轨迹中剩余内轮的浪费。不能把这些未到达的 correction 当作局部拒绝样本。

所有截掉后缀的计数经 128-token 输出上限裁剪后相同。它们不是最终少输出的 token 数；真正提前进入 Target 会改变验证长度、bonus 和后续 cycle。25 次零损失观察仍不能保证新输入上无损，尤其事件集中在八条请求。

当前更合理的候选是“极低 margin 时提前交给 Target”，而不是“高 margin 时视为已通过”。若实施在线策略，仍需完整 Target 验证、输出一致性门槛和完整 proposal→Target cycle 的性能测量。

## 产物与复现

证据包包括 `heldout/` 和 `pilot/`。主要文件：

- `heldout/confidence_analysis/events.csv/json`：逐位置来源、Pre-Verify margin、Target 到达/接受和后缀标签。
- `margin_groups`、`origin_groups`、`category_groups`：置信度分组及语料拆分。
- `stop_policies.csv/json`：各规则汇总及逐次触发明细。
- `examples.json`：上述反例的 token ID、概率、上下文与原始行号。
- `heldout/control_comparisons.json`、`instrumentation_parity.json`：同引擎对照。
- `heldout/hypotheses.json`、`contract.json`：固定阈值、独立样本选择和指纹。
- `heldout/excluded_attempts.json`：未通过对照的早期采集边界。

```bash
.venv/bin/python benchmarks/hierarchical/run_disagreement.py <fresh_directory> \
  --gpu 1 --confidence-only --dataset <heldout/dataset.jsonl>
.venv/bin/python benchmarks/hierarchical/analyze_confidence.py \
  <pilot_directory> <fresh_directory>
.venv/bin/python -m pytest tests/benchmarks/test_hierarchical_measurement.py -q
```

测试为 13 passed；模型运行及分析完成。仅修改 benchmark 与检查代码，没有实施生产提前停止策略。已有 Gemma AR/speculative 严格一致性限制未解决，本实验只证明同引擎有无采集的输出一致。AI assistance was used.
