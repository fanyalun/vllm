# h4 内外层首次拒绝的分布诊断

## 结论

本次 h4 样本中，内层分布分歧大，不等于后续外层一定拒绝。相较于分布距离本身，双方是否把对方首选排在前二名，是一个值得继续验证的候选特征；它仍不是经过独立验证的提前停止规则。

严格按预先定义的“分布近似”分组，本样本没有近似内层拒绝事件，无法估计这一组的后缀接受率。外层首次拒绝则同时存在近似分歧和明显不一致。

## 范围与验证

- Gemma-4-26B-A4B-it 与 assistant；Pre-Verify h4；TP1/B1；greedy；MTP D4、四轮、外层容量 20；每条输出 128 token。
- HumanEval、Alpaca、GSM8K、UltraFeedback 各一条正式请求；四条 warmup 排除。
- 46 个匹配的外层 cycle，77 个内层首次拒绝事件，20 个外层首次拒绝事件。
- 四条输出的全部 512 token，以及逐步 accepted/scheduled counters，与本次未采集 logits 的 h4 对照完全一致。
- 运行时检查：Draft logits argmax 对齐真实 Draft token；Pre-Verify logits argmax 对齐真实输出；拼接候选对齐各轮 Pre-Verify；Target logits 对齐外层接受前缀。
- 一次改变采集实现的 h4 复测未通过对照，已排除，见 `excluded_repeat.json`。当前报告使用原始通过对照的 h4 采集，不合并失败版本，也不包含其他 h 配置。
- 本次没有评估提前停止性能，没有修改生产解码逻辑。现有 Gemma AR/speculative 严格一致性限制仍适用。

## 指标定义

实际生成使用 greedy。为比较 logits 分布，另外对完整有效词表 logits 做 T=1 softmax；这些概率不是本次运行的随机接受概率。保留 Draft 的原生词表掩码。

记录双方 Top-8 token、logits、概率、熵、对方 Top-1 的排名和 logit 差，以及全词表 Jensen–Shannon divergence（JS，单位 nat）、total variation（TV）和双向 KL。JS 的理论上限是 ln(2)，约 0.693；接近此值意味着两边分布高度分离。

预定义分组：

- **近似分歧**：双方首选在对方分布中排名均不超过 2，且双方对该候选的 logit 差均不超过 1。
- **明显不一致**：任一方把对方首选排到第 8 名之后，或 logit 差超过 2。
- **居中**：其余情况。

排名采用 `1 + 严格高于该 token 的 logits 数量`，包含并列名次。Top-k 返回顺序不能用来判定并列时的实际 argmax；案例按实际 token 及最小 token ID 的 argmax 规则解读。

“后缀”指 Pre-Verify 修正后实际形成的后续候选，不是已被丢弃的原始 Draft 后缀。仅当 Target 到达本轮 correction 位置时，才判断它是否接受修正及后缀。提前被 Target 拒绝的事件标记为未到达。

## 1. 内层首次拒绝：分布相近还是明显不一致？

| 类别 | 内层事件 | Target 到达修正位置 | 接受修正 token | 有后缀的到达事件 | 后缀全收 | 后缀零接受 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 近似分歧 | 0 | 0 | 无样本 | 0 | 无样本 | 无样本 |
| 居中 | 1 | 1 | 0/1 | 1 | 0/1 | 1/1 |
| 明显不一致 | 76 | 41 | 24/41 | 38 | 9/38 | 22/38 |

明显不一致组另有 7/38 个部分接受后缀的事件。35 个未到达事件没有被当作该处修正错误。

近似阈值收紧到 0.25、0.5 后也没有样本。全部内层拒绝的 JS 范围是 0.2565–0.6930。因此本次不能对真正“分布很接近”的内层分歧作统计推断。

只取每个外层 cycle 最早发生内层拒绝的一轮，仍有 26 个明显不一致事件，其中 Target 到达 24 个、接受修正 15 个；有后缀的 23 个事件中，5 个后缀全收、13 个零接受。不能把所有内轮重复计数当作独立样本。

## 2. 排名相近与分布相近不是同一个条件

有 14/77 个事件中双方首选互在对方前二名，但它们全部属于上表的明显不一致组：至少有一方对另一个候选的概率非常低。

| 排名关系 | 事件数 | Target 到达 | 修正 token 接受 | 后缀接受 token / 送验 token | 后缀全收事件 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 双方首选互在对方前二名 | 14 | 8 | 7/8 = 87.5% | 67.6% | 2/6 |
| 其他排名关系 | 63 | 34 | 17/34 = 50.0% | 25.5% | 7/33 |

这提示交叉排名值得作为下一轮实验的候选信号。但上述结果仅有 8 个到达的互为前二名事件，属于本次样本内观察；后缀比例的分母为有非空后缀的到达事件中的 token 总数，不是事件比例。

## 3. 具体案例

所有行号均指 `h4/trace/distributions.jsonl`，仅包含正式请求。

### 案例 A：内层明显分歧，外层立刻拒绝

第 1 行，HumanEval，第 1 内轮，接受 0 个 Draft token：

- Draft 首选 `numbers`，概率 94.74%；把 Pre-Verify 首选 `for` 排第 31。
- Pre-Verify 在 `for` 和 `#` 上精确并列，概率各 41.00%；实际选 `for`，把 `numbers` 排第 118。
- Draft/Pre-Verify JS = 0.6854。
- 外层接受 0/16，修正后的 15-token 后缀全部没有被接受。

同一个位置反向看，Pre-Verify/Target 的分布却很近：Target 选 `#`（39.70%），`for` 为 27.29%，JS 仅 0.02493。内层的大分歧与外层的小分歧可以发生在同一个位置。

### 案例 B：内层分布几乎分离，修正后外层全部接受

第 17 行，HumanEval，第 3 内轮，先接受 1 个 Draft token，修正位置为外层候选 offset 11：

- Draft 选 `logic`，概率 99.97%；`test` 排第二，但概率仅 0.0261%。
- Pre-Verify 选 `test`，概率 99.40%；把 `logic` 排第 5。
- 双方对对方首选的 logit 差分别为 8.25、8.8125，JS = 0.6912。
- Target 接受整个候选 17/17，包括 `test` 和其后全部 5 个 token。

这是“大分歧应该直接放弃后续循环”的明确反例。此处 Pre-Verify 对 Draft 做出了有效修正。

### 案例 C：Top-2 排名翻转，但不是小概率差

第 2 行，HumanEval，第 2 内轮：

- Draft 选 `are`，概率约 99.98%；`in` 排第二，logit 差 8.75。
- Pre-Verify 选 `in`，概率 51.72%；`are` 排第二，概率 40.28%，logit 差 0.25。
- 双方首选互为第二名，但两边分布不是轻微扰动；JS = 0.2719。
- Target 接受修正，随后接受 3/5 个后缀 token；整个外层 cycle 接受 9/11。

## 4. 反向检查外层首次拒绝

| Pre-Verify / Target 类别 | 首次拒绝次数 | 比例 | JS 中位数 |
| --- | ---: | ---: | ---: |
| 近似分歧 | 5 | 25% | 0.0596 |
| 居中 | 5 | 25% | 0.1468 |
| 明显不一致 | 10 | 50% | 0.3844 |

20 次中有 9 次双方首选互在对方前二名，2 次至少一方最高两个 logits 精确相等。Top-2 排名翻转不等于低 margin，也不保证完整分布近似。

外层明显不一致的例子：第 7 行、候选 offset 5，Pre-Verify 选 `ly`，Target 选 `//`。前者把 `//` 排第 1133，后者把 `ly` 排第 19，JS = 0.4431。

## 5. 哪类候选最容易成为外层首次拒绝点？

| token 来源 | Target 到达的 token 数 | 此处首次拒绝 | 到达后的拒绝比例 |
| --- | ---: | ---: | ---: |
| Draft 与 Pre-Verify 已一致通过 | 405 | 1 | 0.25% |
| 内层拒绝后 Pre-Verify correction | 42 | 18 | 42.86% |
| 内层全收后 Pre-Verify bonus | 95 | 1 | 1.05% |

20 次外层首次拒绝中，18 次落在 correction 上。这个现象比“第一轮接受少”更具体：可继续检验 correction 的累计数量、交叉排名、双方对 correction 的概率和 Pre-Verify margin 是否能预测继续内循环的收益。

这些是条件于 Target 已到达的描述性比例；在线决策尚不知道 Target 是否能到达。下一步必须按请求留出验证，并评价完整 cycle 的实际 token yield/耗时。不能直接将这些比例当作在线概率，或据此宣布已有可靠停止规则。

## 产物与复现

- `inner_rejections.csv/json`：逐内层首次拒绝、分布指标及后缀标签。
- `outer_rejections.csv/json`：逐外层首次拒绝及分布指标。
- `inner_summary.csv/json`、`rank_summary.csv/json`、`position_summary.csv/json`：分组统计。
- `near_sensitivity.csv/json`：预定义近似阈值敏感性。
- `instrumentation_parity.json`、`audit.json`：输出一致性和逐 cycle 对齐。
- `contract.json`、`capture_sources/`：数据及代码指纹、实际采集源码快照。
- `excluded_repeat.json`：未通过对照的复测边界。

分析命令：`.venv/bin/python benchmarks/hierarchical/analyze_disagreement.py benchmark_results/gemma_h4_disagreement_4x128_20260915`。

采集命令：`.venv/bin/python benchmarks/hierarchical/run_disagreement.py <fresh_directory> --gpu 1`。采集默认只运行 h4 及其对照；分析必须通过 token/逐步计数一致性门槛。

验证：`tests/benchmarks/test_hierarchical_measurement.py` 11 passed；新增脚本通过 Ruff。所有报告均为诊断，不包含性能宣称。AI assistance was used.
