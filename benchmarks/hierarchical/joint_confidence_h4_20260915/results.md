# h4 接受长度与 correction margin 联合规则

## 结论

当前内轮接受长度 L 与 correction margin M 的 AND 规则，相比单独长度有明显改善，但未优于保守的单独 M<0.25 规则。L=0 可以给较宽 margin 阈值增加筛选能力；L≤1 的增量很有限。不能仅以误停率下降判断更好，必须同时看跳过内轮数和截掉的有效 suffix。

## 口径与复现

- 复用已通过采集一致性审计的 h4 pilot 4 条、独立请求集 16 条，每条 128 tokens。此次没有新增 GPU 运行。
- Gemma、MTP D4、四轮内循环、B1、greedy、raw prompt、ignore_eos=True，沿用原实验配置。
- L 是当前内轮 Draft 被 Pre-Verify 接受的 token 数，不含 correction，不是累计长度或第一轮长度。M 是当前 correction 位置 Pre-Verify 自己的 Top-1/Top-2 logit 差。
- 每个外循环在前 3 个内轮中找第一个同时满足条件的 correction；保留当前 correction，跳过后续内轮。最后一轮不能产生节省，排除。
- 所有规则使用相同轨迹和首触发口径；逐事件决策见 decisions.csv，完整 25 组规则分别在两个数据集计算，见 policies.csv。
- suffix_cut 是原轨迹中 correction 之后已接受的 token，裁剪到输出 128 上限；不是最终输出丢失数，也不是端到端速度。局部 correction 标签只使用 Target 已到达的位置，另见 local_corrections.csv。
- 本次联合网格是在看过上一轮结果后新增的探索分析。16 条请求相对 pilot 独立，但不再是这次联合规则选择的全新盲测集。
- 读取时重新检查采集输出一致性、逐步计数与位置对齐；单独规则与原分析函数逐 cycle 比对一致。trace 与分析源码指纹见 audit.json。

```bash
.venv/bin/python -m benchmarks.hierarchical.analyze_joint_confidence \
  benchmark_results/gemma_h4_disagreement_4x128_20260915 \
  benchmark_results/gemma_h4_confidence_late_16x128_20260915 \
  benchmarks/hierarchical/joint_confidence_h4_20260915
.venv/bin/python -m pytest tests/benchmarks/test_hierarchical_measurement.py -q
```

输入证据归档在相邻目录 `../confidence_h4_20260915/raw_evidence.tar.gz`，解包后的 pilot/ 与 heldout/ 可代替上述两个输入路径。AI assistance was used for analysis, implementation, and report preparation.

## 16 条请求：176 个外循环

所有行都要求当前存在 correction；D4 中 L≤1 自然意味着 correction。

| 规则 | 触发次数 | 跳过内轮 | 零有效 suffix 次数 | 误停次数 | 截掉有效 suffix tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| L≤1 | 84 | 219 | 61 | 23 | 142 |
| M<0.25 | 25 | 55 | 25 | 0 | 0 |
| L≤1 AND M<0.25 | 21 | 44 | 21 | 0 | 0 |
| L=0 AND M<0.25 | 17 | 33 | 17 | 0 | 0 |
| M<0.5 | 40 | 94 | 37 | 3 | 27 |
| L≤1 AND M<0.5 | 36 | 80 | 33 | 3 | 26 |
| L=0 AND M<0.5 | 32 | 69 | 30 | 2 | 16 |
| M<1 | 61 | 144 | 54 | 7 | 51 |
| L≤1 AND M<1 | 58 | 130 | 51 | 7 | 48 |
| L=0 AND M<1 | 47 | 100 | 44 | 3 | 31 |

M<0.25 时，加 L≤1 少跳过 11 轮，加 L=0 少跳过 22 轮；三者都没有观察到截掉有效 suffix，联合规则没有提供额外安全性证据。单独 M<0.25 在观测的节省/截断两项指标上占优。

M<0.5 时，加 L≤1 只把截断 27 降至 26，误停仍为 3 次，跳过内轮却从 94 降至 80。加 L=0 则从 27 降至 16（减少 40.7%），误停从 3 降至 2，跳过内轮从 94 降至 69（减少 26.6%）。这是一种收益与风险取舍，不是全面占优。

M<1 时，加 L=0 的零 suffix 比例从 54/61=88.5% 升至 44/47=93.6%，但跳过内轮从 144 降至 100，截断从 51 降至 31。相比单独 M<0.5，它多跳过 6 轮，也多截掉 4 个 token，仍无确定净收益结论。

## 局部 correction 拒绝预测

以下统计全部内轮中 Target 已到达的 correction，不等同于前表每 cycle 首触发。未到达的 token 不标记为拒绝。

| 规则 | 拒绝 / 到达 | 拒绝率 |
| --- | ---: | ---: |
| M<0.25 | 16/17 | 94.1% |
| L≤1 AND M<0.25 | 14/15 | 93.3% |
| L=0 AND M<0.25 | 11/11 | 100% |
| M<0.5 | 27/33 | 81.8% |
| L≤1 AND M<0.5 | 23/28 | 82.1% |
| L=0 AND M<0.5 | 20/23 | 87.0% |

L=0 的局部筛选有信号，但样本更少。把 16/17 变为 11/11 并不自动改善停止决策：被筛掉的低 margin correction 即使本身接受，其后续 suffix 也可能完全没有收益。

## Pilot 对照及限制

pilot 中单独 M<0.25 跳过 16 轮、零截断；加 L≤1 后 15 轮、零截断，加 L=0 后 11 轮、零截断。单独 M<0.5 跳过 24 轮、截断 3 tokens；加 L=0 后跳过 16 轮、仍截断 3 tokens。较宽 margin 下 L=0 的改善没有在小 pilot 中稳定重现。

16 请求集中 M<0.25 的 25 次触发来自 8 条请求，其中 10 次 Target 实际已在更早位置拒绝。联合 L≤1 的 21 次来自 8 条请求，8 次已更早拒绝；联合 L=0 的 17 次来自 7 条请求，7 次已更早拒绝。此类信号可以提示继续计算的低收益，但不能据此宣称当前 correction 都会被拒绝。

下一步应固定候选规则后，在新请求上做真实自适应运行，比较完整输出、内轮调用、Target 调用与耗时。当前支持保留 M<0.25 为保守基线，L=0 AND M<0.5 为较积极候选；尚无理由用 L≤1 AND M<0.25 替换基线。
