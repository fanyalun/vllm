# h4 三档停止策略性能矩阵

三档均接入 `method="hierarchical"`，默认 `hierarchical_stop_policy="low_error"`：

| 配置值 | 当前 correction 的停止条件 |
| --- | --- |
| `low_error` | 本轮接受长度 L=0 时 M<2，否则 M<0.25 |
| `balanced` | M<1 |
| `aggressive` | M<2 |

M 是 Pre-Verify 自身 Top-1/Top-2 logit margin。保留当前 correction 后提前结束；最后一轮没有可跳过的内轮。`none` 保留固定四轮作为诊断控制，不在本次用户指定的五方案矩阵中。

## 范围

- Gemma-4-26B-A4B-it + assistant，Pre-Verify h4，MTP D4，最多四轮，外层最大候选容量 20。
- 同一份 16 条请求，来自上一轮独立请求集；raw prompt，不套 chat template。
- 输出长度 512，greedy、ignore_eos=True；batch size 1、4、8、16，按固定分组提交。
- 五方案：AR、原生 MTP D4、三级 low_error、balanced、aggressive，总计 20 个配置。
- TP1、同步调度、关闭 prefix caching、关闭图像和视频输入、max model length 1024、max batched tokens 4096、GPU memory utilization 0.90，各方案一致。
- 多请求支持限 Gemma4 MTP。已停止的请求从后续内轮中移出，保留原 request-state 索引；每个请求独立返回实际候选长度。压缩后的 MTP prefill 使用新建的 eager attention metadata，MTP decode 和 Pre-Verify 支持 CUDA Graph。
- 每个配置先跑 16×64 warmup；正式测量若新增 Pre-Verify CUDA Graph，则保留该次数据并重测，最多三次。启动、编译、warmup 不计入吞吐。
- 使用实验目录下按方案隔离的编译缓存，避免旧 AOT 缓存中 raw token IDs / embeddings 签名混用。

## 运行

```bash
.venv/bin/python -m benchmarks.hierarchical.run_policy_matrix \
  benchmark_results/gemma_h4_policy_smoke --smoke
.venv/bin/python -m benchmarks.hierarchical.run_policy_matrix \
  benchmark_results/gemma_h4_policy_16x512
.venv/bin/python -m benchmarks.hierarchical.summarize_policy_matrix \
  benchmark_results/gemma_h4_policy_16x512
```

`--smoke` 仅运行 B4/B16 的 low_error，16×32 输出。启动器等待 GPU 连续空闲 30 秒；`--wait-pid PID` 可先等待既有实验队列结束，避免把队列切换时的短暂空闲当成可用资源。不会终止既有进程。

## 统计与边界

- 吞吐：8192 / 各固定批次 generate 耗时之和；分别报告相对同 batch AR 和 MTP 的比值。
- 外层接受率：接受的候选 token 总数 / 实际验证的候选总数；另报平均接受 Draft 长度，以及含 Target bonus 的平均长度。
- 内层接受率：Pre-Verify 接受的 Draft token / Draft 提出的 token，统计实际执行的内轮。
- 内轮次数与跳过次数以“请求×内轮”计；batch_round_calls 为真实批量 Pre-Verify 调用次数。两者不能互换成耗时节省。
- engine_steps 为 Target runner 调用次数，包含 prefill；outer_request_steps 为逐请求投机验证次数，不是批量 Target 调用次数。
- 记录每条输出与同 batch AR 的首次差异。输出一致性、运行计数完整性和吞吐分开报告，不能用高接受率证明严格等价。
- 策略改变后续轨迹，不能从本次实际停止计数直接推断离线意义下的误停率；需要另做反事实诊断。
- 每配置当前计划一次满足热图覆盖条件的正式测量，小样本性能差异不等于稳定部署收益。

每个配置保存输出、详细接受计数、策略计数和耗时。完整矩阵结束后才生成审计完成标记。AI assistance was used for implementation, tests, and experiment preparation.

正式启动器跑完后自动执行汇总、生成 report/results.md、PNG/PDF 图及证据包。
出现失败时保留日志并退出，不生成完成标记；自动生成的图仍需人工目视检查。
