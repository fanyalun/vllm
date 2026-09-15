# ReplaySSM 双 checkpoint 实现与验证

## 实现状态

- 新开关：`replayssm_spec_dual_checkpoint=True`，默认关闭。
- 每层两个 FP32 checkpoint；按上一轮实际长度确认全接受并切换槽位。
- 拒绝后仅提交已接受输入前缀；`H+T>W` 时在本轮写入前 flush，等于上限不 flush。
- prefill 和首次 decode 重置元数据；支持固定地址 CUDA Graph、变长窗口、请求重排和 block 复用。
- 模型级页大小估算与层级实际分配均包含新增 checkpoint。

## 验证

**47 项测试通过，全部适用 pre-commit hooks 通过。** 原始输出见 `tests.log` 与 `precommit.log`。

```bash
.venv/bin/python -m pytest \
  tests/config/test_replayssm_dual_checkpoint.py \
  tests/config/test_replayssm_flush_interval.py \
  tests/v1/worker/test_gdn_dual_checkpoint_metadata.py \
  tests/kernels/test_replayssm_dual_checkpoint_gdn.py \
  tests/kernels/test_replayssm_flush_interval.py -q
```

新 kernel 的测试覆盖 FP32/BF16、实际窗口长度 1/2/3/4/5/7/17 及长窗口的所有接受前缀、连续全接受/拒绝、每轮变长、容量边界、环形回绕、prefill 重置、padding 和 eager/graph。旧路径回归覆盖 D=4/8/16/32 与五种 flush interval。

整模型使用本地 Qwen3.6-35B-A3B、TP1、两条固定 prompt、每条 64 tokens；V1 使用 MTP，V2 使用 DSpark。`*_verified.json` 记录当前核心源码哈希，重复运行均为 2/2 输出一致。

| 检查 | 结果 |
| --- | --- |
| V1 新模式 eager 与最终 graph | 2/2 逐 token 一致 |
| V1 新模式重复请求 | 2/2 逐 token 一致 |
| V2 新模式 graph 重复请求 | 2/2 逐 token 一致 |
| V1 原版与新模式 | 1/2 一致；另一条第 36 个 token 分歧 |
| V2 新模式 eager 与 graph | 1/2 一致，未建立跨模式逐 token 等价 |
| V2 原版重复请求 | 0/2 一致，出现异常重复文本；性能基线无效 |

V1 第一个分歧位置，新模式的两个候选 logprob 均为 -1.2617556；原版为 -1.1672983 与 -1.2922983。该现象与浮点重排触发 argmax 变化相符，但不能据此宣称完整模型逐 token 等价。原版 V2 的重复请求失败与其缺少首次 decode 的历史重置信息一致；未用这项失败结果计算加速比。

## 受控 kernel 性能

A100、B=1、D=4、窗口配置值 16、Qwen GDN head shape、CUDA Graph 计时。数值是单层完整 cursor commit + flush + verify/tail 周期。原版配置 16 对应逻辑 L=21、物理 ring=32；新模式对应硬上限 W=16、物理 ring=16。

| 接受轨迹 | 原版 μs/轮 | 双 checkpoint μs/轮 |
| --- | ---: | ---: |
| 全接受 | 18.80 | 31.00 |
| 零草稿接受 | 18.25 | 31.34 |
| 交替全接受与拒绝 | 18.60 | 31.02 |

**当前实现没有取得 kernel 加速。** 全接受时新模式历史长度和 flush 率降为零，但每轮生成尾 state 的成本仍高于节省的重放成本。E2E smoke 的计时仅作运行记录，样本量、非交错测量、输出分歧和失效的 V2 原版基线均不支持 serving 加速结论。

单层 state+ring 存储从 2,494,464 bytes 变为 4,392,960 bytes，不包含 Conv 与混合 attention 页对齐。一个新增 FP32 checkpoint 为每层每请求 2 MiB，实际净增量受 ring 缩小影响。

## 复现与边界

- `manifest.json` 记录基线提交、当前分支、解释器与源码哈希；`summary.json` 汇总指标及限制。
- `benchmarks/replayssm/dual_checkpoint_kernel.py --output <json>` 复现三种接受轨迹的 kernel 对照。
- 整模型：使用 `.venv/bin/python benchmarks/replayssm/dual_checkpoint_smoke.py --dual --output <json>`；设置 `VLLM_USE_V2_MODEL_RUNNER=0` 使用 MTP，或设为 `1` 并加 `--method dspark`。`--eager` 关闭 graph。
- 本次使用已有环境，无新增依赖。运行时将 `.venv/bin` 放到 PATH 前部，并设置 HF 离线变量。
- `*_verified.json` 是最终核心源码验证记录；其余 cell 是开发过程中的有界检查记录。
- DSpark 置信度控制、TP/EP 扩展测试、长上下文与 serving 规模 benchmark 留待后续；新模式默认关闭。
