# 三级自投机解码相对 MTP 的性能模型

本文件给出从完整周期成本推导的模型和待执行测量合同。公式是本次推导，
不是 GPU 实测结果。2026-09-21 核对的实现包含 MTP、近似 Pre-Verify、
最终精确 Target 三个串行阶段。

## 1. 必须分别测量的变量

对一个外层周期，令：

| 变量 | 定义 |
| --- | --- |
| B | 该周期真实活跃请求数 |
| d | 每轮内层 MTP 提出的候选数，当前常用值 4 |
| R | 实际执行的内循环轮数，不是配置上限 |
| K_r | 第 r 轮被 Pre-Verify 接受的 MTP 连续前缀长度，0 至 d |
| L | 交付 Target 的候选数；未截断时为 sum(K_r+1) |
| A | Target 接受的连续候选前缀长度，0 至 L |
| Y | 最终返回的 token 数；普通未截断周期为 A+1 |
| C | 同一周期完整墙钟成本，毫秒 |

K_r+1 包含 Pre-Verify 的 correction/bonus，但这些仍只是外层候选。
A 不包含 Target 的 correction/bonus，Y 包含它。
EOS、输出预算截断、末尾未验证 proposal、prefill 单独记录；不能把
所有周期的 Y 都强制记成 A+1。

两个接受率分别为 sum(K_r)/sum(d_r) 和 sum(A)/sum(L)。
平均接受长度 E[A]、平均交付长度 E[L]、按位置的连续存活率也必须报告。
按窗口平均 A/L 与 sum(A)/sum(L) 不等价，后者为主指标。

## 2. 相对 MTP 的必要且充分成本条件

在固定工作负载、相同输出语义和稳定周期统计下，令 MTP 基线
q_2=E[C_2]/E[Y_2]，单位 ms/token。三级的 q_3=E[C_3]/E[Y_3]。

    S_3/2 = q_2 / q_3
          = E[Y_3] E[C_2] / (E[Y_2] E[C_3])

    三级加速 <=> E[C_3] < q_2 E[Y_3]

实际数据使用 sum(Y)/sum(C)，不能平均逐周期 Y/C，也不能用两种方法
不匹配的计时范围。完整生成的最终判据是 returned_tokens/wall_time；
周期模型用于解释和预测，需另报其与完整生成的误差。

对 B>1，以一次批次迭代的总 Y=sum_b Y_b 除以批次墙钟 C；
同一个批次时间只能计一次，不能把共享事件时间按请求相加。

## 3. Pre-Verify 的可用时间预算

串行执行的三级周期分解为：

    C_3 = I + sum_r (M_r + P_r + H_r) + V(B,L+1) + O

I 是从 canonical Target 状态初始化私有状态的成本；M 是 MTP 草稿；
P 是 Pre-Verify；H 是内层采样、状态维护、控制和元数据；V 是最终
Target forward；O 是外层采样、提交和其他未重叠成本。
proposal 是嵌套区间，不能再与其内部 M/P/H 相加。

记 C_nonP=C_3-sum_r P_r，则允许的 Pre-Verify 总预算为：

    P_budget = q_2 E[Y_3] - E[C_nonP]

若 P_budget<=0，即使 Pre-Verify 免费，固定该策略和接受分布也无法加速。
若只把 Pre-Verify 优化 s 倍且其他成本与输出分布不变，则：

    s_required = E[sum_r P_r] / P_budget
    加速条件为 s > s_required，且 P_budget > 0

恰好等于阈值只是持平。要求至少 g 倍整体加速时，将预算中的 q_2
换成 q_2/g。优化若改变接受分布，必须重新计算，不能沿用旧预算。

固定 R、每轮近似等成本时，令 t_T 为同宽精确 forward 时间，
s=t_T/t_P，m 为每轮 Draft 成本，h 为每轮维护成本，i 为初始化，
o 为外层额外成本，v(L) 为最终长窗口 Target 成本：

    C_3 ≈ i + R(m+h+t_T/s) + v(L) + o
    s > R t_T / [q_2 (E[A]+1) - i - R(m+h) - v(L) - o]

分母必须为正。随机 R 时使用实测逐轮求和，不能随意以
E[R] E[P] 代替 E[sum P_r]；实际活跃 batch、width 与停止事件相关。

## 4. 接受率应达到多少

固定交付长度 L、无输出截断时，ρ=E[A]/L：

    ρ > (E[C_3]/q_2 - 1) / L

若右侧>=1，该成本下即使全接受也不可能严格加速。
随机长度时仍有 E[A]=ρ_weighted E[L]，但成本必须对实际 L 的分布求均值。
目标加速 g 的最低接受率是 (g E[C_3]/q_2-1)/E[L]。

令 s_j=P(A>=j)，则固定 L 时 E[A]=sum_{j=1}^L s_j。
随机 L 时 E[A]=sum_{j>=1}P(L>=j,A>=j)。后者保留停止与接受的相关性。
逐 token 独立且匹配概率为 α 的简化假设下：

    E[A] = α(1-α^L)/(1-α)

真实模型无需满足独立假设；正式建模使用经验连续前缀分布。
token 平均匹配率不能代替连续接受率。长草稿末端的价值取决于前面的
全部候选仍被接受的概率。

## 5. 为什么 balanced 不一定最大化速度

当前 should_stop_inner 在发生拒绝且 logits top1-top2 margin<1 时停止。
这是误差信号，不包含 MTP/Pre-Verify/Target 的时间成本。

对已观察到的历史 h，设立即停止的预计输出和成本为 y(h)、c(h)；
继续一轮并停止的增量为 Δy(h)、Δc(h)。最大化总体长期吞吐的
最优停止问题，可用平均收益形式求解：

    选择使 E[Y - λ C] 最大的停止策略
    在最优 λ=E[Y]/E[C] 时，最优策略的 E[Y-λC]=0

一步前瞻的继续条件为 Δy(h)>λ Δc(h)，其中 Δc 包含新增 Draft、
Pre-Verify、维护，以及更长 Target verify 的边际成本。
完整最优策略要比较停止与继续后的最优未来价值，不能把一步规则
宣称为全局最优。以 λ=1/q_2 可筛选相对 MTP 是否有正收益。

若前缀长度为 l，增加 m 个候选的收益近似为
sum_{j=l+1}^{l+m} P(A>=j | h)，而不是 m 乘一个全局平均接受率。
估计应使用独立校准数据，并在留出请求上评价；同一数据选择阈值再
报告最优收益会乐观偏置。第一阶段先测现有 balanced，不改变策略。

## 6. 本次测量合同与实现缺口

共同候选配置：Qwen3.6-35B-A3B、TP1、greedy、原生 top8 候选中
p=0.125、preserve 权重；V2 使用 replay_tail/windowed_three_level、
window_size=1、alpha=0.95、beta=0.36328125、私有 SSM carry。
模型/Conv dtype 与 FP32 SSM、GPU、prompt、prefix 长度、commit、
工作树指纹和采样参数均需写入 manifest。

用户已明确：单项指标可以采用两级解码，用 MoE-Skip 作为 Draft。
实验 1 因此优先采用 MoE-Skip+GDN V2 自回归生成 32 个候选，再由
原模型验证；不执行内层 MTP。该实验估计近似 Draft 的接受质量，
其串行起草成本不能用作三级方案中批量 Pre-Verify 的成本。
此外，两级单 token 起草没有内层拒绝后 SSM carry 的误差来源，
其接受分布不能未经校验就当作三级 carry 路径的接受分布。
实验 2 按用户确认保留三级原 balanced，最大交付 D=32。
两者记录逐周期 L/A/Y/R、逐轮 K/margin、停止原因和完整成本。
交付 A 和 L 的 0..32 直方图、A 的 survival 曲线、全接受率、零接受率、
P50/P90/P95 和样本数；内层接受率单列。
首尾截断窗口不得悄悄混入固定 D=32 的条件分布，但必须计入完整吞吐。

当前配置强制外层容量等于 inner_num_rounds*(d+1)。d=4 时设置
8 轮得到容量 40，而且拒绝会使实际交付长度变化；它不是固定 D=32。
要做固定 D=32，需要独立候选上限与剩余长度控制，且零接受时
可能需要 32 轮。单纯把轮数设置成 8 不满足实验 1。
两级实验 1 不需要这个三级控制器变化。若实验 2 保留三级路径，
仍需处理该接口；现有 windowed GDN 还限制内层 D=4 且轮数<=4。
当前 method=moe_skip 的配置校验不接受 preverify_gdn_mode，
因此不能只在命令行添加 V2 字段就声称已经启用。

实验 3：B=1/8/32/64，每请求候选 d=1..32，四个变体：
原模型、仅 p=0.125、仅 GDN V2、两者组合，共 512 个配置格。
输入实际宽度为 d+1（anchor 加候选），明确记录而不把 d 当总行数。
各变体用相同权重、输入、prefix 和独立恢复的初始状态；分别报告
完整 forward+LM head、私有状态初始化/维护，以及最终原模型 verify。
同宽 exact 私有 forward 可作为消融控制，但不能冒充实际 Target
runner 的 V(B,d+1)；两条路径需要分别标识和测量。
内层 d=4、外层 D=32 时，周期使用 P(B,5) 和 V(B,33)，不是 P(B,33)。

CUDA Graph 与 eager 分开，不混合比较；warmup 后计时，重复顺序交错，
状态恢复放在纯 forward 计时外，并计入完整周期。探针运行与正式
wall-clock 运行分开，核验输出一致。近似模型无需和 exact logits 相同，
但各自 graph/eager、一致初态下的重复输出、Target 隔离必须验证。
与同条件 AR 的最终 token 差异单独报告，未通过不能称为无损加速。

用户已明确要求保持完整 forward，放不下记容量受限，不使用逐层和替代。
大 batch 必须记录真实活跃请求数、显存占用和容量失败。B1 重复执行
不是 B64。不能为填表擅自量化或改 TP。
GPU 被其他任务占用时，不能把争用条件的 latency 当正式比较结果。

## 7. 当前状态

模型推导和源码路径检查完成；上述三个新实验尚未运行。
2026-09-21 沙箱外 nvidia-smi 确认两张 A100 80GB 均有其他计算任务，
利用率分别约 87% 和 85%。不得终止无关任务。
已有 run_batch.py、batch_worker.py、moe_gdn_ablation_worker.py 可复用
部分采集逻辑，但现有消融只测 B1、输入宽度 5，不能直接覆盖本合同。

新增 `calculate_break_even.py` 可对 baseline/candidate 批次周期 JSON
计算实际加速比、可用 Pre-Verify 预算及所需加速倍数。已用数值案例
检查持平、无法达到目标及不同周期输出数的聚合；没有使用模拟值冒充实测。

新增 `run_long_draft.py` 和 `long_draft_worker.py` 为实验 1/2 的待 GPU
验证入口。它们使用容量 40 的现有容器预留空间，安装 B1 专用 benchmark
controller：固定模式逐 token 调用私有 V2、完全跳过实际 MTP proposal；
balanced 模式保留原控制器及 margin 条件，最多执行 32 轮，在达到
32 个候选后停止，将末轮超出的候选截掉。原 Target verifier 不改。
其 config 初始化字段与运行时 probe 覆盖字段在 manifest 分别记录。
容器仍有 MTP 加载/预留开销，不能作为优化后的两级吞吐结果。

GPU 空闲后，先执行每种模式 1 请求、64 token 的 paired smoke，
再使用同一数据集运行正式接受率采集。命令示例：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH="$PWD/benchmarks/hierarchical:$PWD" \
.venv/bin/python benchmarks/hierarchical/run_long_draft.py \
  --dataset benchmarks/hierarchical/previous_config_20260909/samples_16.jsonl \
  --output benchmark_results/long_draft_smoke_fixed \
  --mode two_level_fixed --samples 1 --tokens 64
```

第二个 smoke 将 mode 改为 `three_level_balanced`，使用另一个 output。
正式运行默认 16 请求、每条 256 token、两遍无探针计时及独立 audit。
只有运行结束且重复/探针输出一致才生成 complete.json；这仍不等于
已通过 AR parity。512 格完整 forward harness 尚待实现及 GPU 验证。

## 8. GPU 空闲自动队列

`watch_long_draft.py --output <新目录>` 每 10 秒查询 GPU，连续三次
确认同卡无计算进程、显存占用低于 1024 MiB、利用率不超过 2%，
且总显存至少 80000 MiB 后自动运行。首次选卡后各阶段保持同一 GPU UUID。
执行顺序为 fixed smoke、balanced smoke、fixed 正式、balanced 正式。
smoke 为 1 请求、64 token、一遍测量；正式为 16 请求、256 token、两遍测量。

进度原子写入 output/status.json；命令、各任务日志、结果和成功回执
分别保存。只有子进程成功退出且 complete.json 有效时才继续。
源码指纹只覆盖 vLLM 运行时、所用采集脚本和数据集，不因无关文档或
benchmark 提交而变化。首个实验启动前允许刷新源码基线，记录
request.json/source_history，并重新累计空闲样本；一旦开始首个实验，
后续等待、运行及结果校验均禁止源码变化。每个任务记录实际 commit、
运行时 diff 和指纹。已成功任务可在同一源码版本恢复时跳过；失败或未完成任务保留产物并
要求检查，避免无人值守重复启动有问题的配置。GPU 查询失败会保持等待，
不会误判为空闲；运行中查询失败、源码变化或外部 GPU 进程进入则停止
自己的实验并保留失败信息。文件锁仅协调使用本脚本的进程，不是系统独占。

创建 output/STOP 文件可停止等待或结束本程序启动的实验进程组，
不终止其他人的任务。队列完成状态为 ACCEPTANCE_QUEUE_COMPLETE，
不代表第 3 项已完成。本队列没有自动修复失败实验的能力。
