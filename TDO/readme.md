## 1. 动机

现有树形 speculative decoding 通过给 target model 一次验证更多候选分支来提高单轮 accepted length，但这种策略对 **MoE + Expert Parallelism (EP) + 大 batch** 并不一定划算。随着 draft tree 变宽，target 需要验证更多 speculative tokens；不同分支还可能路由到不同 experts，从而扩大 expert activation union、增加 Expert 计算和 EP 通信。EVICT 已经直接观察到这一问题，并指出 MoE 上 tree 扩张会显著增加 target-side verification cost。([arXiv][1])

另一方面，DSpark/PCTree 表明，**分支探索不一定需要昂贵的 autoregressive drafter forward**。DSpark 用一次并行 backbone 得到整个 block 的表示，再通过轻量 Markov head 恢复 token 间依赖；PCTree进一步利用这个 Markov head，在**不增加额外 backbone pass**的情况下，为不同 parent 扩展多个 continuation，将 chain 转成 tree。([arXiv][2])

与此同时，vLLM 的 DBO 已经证明，在 DP+EP 部署下，MoE sparse All-to-All 并不是一个不可利用的纯等待阶段：它可以与独立计算 overlap。DBO 当前采用两个 microbatch，让一个 microbatch 等待 Dispatch/Combine 通信时执行另一个 microbatch 的 Attention、MLP 等计算。([vLLM][3])

最后，SSD/Saguaro 又解决了一个关键依赖：正常 speculative decoding 必须等 target verification 完成后才能知道下一轮 draft 的 prefix；SSD 则在当前 verification 尚未完成时，提前预测可能的 verification outcomes，并为这些未来状态预计算下一轮 draft。如果真实 outcome 命中预测集合，就可以直接复用预计算结果。([arXiv][4])

因此本文的核心问题是：

[
\boxed{
\text{MoE 是否改变了 speculative redundancy 的最佳放置位置？}
}
]

即，与其把更多冗余计算放到昂贵的 target tree verification 上，是否可以**将其迁移到更便宜的 drafter，并进一步隐藏在 EP All-to-All 通信窗口中**？

---

# 2. 核心思想

我们提出 **Target-Draft Overlap（TDO）**。

其核心思想可以概括为：

[
\boxed{
\text{Verifier-side redundancy}
\rightarrow
\text{Drafter-side redundancy}
\rightarrow
\text{hide under EP communication}
}
]

传统 Tree SD 的做法是：

```text
更多 speculative branches
          ↓
      更大的 Draft Tree
          ↓
Target 一次验证更多 nodes
          ↓
更多 MoE routing / Expert / A2A
```

TDO 则反过来：

```text
Target：
只验证 narrow chain / small tree
             ↓
降低 MoE verification cost


Draft：
探索更多可能的 future outcomes / branches
             ↓
增加的是便宜的 draft computation
             ↓
放到 Target EP All2All 期间执行
```

因此 TDO 和 DBO 有本质区别：

[
\text{DBO}:\quad
Target(B_0)\parallel Target(B_1)
]

而 TDO 是：

[
\boxed{
\text{TDO}:\quad
Target_t(B)\parallel Draft_{t+1}(B)
}
]

TDO **不拆 batch、不使用两个 target microbatch**，而是让当前 round 的 target verification 与下一轮 speculative drafting 并行。

其核心假设是：

> **对于 MoE target，把一个 speculative branch 放到 verifier 侧的边际成本，高于把这个 branch 放到 drafter 侧的边际成本；如果后者还能隐藏在 EP communication 中，那么 speculative decoding 的最优策略可能从 verifier-heavy 转向 drafter-heavy。**

---

# 3. 设计

TDO 初步可以由四个模块组成。

### 3.1 Narrow Target Verification

Target 不再验证很宽的 tree，而主要验证一条长度为 (K) 的 chain，或者一个严格受限的 small tree：

```text
D → E → F → G → H
```

目标不是最大化：

[
\text{accepted tokens / round}
]

而是最大化：

[
\boxed{
\frac{\text{committed tokens}}
{\text{target verification cost}}
}
]

这与 EVICT 所揭示的现象一致：对于 MoE，并非所有高概率 tree nodes 都值得付出 target verification cost。([arXiv][1])

### 3.2 Future Outcome Prediction

当前 target 正在验证：

```text
E F G H
```

可能出现：

```text
O1 = accept E F G H + X
O2 = accept E F G   + Y
O3 = accept E F     + Z
O4 = accept E       + W
...
```

利用 DSpark 已有的：

* confidence head；
* Markov distribution；
* draft logits；

对这些 future outcomes 排序，只选择概率最高的 Top-(M) 个状态。

SSD/Saguaro 已经证明了这种“verification 尚未结束时预测未来 outcome”的执行方式可以打破 `verification → next drafting` 的串行依赖。([arXiv][4])

### 3.3 Cheap Draft-Side Branch Exploration

对于一个已经计算出 DSpark backbone 表示的状态，可以使用 PCTree 的 parent-conditioned Markov head：

[
L_k(\cdot|x_{k-1})
==================

U_k+B(x_{k-1},\cdot)
]

廉价扩展多个 branch，而不需要为同一状态重复执行 backbone。PCTree 已经证明这种 inference-only tree expansion 可以显著提高 accepted length，并且无需额外 backbone pass。([arXiv][5])

需要注意的是：

> **不同 SSD future outcomes 不一定能共享同一个 DSpark backbone。**

因此最终实现更可能是：

[
\boxed{
\text{少量 future-state backbone computation}
+
\text{大量 cheap Markov branching}
}
]

而不是“一个 backbone 免费生成所有未来状态”。

### 3.4 Target-Draft Overlap

Target 执行 MoE layer：

```text
Attention
   ↓
Router
   ↓
Dispatch A2A ======================
                 ↑
                 │
             Draft compute
   ↓
Expert
   ↓
Combine A2A  ======================
                 ↑
                 │
             Draft compute
```

vLLM DBO 已经证明这些 sparse A2A communication region 可以与独立 computation overlap。([vLLM][3])

TDO 利用同样的底层机会，但插入的不是另一个 target microbatch，而是：

```text
future draft O1
future draft O2
future draft O3
...
```

verification 完成以后：

```text
Actual outcome = O2
        ↓
cache hit
        ↓
直接取出 O2 对应的 next-round draft
        ↓
立即进入下一轮 Target verification
```

理想 steady-state：

```text
Target:  Verify₀ ───── Verify₁ ───── Verify₂ ─────
Draft:         Draft₁        Draft₂        Draft₃
                 ↑             ↑
              hidden inside EP communication
```

---

# 4. 初步实验验证方案

当前阶段**不实现 TDO**，而是先验证 TDO 所依赖的四个必要假设。

### Experiment 1：验证“大 batch + MoE 下 wide tree 是否真的不划算”

这是最优先的实验。

选择 MoE target + EP，sweep：

```text
Batch size:
1 / 8 / 16 / 32 / 64 / 128

Verification width:
Chain
Tree-4
Tree-8
Tree-16
Tree-32
Tree-64
```

记录：

* target verification latency；
* mean committed / accepted length；
* Dispatch A2A；
* Expert compute；
* Combine A2A；
* unique activated experts；
* 每个 EP rank 的 token load。

最重要的指标不是单纯 `T_verify`，而是：

[
\boxed{
C_{\rm commit}
==============

\frac{T_{\rm verify}}
{\text{mean committed tokens}}
}
]

目标验证：

[
\boxed{
\text{tree width增加后，
verification cost 的增速是否超过 acceptance gain}
}
]

如果在 large-batch regime 下出现明显的最优 tree width，超过该宽度后 `verification latency / committed token` 持续上升，则论文最核心的 motivation 成立。EVICT 的结果已经为这一现象提供了外部证据。([arXiv][1])

---

### Experiment 2：验证“branch redundancy 放在 Draft 侧是否真的更便宜”

直接比较：

```text
DSpark chain
vs
PCTree
```

因为二者共享 backbone，只改变 Markov decoding policy。([arXiv][5])

测：

[
\Delta T_D(N)
=============

## T_{\rm draft-tree}(N)

T_{\rm draft-chain}
]

然后用 Experiment 1 得到：

[
\Delta T_V(N)
=============

## T_{\rm verify-tree}(N)

T_{\rm verify-chain}.
]

定义：

[
\boxed{
R_{\rm migration}(N)
====================

\frac{\Delta T_V(N)}
{\Delta T_D(N)}
}
]

如果：

[
R_{\rm migration}\gg1
]

则说明：

> **同样用于增加 speculative branch coverage，把计算放在 drafter 侧比 verifier 侧便宜得多。**

这将直接支撑“Verifier → Drafter redundancy migration”的核心思想。

---

### Experiment 3：验证 EP communication 是否真的提供足够 overlap capacity

这里**不需要实现 TDO**，直接把现成 DBO 当作硬件/运行时 probe。

vLLM 官方 DBO 本来就是通过：

[
\text{A2A}*{B_0}
\parallel
\text{Compute}*{B_1}
]

隐藏 sparse A2A。([vLLM][3])

使用 Nsight Systems/profile 数据测：

```text
DBO OFF
vs
DBO ON
```

重点不是 DBO 最终提高多少 throughput，而是估计：

[
\boxed{
W_{\rm overlap}
===============

\text{真实 A2A 中能够容纳的 independent compute 时间}
}
]

如果 DBO 能在你的目标 `batch × EP × hardware` regime 下隐藏大量计算，就说明 TDO 的底层 execution opportunity 真实存在。

如果几乎没有 overlap capacity，则 TDO 可以直接停止。

---

### Experiment 4：验证 future outcome 是否足够可预测

完全离线即可。

正常运行 chain speculative decoding，每轮保存：

```text
draft logits
DSpark confidence
Markov logits

真实：
accept length
correction / bonus token
```

定义真实 outcome：

[
O=(k,x)
]

其中 (k) 为 accepted prefix length，(x) 为 target correction/bonus token。

预测 Top-(M) outcomes：

[
M=1,2,4,8,16,32
]

计算：

[
\boxed{
Hit@M
=====

P(O_{\rm real}\in\hat{\mathcal O}_M)
}
]

SSD/Saguaro 已经表明提前预测 verification outcome 并缓存下一轮 speculation 是可行的；这里需要验证的是，在**你的 MoE workload 和 drafter 上**，用多大的 (M) 才能达到足够高的 coverage。([arXiv][4])

---

最后把四组实验输入一个 **Oracle TDO Simulator**：

[
T_{\rm TDO}
\approx
T_{\rm verify-chain}
+
T_{\rm exposed\ draft}
+
P_{\rm miss}T_{\rm fallback}.
]

其中：

[
T_{\rm exposed\ draft}
======================

\max(0,T_D(M)-W_{\rm overlap}).
]

首先计算一个最乐观上界：

[
\boxed{
R_{\rm TDO}^{upper}
===================

\frac{\tau_{\rm chain}}
{T_{\rm verify-chain}}
}
]

如果即使假设：

* 100% draft compute 被隐藏；
* 100% future-outcome cache hit；

这个理论上界仍然打不过最好的 tree baseline，那么可以直接停止 TDO。

反之，如果在 **large batch + MoE + EP** 下仍存在明显的理论收益空间，再进入真正的 TDO runtime 实现。

---

# 5. 参考文献

**[1] Li et al. From Chains to Trees: Parent-Conditioned Drafting for Semi-Autoregressive Speculative Decoding. 2026.**
提出 PCTree，利用 DSpark 已训练好的 Markov head 对不同 parent 分别扩展 child，在不增加额外 backbone pass 的情况下将 chain 转化为 tree。([arXiv][5])
地址：`https://arxiv.org/abs/2608.02123`

**[2] Cheng et al. DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation. 2026.**
提出 parallel backbone + lightweight sequential module 的 semi-autoregressive drafter，以及基于 prefix survival probability 的 confidence-scheduled verification。([arXiv][2])
地址：`https://arxiv.org/abs/2607.05147`

**[3] Kumar, Dao, May. Speculative Speculative Decoding. 2026.**
提出 SSD 以及 Saguaro：当前 verification 尚未完成时预测可能的 verification outcomes，并提前为这些未来状态生成下一轮 speculation，从而打破 drafting 与 verification 的串行依赖。([arXiv][4])
地址：`https://arxiv.org/abs/2603.03251`

**[4] Pan et al. Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding. 2026.**
提出 EVICT，并直接研究 MoE tree verification 的成本问题：tree 分支增加会扩大 activated-expert union，从而提高 target verification cost。([arXiv][1])
地址：`https://arxiv.org/abs/2605.00342`

**[5] vLLM. Dual Batch Overlap Design Document. 2026.**
vLLM 官方 DBO 设计文档。DBO 将 batch 拆为两个 microbatch，通过 yield points 使一个 microbatch 的 sparse All-to-All communication 与另一个 microbatch 的 computation overlap。([vLLM][3])
地址：`https://docs.vllm.ai/en/latest/design/dbo/`

**[6] vLLM. DBO source/design document on GitHub.**
DBO 的源码设计入口，可进一步追踪 `GPUModelRunner`、`UBatchWrapper`、`UBatchContext` 以及 `FusedMoEModularKernel` 中的 yield/recv-hook 实现。([GitHub][6])
地址：`https://github.com/vllm-project/vllm/blob/main/docs/design/dbo.md`

这套结构里，真正最关键的第一张实验图应该是 **`Target verification time / committed token` 随 `batch size × tree width` 的变化**。它决定了你的论文有没有一个真实存在、值得解决的问题；TDO 本身应该建立在这条 observation 被数据确认之后。

[1]: https://arxiv.org/abs/2605.00342?utm_source=chatgpt.com "Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding"
[2]: https://arxiv.org/abs/2607.05147?utm_source=chatgpt.com "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"
[3]: https://docs.vllm.ai/en/latest/design/dbo/?utm_source=chatgpt.com "Dual Batch Overlap - vLLM"
[4]: https://arxiv.org/abs/2603.03251?utm_source=chatgpt.com "Speculative Speculative Decoding"
[5]: https://arxiv.org/abs/2608.02123?utm_source=chatgpt.com "From Chains to Trees: Parent-Conditioned Drafting for Semi-Autoregressive Speculative Decoding"
[6]: https://github.com/vllm-project/vllm/blob/main/docs/design/dbo.md?utm_source=chatgpt.com "vllm/docs/design/dbo.md at main · vllm-project/vllm · GitHub"
