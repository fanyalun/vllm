# 分支适配与本次运行问题

实验使用 `exp/replayssm-official-qwen36-tp2-ep2`，源码提交为
`501b9513ed8fd663d90ea3a2326ac9fa0a633b0f`。本次通过 `git ls-remote`
核对，GitHub 同名分支指向相同提交。分支名称不强制使用 TP2/EP2；
本轮所有模型实验显式 TP1、EP 关闭，每个进程只可见一张 A100。

ReplaySSM 已有适配：`qwen_gdn_linear_attn.py` 根据
`use_replayssm_spec` 调用 `gdn_replayssm_spec_decode`，正式配置的独立
CUDA profile 中可看到对应 kernel。四种草稿长度的单步和随机接受/回滚
kernel 检查均通过，见 `kernel_correctness.json`。这些检查不等同于
AR、SD 和 ReplaySSM 整模型逐 token 完全一致，后者另见输出审计。

首次自动显存探测在大验证宽度下发生 OOM。该路径在
`GPUModelRunner._init_minimal_kv_cache_for_profiling` 中使用最大 CUDA
Graph token capture size 作为临时 cache block 数；草稿宽度也计入
capture size，因此临时分配可能很大。这发生在正式 cache 分配之前。
原始 OOM 日志、参数及命令的副本见 `diagnostics/initial_profile_oom_*`；
它们是已替代的诊断记录，不计入正式性能矩阵。
最终实验使用公开的 `kv_cache_memory_bytes=7*1024**3` 配置跳过该自动
探测，所有方法、所有配置使用同一预算，仍保留 CUDA Graph。
B16/D32 ReplaySSM 在此预算下报告 40 条最大上下文请求的 cache 容量，
并完成实际 16 条请求并发执行及三轮正式计时。

初版 benchmark 自身也有两次启动失败：worker RPC 不能直接序列化所用
函数、worker extension 模块路径不能被子进程导入。最终脚本改为扩展
类的命名 RPC，并显式加入脚本所在目录。这些是测量脚本的问题，不是
ReplaySSM 模型适配缺失；失败日志保留在本地 `startup_failures/`。

大草稿宽度会触发耗时的 Triton/PTX 编译。正式轮次使用 worker JIT
事件计数，任何含 JIT 的轮次归档为额外预热，重新补足三轮无 JIT 测量。
没有按吞吐高低挑选轮次。模型加载、编译、预热、独立 profile 均不计入
端到端计时。

当前 `replayssm_config.py` 的 GDN 专用参数表面向 Blackwell；A100
走通用默认参数。实验保留这一源码行为，不在不同配置间调 kernel 参数。
因此结果评价的是该提交及本轮配置在 A100 上的表现，不是对 ReplaySSM
算法所有实现或硬件的性能上限判断。

本文及实验脚本使用 AI 辅助生成。
