# AR and full-budget MoE-Skip performance diagnosis

This diagnostic investigates why h=8 drafting plus verification appeared faster
than synchronous AR in the static budget experiment. It does not assume the
speedup comes from expert skipping: h=8 retains the native expert budget.

Run `diagnose_performance_paths.py --output <fresh-directory> --gpu 1` with the
repository's `.venv/bin/python`. `--wait-pids <pid>` waits for an existing job to
exit; the driver also waits for GPU memory use below 1 GiB before starting.
The six cells cover Qwen3.6 and Gemma4 with synchronous AR, h=8/D=8, and
asynchronous AR. Each cell warms up the first two archived prompts, then measures
three repetitions of two requests with 512 output tokens. Initialization and
warmup are excluded. Prompt identities match the archived static experiment;
generated continuations are not forced to be identical across methods.

Uninstrumented measurements precede installation of worker timing wrappers.
Separate requests capture nested CUDA-event intervals for 512 output tokens and
a CPU/CUDA profiler trace for 64 output tokens. Target graph replay covers the
backbone, while the draft graph also includes logits, argmax and input updates.
These graph timings therefore have different boundaries. Sampling, complete
proposal and Target execution spans are retained to account for the boundaries.
Do not sum nested spans or add CPU durations to overlapping CUDA intervals.

Run `analyze_performance_paths.py --output <directory>` after completion. The
analysis retains event instrumentation overhead and output-repeat comparisons.
CUDA busy time uses the union of kernel intervals across streams. Profiler
measurements include prefill and terminal proposals and are diagnostic only.
The six-cell run is a small reproduction, not a replacement for the 16x512
static matrix or proof of output equivalence. Fixed-prefix replay and further
ablations may be needed depending on the measured differences.
