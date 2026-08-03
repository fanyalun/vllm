# Agent Instructions for vLLM

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# If you are only making Python changes:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# If you are also making C/C++ changes:
uv pip install -e . --torch-backend=auto
```

### Running tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies.
# requirements/test/cuda.txt is pinned to x86_64; on other platforms, use the
# unpinned source file instead:
uv pip install -r requirements/test/cuda.in    # resolves for current platform
# Or on x86_64:
uv pip install -r requirements/test/cuda.txt

# Run a specific test file (use .venv/bin/python directly;
# `source activate` does not persist in non-interactive shells):
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

- If a command may spawn helper executables at runtime, make sure `.venv/bin`
  is on `PATH` first. Calling `.venv/bin/python` alone does **not** expose
  tools like `ninja` to subprocesses such as FlashInfer JIT.
- In one-shot shells, prefer `source .venv/bin/activate && ...` or prefix
  `PATH=.venv/bin:$PATH` before running GPU validation scripts.

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files

# Run mypy as it is in CI:
pre-commit run mypy-3.10 --all-files --hook-stage manual
```

The line length limit for Python code is 88 characters. If you are not sure, use pre-commit to check.

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

# 本地模型路径：
/home/fanya/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B

## Experiment Semantics: Qwen3.6 EP + Spec Decode

- Primary target scenario: large-batch expert parallel plus speculative decoding on a single node with 4 or 8 GPUs.
- When analyzing MoE decode or verification behavior, explicitly separate large-batch and small-batch conclusions; do not reuse a small-batch memory-bound conclusion as the default answer for the large-batch EP + spec-decode setting.
- The core validation questions for this workflow are:
  - whether speculative decoding increases expert-load imbalance,
  - how large that increase is,
  - whether rank-local FFN time is strongly correlated with routed assignments in the large-batch setting,
  - whether rank-local FFN time is more strongly correlated with activated expert count in the small-batch setting.
- For draft-drop analysis, the relevant question is how many draft tokens would be lost under the naive policy that drops draft tokens whose routed load exceeds the minimum-rank baseline.
- For Qwen3.6 linear attention, remember that speculative decoding parallelism is limited by verification-stage state-cache retention: every draft token may need its own linear state until acceptance is known.
- The current design direction under consideration is: predict acceptance length, keep only the last predicted accepted token's linear state on GPU, offload the other draft-token states to CPU, restore the correct state from CPU on misprediction, and clear the unused states after success or recovery.
