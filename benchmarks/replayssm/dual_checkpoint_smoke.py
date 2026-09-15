# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one reproducible Qwen GDN dual-checkpoint integration comparison cell."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument(
        "--draft-model",
        default=("/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark"),
    )
    parser.add_argument("--method", choices=["mtp", "dspark"], default="mtp")
    parser.add_argument("--draft", type=int, choices=[4, 8], default=4)
    parser.add_argument("--dual", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    spec = {"method": args.method, "num_speculative_tokens": args.draft}
    if args.method == "dspark":
        spec["model"] = args.draft_model
    config = dict(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        language_model_only=True,
        mamba_ssm_cache_dtype="float32",
        max_model_len=512,
        max_num_seqs=2,
        max_num_batched_tokens=512,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        enforce_eager=args.eager,
        gpu_memory_utilization=0.95,
        kv_cache_memory_bytes=2 * 1024**3,
        seed=0,
        use_replayssm_spec=True,
        replayssm_buffer_len=16,
        replayssm_spec_dual_checkpoint=args.dual,
        speculative_config=spec,
        additional_config={"gdn_prefill_backend": "triton"},
        compilation_config={
            "cudagraph_capture_sizes": [args.draft + 1, 2 * (args.draft + 1)],
            "max_cudagraph_capture_size": 2 * (args.draft + 1),
        },
        kernel_config={"enable_flashinfer_autotune": False},
        disable_log_stats=False,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        config=json.loads(json.dumps(config)),
        runner=os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
    )
    repo = Path(__file__).resolve().parents[2]
    sources = [
        "vllm/model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py",
        "vllm/model_executor/layers/fla/ops/gdn_replayssm_dual_checkpoint.py",
        "vllm/v1/attention/backends/gdn_attn.py",
        "vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
    ]
    result["source_sha256"] = {
        path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in sources
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    llm = LLM(**config)
    prompts = [
        "Explain why the sky looks blue in three short sentences.",
        "A shop has 12 boxes with 8 pencils each and sells 19 pencils. "
        "How many pencils remain? Explain the calculation.",
    ]
    samples = SamplingParams(
        temperature=0, max_tokens=64, ignore_eos=True, seed=0, logprobs=5
    )

    def generate():
        return llm.chat(
            [[{"role": "user", "content": p}] for p in prompts],
            samples,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )

    warmup = generate()
    started = time.perf_counter()
    outputs = generate()
    elapsed = time.perf_counter() - started
    result.update(
        elapsed_s=elapsed,
        prompt_token_ids=[list(o.prompt_token_ids) for o in outputs],
        token_ids=[list(o.outputs[0].token_ids) for o in outputs],
        warmup_token_ids=[list(o.outputs[0].token_ids) for o in warmup],
        logprobs=[
            [
                {str(token): value.logprob for token, value in position.items()}
                for position in o.outputs[0].logprobs
            ]
            for o in outputs
        ],
        text=[o.outputs[0].text for o in outputs],
        metrics={
            m.name: m.value
            for m in llm.get_metrics()
            if hasattr(m, "value") and "spec_decode" in m.name
        },
        complete=True,
    )
    assert all(len(ids) == 64 for ids in result["token_ids"])
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": True, "elapsed_s": elapsed}), flush=True)


if __name__ == "__main__":
    main()
