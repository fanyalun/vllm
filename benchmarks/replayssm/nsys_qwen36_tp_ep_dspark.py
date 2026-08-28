# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture one steady Qwen3.6 ReplaySSM + DSpark speculative iteration.

The script is intended to run under ``nsys profile`` with
``--capture-range=cudaProfilerApi``. It warms up all model and ReplaySSM kernels,
then asks vLLM's CUDA profiler wrapper to capture exactly one steady decode
iteration: target verify and acceptance followed by the next DSpark proposal.
"""

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/data1/fanya/Qwen/Qwen3.6-35B-A3B",
    )
    parser.add_argument(
        "--draft-model",
        default=(
            "/home/fanya/data1/fanya/models/"
            "Qwen3.6-35B-A3B-speculator.dspark"
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--buffer-len", type=int, default=16)
    parser.add_argument("--num-speculative-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def _counter(metrics, name: str) -> float:
    return sum(
        metric.value
        for metric in metrics
        if getattr(metric, "name", None) == name and hasattr(metric, "value")
    )


def main() -> None:
    import torch

    from vllm import LLM, SamplingParams

    args = parse_args()
    spec_window = args.num_speculative_tokens + 1
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
        language_model_only=True,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=False,
        disable_log_stats=False,
        trust_remote_code=True,
        seed=0,
        speculative_config={
            "method": "dspark",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
        use_replayssm_spec=True,
        replayssm_buffer_len=args.buffer_len,
        additional_config={"gdn_prefill_backend": "triton"},
        compilation_config={
            "cudagraph_capture_sizes": [spec_window],
            "max_cudagraph_capture_size": spec_window,
        },
        profiler_config={
            "profiler": "cuda",
            # Step 1 is the new request's prefill. Step 2 is the first steady
            # target verify + accept + next-draft iteration and starts capture.
            "delay_iterations": 2,
            "max_iterations": 1,
        },
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        seed=0,
    )

    warmup_messages = [
        [{"role": "user", "content": "What is 19 multiplied by 23?"}]
    ]
    profile_messages = [
        [
            {
                "role": "user",
                "content": (
                    "A box has 48 red balls and 36 blue balls. If 21 balls "
                    "are removed, how many balls remain?"
                ),
            }
        ]
    ]

    llm.chat(
        warmup_messages,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    torch.cuda.synchronize()

    metrics_before = llm.get_metrics()
    drafts_before = _counter(
        metrics_before, "vllm:spec_decode_num_drafts"
    )
    accepted_before = _counter(
        metrics_before, "vllm:spec_decode_num_accepted_tokens"
    )
    llm.start_profile("qwen36_tp2_ep2_replayssm_dspark")
    outputs = llm.chat(
        profile_messages,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    llm.stop_profile()
    torch.cuda.synchronize()

    metrics_after = llm.get_metrics()
    drafts = _counter(
        metrics_after, "vllm:spec_decode_num_drafts"
    ) - drafts_before
    accepted = _counter(
        metrics_after, "vllm:spec_decode_num_accepted_tokens"
    ) - accepted_before
    result = {
        "model": args.model,
        "draft_model": args.draft_model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "expert_parallel_size": args.tensor_parallel_size,
        "all2all_backend": "allgather_reducescatter",
        "use_replayssm_spec": True,
        "replayssm_buffer_len": args.buffer_len,
        "num_speculative_tokens": args.num_speculative_tokens,
        "output_tokens": len(outputs[0].outputs[0].token_ids),
        "draft_rounds": drafts,
        "accepted_tokens": accepted,
        "mean_acceptance_length": 1.0 + accepted / drafts if drafts else None,
    }
    print("NSYS_PROFILE_RUN_JSON " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
