# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument(
        "--method",
        choices=("ar", "moe_skip", "mtp", "eagle3", "dspark"),
        required=True,
    )
    parser.add_argument("--spec-model")
    parser.add_argument("--draft-length", type=int)
    parser.add_argument("--top-h", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--logprobs", type=int)
    parser.add_argument("--trace-dir")
    return parser.parse_args()


def load_prompts(path: Path, num_samples: int) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as dataset_file:
        for index, line in enumerate(dataset_file):
            if index >= num_samples:
                break
            row = json.loads(line)
            prompt = row.get("prompt", row.get("question", row.get("text")))
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"Dataset row {index} has no non-empty prompt")
            prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
            if row.get("prompt_sha256", prompt_sha256) != prompt_sha256:
                raise ValueError(f"Dataset row {index} has an invalid prompt hash")
            samples.append(
                {
                    "sample_index": row.get("sample_index", index),
                    **({"category": row["category"]} if "category" in row else {}),
                    **(
                        {"source_index": row["source_index"]}
                        if "source_index" in row
                        else {}
                    ),
                    **(
                        {"prompt_token_count_manifest": row["prompt_token_count"]}
                        if "prompt_token_count" in row
                        else {}
                    ),
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256,
                }
            )
    if len(samples) != num_samples:
        raise ValueError(
            f"Expected {num_samples} dataset rows, found {len(samples)} in {path}"
        )
    return samples


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.parent / "progress.json"
    if args.method != "ar" and args.draft_length is None:
        raise ValueError("--draft-length is required for speculative methods")
    if args.method == "ar" and args.draft_length is not None:
        raise ValueError("--draft-length is not valid for ar")
    if args.method in ("eagle3", "dspark") and args.spec_model is None:
        raise ValueError(f"--spec-model is required for {args.method}")
    if args.method not in ("mtp", "eagle3", "dspark") and args.spec_model is not None:
        raise ValueError(f"--spec-model is not valid for {args.method}")
    if args.sample_index is not None and not (
        0 <= args.sample_index < args.num_samples
    ):
        raise ValueError("--sample-index must select one of --num-samples rows")
    if args.logprobs is not None and args.logprobs < 1:
        raise ValueError("--logprobs must be positive")
    if args.max_model_len <= args.max_tokens:
        raise ValueError("--max-model-len must exceed --max-tokens")
    if args.trace_dir:
        if args.method == "ar":
            raise ValueError("--trace-dir is not valid for ar")
        trace_env = (
            "VLLM_MOE_SKIP_TRACE_DIR"
            if args.method == "moe_skip"
            else "VLLM_DRAFT_TOPK_TRACE_DIR"
        )
        os.environ[trace_env] = args.trace_dir

    from vllm import LLM, SamplingParams

    samples = load_prompts(Path(args.dataset), args.num_samples)
    if args.sample_index is not None:
        samples = [samples[args.sample_index]]
    speculative_config = None
    if args.method == "moe_skip":
        speculative_config = {
            "method": "moe_skip",
            "num_speculative_tokens": args.draft_length,
            "moe_skip_top_h": args.top_h,
        }
    elif args.method == "mtp":
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": args.draft_length,
        }
        if args.spec_model is not None:
            speculative_config["model"] = args.spec_model
    elif args.method in ("eagle3", "dspark"):
        speculative_config = {
            "method": args.method,
            "model": args.spec_model,
            "num_speculative_tokens": args.draft_length,
        }

    start = time.perf_counter()
    engine_kwargs = {}
    if args.method != "ar":
        engine_kwargs["per_request_spec_decode_metrics"] = "detailed"
    max_num_batched_tokens = args.max_num_batched_tokens or args.max_model_len
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        enforce_eager=args.mode == "eager",
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=0.95,
        speculative_config=speculative_config,
        disable_log_stats=True,
        seed=0,
        **engine_kwargs,
    )
    init_seconds = time.perf_counter() - start
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        logprobs=args.logprobs,
    )

    outputs = []
    generation_start = time.perf_counter()
    for sample in samples:
        request_output = llm.generate([sample["prompt"]], sampling_params)[0]
        prompt_token_count = len(request_output.prompt_token_ids)
        manifest_count = sample.get("prompt_token_count_manifest")
        if manifest_count is not None and prompt_token_count != manifest_count:
            raise RuntimeError(
                f"Sample {sample['sample_index']} prompt-token count changed: "
                f"manifest={manifest_count}, runtime={prompt_token_count}"
            )
        if prompt_token_count + args.max_tokens > args.max_model_len:
            raise RuntimeError(
                f"Sample {sample['sample_index']} requires "
                f"{prompt_token_count + args.max_tokens} tokens, exceeding "
                f"max_model_len={args.max_model_len}"
            )
        completion = request_output.outputs[0]
        token_ids = list(completion.token_ids)
        if len(token_ids) != args.max_tokens:
            raise RuntimeError(
                f"Sample {sample['sample_index']} produced {len(token_ids)} tokens; "
                f"expected exactly {args.max_tokens}"
            )
        metrics = (
            completion.spec_decode_metrics.to_dict()
            if completion.spec_decode_metrics is not None
            else None
        )
        output_logprobs = None
        if completion.logprobs is not None:
            output_logprobs = []
            for position_logprobs in completion.logprobs:
                if position_logprobs is None:
                    output_logprobs.append(None)
                    continue
                candidates = [
                    {
                        "token_id": token_id,
                        "logprob": logprob.logprob,
                        "rank": logprob.rank,
                        "decoded_token": logprob.decoded_token,
                    }
                    for token_id, logprob in position_logprobs.items()
                ]
                candidates.sort(
                    key=lambda candidate: (
                        candidate["rank"] is None,
                        candidate["rank"] or 0,
                        candidate["token_id"],
                    )
                )
                output_logprobs.append(candidates)
        outputs.append(
            {
                **sample,
                "prompt_token_count": prompt_token_count,
                "request_id": request_output.request_id,
                "token_ids": token_ids,
                "text": completion.text,
                "finish_reason": completion.finish_reason,
                "logprobs": output_logprobs,
                "spec_decode_metrics": metrics,
            }
        )
        progress_path.write_text(
            json.dumps(
                {
                    "status": "running",
                    "completed_samples": len(outputs),
                    "expected_samples": len(samples),
                    "last_sample_index": sample["sample_index"],
                    "category_counts": dict(
                        sorted(
                            Counter(
                                output.get("category", "uncategorized")
                                for output in outputs
                            ).items()
                        )
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    result = {
        "model": args.model,
        "dataset": str(Path(args.dataset).resolve()),
        "method": args.method,
        "spec_model": args.spec_model,
        "mode": args.mode,
        "draft_length": args.draft_length,
        "top_h": args.top_h if args.method == "moe_skip" else None,
        "num_samples": len(samples),
        "sample_index": args.sample_index,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "num_logprobs": args.logprobs,
        "temperature": 0,
        "ignore_eos": True,
        "batch_size": 1,
        "category_counts": dict(
            sorted(
                Counter(
                    sample.get("category", "uncategorized") for sample in samples
                ).items()
            )
        ),
        "prompt_format": "raw_text_no_chat_template",
        "seed": 0,
        "init_seconds": init_seconds,
        "generation_seconds": time.perf_counter() - generation_start,
        "outputs": outputs,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    progress_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_samples": len(outputs),
                "expected_samples": len(samples),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"CELL_COMPLETE {output_path}", flush=True)


if __name__ == "__main__":
    main()
