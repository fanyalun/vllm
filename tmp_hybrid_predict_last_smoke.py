import gc
import json
import os
import sys

import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "/home/fanya/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B"

DEFAULT_PROMPTS = [
    "Return exactly one word: alpha",
    (
        "Important: The secret number is 42. "
        "The sky is green in this hypothetical world. "
        "Apples grow on trees in the forest. "
        "Rivers flow through the valleys and mountains. "
        "Birds sing songs in the early morning light. "
        "The weather today is sunny with clear skies ahead. "
        "Flowers bloom in the garden during spring season. "
        "Now answer with ONLY the number and nothing else: "
        "What is the secret number plus one?"
    ),
]


def get_prompts() -> list[str]:
    prompt_case = os.getenv("SMOKE_PROMPT_CASE", "default")
    if prompt_case == "swap":
        return [DEFAULT_PROMPTS[1], DEFAULT_PROMPTS[0]]
    if prompt_case != "default":
        raise ValueError(f"Unsupported SMOKE_PROMPT_CASE={prompt_case!r}")
    return list(DEFAULT_PROMPTS)


COMMON = dict(
    model=MODEL,
    tokenizer=MODEL,
    tensor_parallel_size=2,
    trust_remote_code=True,
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=2,
    gpu_memory_utilization=0.9,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    mamba_cache_mode="align",
    enforce_eager=True,
    seed=123,
)


def cleanup() -> None:
    cleanup_dist_env_and_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_once(mode: str) -> dict[str, list]:
    prompts = get_prompts()
    print(
        json.dumps(
            {
                "mode": mode,
                "prompt_case": os.getenv("SMOKE_PROMPT_CASE", "default"),
                "num_prompts": len(prompts),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    llm = LLM(
        **COMMON,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 2,
            "hybrid_spec_state_offload_mode": mode,
            "hybrid_spec_state_ewma_alpha": 0.5,
        },
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=24)
    outputs = llm.generate(prompts, sampling_params)
    result = {
        "texts": [output.outputs[0].text for output in outputs],
        "token_ids": [output.outputs[0].token_ids for output in outputs],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    del llm
    cleanup()
    return result


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit("Usage: tmp_hybrid_predict_last_smoke.py [mode]")

    if len(sys.argv) == 2:
        mode = sys.argv[1]
        if mode not in ("disabled", "predict_last"):
            raise SystemExit(f"Unsupported mode={mode!r}")
        run_once(mode)
        return

    disabled = run_once("disabled")
    predict_last = run_once("predict_last")
    same = disabled["token_ids"] == predict_last["token_ids"]
    print(f"MATCH={same}", flush=True)
    if not same:
        print(
            json.dumps(
                {
                    "disabled": disabled,
                    "predict_last": predict_last,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
