# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a four-prompt, 256-output-token MoE-Skip branch sensitivity probe."""

import argparse
import json
from pathlib import Path

from run_performance import MODELS, ROOT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft-length", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    from vllm import LLM, SamplingParams

    source = ROOT / "benchmark_results" / MODELS[args.model]["source"]
    contract = json.loads((source / "EXPERIMENT_CONTRACT.json").read_text())
    samples = [
        json.loads(line) for line in Path(contract["dataset"]).read_text().splitlines()
    ][:4]
    write_json(directory / "samples.json", samples)
    config = {
        "model": args.model,
        "model_path": MODELS[args.model]["model"],
        "num_samples": 4,
        "max_tokens": 256,
        "draft_length": args.draft_length,
        "threshold": args.threshold,
        "top_h": 4,
        "batch_size": 1,
        "temperature": 0,
        "enforce_eager": True,
        "control": args.control,
        "source_dataset": contract["dataset"],
        "scope": "All low-margin draft positions with nonempty in-budget suffix",
    }
    write_json(directory / "contract.json", config)
    llm = LLM(
        model=config["model_path"],
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        speculative_config={
            "method": "moe_skip",
            "num_speculative_tokens": args.draft_length,
            "moe_skip_top_h": 4,
        },
        per_request_spec_decode_metrics="none",
        disable_log_stats=True,
        seed=0,
        worker_extension_cls="branch_probe_worker.BranchProbeWorker",
    )
    sampling = SamplingParams(temperature=0, max_tokens=256, ignore_eos=True)
    tokenizer = llm.get_tokenizer()
    outputs, events = [], []
    for sample in samples:
        if not args.control:
            llm.collective_rpc(
                "begin_branch_probe",
                args=(sample["prompt_token_count"], 256, args.threshold),
            )
        output = llm.generate([sample["prompt"]], sampling, use_tqdm=False)[0]
        assert len(output.prompt_token_ids) == sample["prompt_token_count"]
        assert len(output.outputs[0].token_ids) == 256
        collected = (
            {"events": [], "proposals": None}
            if args.control
            else llm.collective_rpc("collect_branch_probe")[0]
        )
        for event in collected["events"]:
            event.update(
                sample_index=sample["sample_index"], category=sample["category"]
            )
            j = event["draft_position"] - 1
            event["top1_branch_text"] = tokenizer.decode(event["baseline"][j:])
            event["top2_branch_text"] = tokenizer.decode(event["branch"][j:])
            events.append(event)
        outputs.append(
            {
                "sample_index": sample["sample_index"],
                "category": sample["category"],
                "token_ids": list(output.outputs[0].token_ids),
                "proposals": collected["proposals"],
                "branch_events": len(collected["events"]),
            }
        )
        write_json(directory / "result.json", {"outputs": outputs, "events": events})
        print(f"SAMPLE_COMPLETE {len(outputs)}/4 events={len(events)}", flush=True)
    write_json(
        directory / "audit.json",
        {
            "complete": len(outputs) == 4,
            "output_tokens": sum(len(o["token_ids"]) for o in outputs),
            "events": len(events),
            "shared_prefix_and_forced_top2_checks": (
                "not applicable (control)" if args.control else "passed for every event"
            ),
            "baseline_replay_check": (
                "not applicable (control)"
                if args.control
                else "passed for every proposal with branches"
            ),
        },
    )
    (directory / "RUN_COMPLETE").write_text(
        "4 x 256; control complete\n"
        if args.control
        else "4 x 256; all branch assertions passed\n"
    )


if __name__ == "__main__":
    main()
