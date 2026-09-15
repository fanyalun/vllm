# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze one short test prompt from each of 16 randomly selected MMLU subjects."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    from datasets import Dataset
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rng = random.Random(20260915)
    base = Path("/home/fanya/.cache/huggingface/datasets/cais___mmlu")
    subjects = rng.sample(sorted(p.name for p in base.iterdir() if p.is_dir()), 16)
    tok = AutoTokenizer.from_pretrained(
        "/data1/fanya/Qwen/Qwen3.6-35B-A3B", local_files_only=True
    )
    rows = []
    for subject in subjects:
        path = next((base / subject).rglob("mmlu-test.arrow"))
        ds = Dataset.from_file(str(path))
        order = list(range(len(ds)))
        rng.shuffle(order)
        for idx in order:
            item = ds[idx]
            prompt = (
                "Explain your reasoning for the following "
                + subject.replace("_", " ")
                + " question, discuss the relevant concepts, "
                "and give your final answer.\n\n"
                + item["question"]
                + "\n\n"
                + "\n".join(
                    f"{chr(65 + i)}. {v}" for i, v in enumerate(item["choices"])
                )
            )
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if len(ids) <= 512:
                break
        else:
            raise RuntimeError(subject)
        rows.append(
            dict(
                id=len(rows),
                category=subject,
                source="cais/mmlu",
                split="test",
                row_index=idx,
                source_file=str(path),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                prompt=prompt,
                prompt_token_ids=ids,
                prompt_tokens=len(ids),
                gold_answer=item["answer"],
            )
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                seed=20260915,
                selection=(
                    "16 distinct subjects, shuffled test rows "
                    "with chat prompt <=512 tokens"
                ),
                samples=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    print([(r["category"], r["prompt_tokens"]) for r in rows])


if __name__ == "__main__":
    main()
