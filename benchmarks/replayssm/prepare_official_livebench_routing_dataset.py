# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download a balanced official LiveBench prompt set for routing experiments."""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

CATEGORIES = (
    "coding",
    "data_analysis",
    "instruction_following",
    "language",
    "math",
    "reasoning",
)
DATASET_API = "https://huggingface.co/api/datasets/livebench/{category}"
ROWS_API = "https://datasets-server.huggingface.co/rows"
USER_AGENT = "vllm-replayssm-expert-routing-experiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B"
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=1700)
    parser.add_argument("--livebench-repo-commit", required=True)
    return parser.parse_args()


def get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def download_category(
    category: str, page_size: int
) -> tuple[list[dict[str, Any]], str]:
    metadata = get_json(DATASET_API.format(category=category))
    dataset_sha = metadata["sha"]
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": f"livebench/{category}",
                "config": "default",
                "split": "test",
                "offset": offset,
                "length": page_size,
            }
        )
        page = get_json(f"{ROWS_API}?{query}")
        rows.extend(item["row"] for item in page["rows"])
        total = int(page["num_rows_total"])
        offset = len(rows)
        if offset >= total:
            break
    return rows, dataset_sha


def allocate_rows(total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(CATEGORIES))
    return {
        category: base + int(index < remainder)
        for index, category in enumerate(CATEGORIES)
    }


def select_rows(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    prompt_token_count: Callable[[str], int],
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        turns = row.get("turns", [])
        if (
            len(turns) != 1
            or not isinstance(turns[0], str)
            or not turns[0].strip()
        ):
            continue
        prompt_tokens = prompt_token_count(turns[0])
        if prompt_tokens <= max_prompt_tokens:
            eligible.append({**row, "_prompt_tokens": prompt_tokens})
    if len(eligible) < count:
        raise ValueError(
            f"only {len(eligible)} eligible rows at <= {max_prompt_tokens} "
            f"prompt tokens, expected {count}"
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:count]


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.page_size <= 0 or args.max_prompt_tokens <= 0:
        raise ValueError("rows, page-size, and max-prompt-tokens must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    def prompt_token_count(question: str) -> int:
        messages = [{"role": "user", "content": question}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(token_ids["input_ids"])

    allocation = allocate_rows(args.rows)
    selected: dict[str, list[dict[str, Any]]] = {}
    dataset_shas = {}
    downloaded_counts = {}
    release_counts: dict[str, dict[str, int]] = {}
    task_counts: dict[str, dict[str, int]] = {}
    removed_counts: dict[str, int] = {}
    for index, category in enumerate(CATEGORIES):
        rows, dataset_sha = download_category(category, args.page_size)
        chosen = select_rows(
            rows,
            allocation[category],
            args.seed + index,
            prompt_token_count,
            args.max_prompt_tokens,
        )
        selected[category] = chosen
        dataset_shas[category] = dataset_sha
        downloaded_counts[category] = len(rows)
        release_counts[category] = {}
        task_counts[category] = {}
        removed_counts[category] = sum(
            bool(row.get("livebench_removal_date")) for row in chosen
        )
        for row in chosen:
            release = str(row["livebench_release_date"])[:10]
            release_counts[category][release] = (
                release_counts[category].get(release, 0) + 1
            )
            task = str(row["task"])
            task_counts[category][task] = task_counts[category].get(task, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cursors = {category: 0 for category in CATEGORIES}
    ordered_rows = []
    while len(ordered_rows) < args.rows:
        for category in CATEGORIES:
            cursor = cursors[category]
            if cursor >= len(selected[category]):
                continue
            source = selected[category][cursor]
            ordered_rows.append(
                {
                    "question": source["turns"][0],
                    "category": category,
                    "task": source["task"],
                    "question_id": source["question_id"],
                    "livebench_release_date": str(
                        source["livebench_release_date"]
                    )[:10],
                    "livebench_removal_date": (
                        str(source["livebench_removal_date"])[:10]
                        if source.get("livebench_removal_date")
                        else None
                    ),
                    "dataset_sha": dataset_shas[category],
                    "prompt_tokens": source["_prompt_tokens"],
                }
            )
            cursors[category] += 1
            if len(ordered_rows) == args.rows:
                break

    with args.output.open("x", encoding="utf-8") as output_file:
        for row in ordered_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    first_128_counts = {
        category: sum(row["category"] == category for row in ordered_rows[:128])
        for category in CATEGORIES
    }
    manifest = {
        "source": "official LiveBench Hugging Face datasets",
        "source_urls": {
            category: f"https://huggingface.co/datasets/livebench/{category}"
            for category in CATEGORIES
        },
        "livebench_repo": "https://github.com/LiveBench/LiveBench",
        "livebench_repo_commit": args.livebench_repo_commit,
        "dataset_shas": dataset_shas,
        "downloaded_counts": downloaded_counts,
        "selected_counts": allocation,
        "first_128_category_counts": first_128_counts,
        "release_counts": release_counts,
        "selected_removed_counts": removed_counts,
        "task_counts": task_counts,
        "rows": len(ordered_rows),
        "seed": args.seed,
        "tokenizer_model": args.model,
        "max_prompt_tokens": args.max_prompt_tokens,
        "selected_prompt_tokens": {
            "min": min(row["prompt_tokens"] for row in ordered_rows),
            "mean": sum(row["prompt_tokens"] for row in ordered_rows)
            / len(ordered_rows),
            "max": max(row["prompt_tokens"] for row in ordered_rows),
        },
        "selection": (
            "single-turn official public questions within the prompt-token cap; "
            "independently shuffled per category, then round-robin interleaved"
        ),
        "output": str(args.output.resolve()),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
