# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate untimed logprob audit using the matrix cell configuration."""

import json
import sys
from pathlib import Path

from benchmarks.replayssm import dspark_matrix_cell as cell
from vllm import SamplingParams

if __name__ == "__main__":
    directory = Path(sys.argv[sys.argv.index("--output") + 1])
    original_generate = cell.generate_cohort

    def generate_with_logprobs(llm, prompts, sampling_params):
        params = SamplingParams(
            temperature=0,
            max_tokens=sampling_params.max_tokens,
            ignore_eos=True,
            seed=20260915,
            logprobs=5,
        )
        outputs = original_generate(llm, prompts, params)
        with (directory / "logprobs.jsonl").open("a") as file:
            for output in outputs:
                completion = output.outputs[0]
                file.write(
                    json.dumps(
                        dict(
                            req_id=output.request_id,
                            token_ids=completion.token_ids,
                            logprobs=[
                                {token: value.logprob for token, value in row.items()}
                                for row in completion.logprobs
                            ],
                        )
                    )
                    + "\n"
                )
        return outputs

    cell.generate_cohort = generate_with_logprobs
    cell.main()
