# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check requested Qwen GDN window lengths against sequential recurrence."""

import argparse
import json
import runpy
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    namespace = runpy.run_path("tests/kernels/test_replayssm_spec_decode_gdn.py")
    results = []
    for draft in (4, 8, 16, 32):
        common = dict(
            state_dtype=torch.float32,
            act_dtype=torch.bfloat16,
            HQ=16,
            HV=32,
            K=128,
            V=128,
            buffer_len=64,
            max_spec_len=draft + 1,
        )
        try:
            namespace["_run_single_step"](**common, wp=0)
            namespace["_run_single_step"](**common, wp=64 - draft - 1)
            namespace["_run_rollback"](**common, num_steps=80)
            results.append(dict(draft=draft, passed=True))
        except Exception as error:
            results.append(dict(draft=draft, passed=False, error=str(error)))
        print(results[-1], flush=True)
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    if not all(row["passed"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
