# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-B entry point for Qwen3.6/Gemma4 MTP and DSpark SSD matrices."""

try:
    from benchmarks.replayssm.async_ssd_eagle3_matrix import main
except ModuleNotFoundError:
    from async_ssd_eagle3_matrix import main


if __name__ == "__main__":
    raise SystemExit(main())
