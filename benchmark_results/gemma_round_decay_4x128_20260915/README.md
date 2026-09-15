# Gemma four-round decay diagnostic

B1, greedy, 4x128, MTP D4, four rounds, outer capacity20. Four warmup requests are excluded. Every measured trace is matched step by step to per-request Target counters. Only cycles that reach Target verification are included; final unused proposals are counted separately in round_audit.json. All rounds use the same matched cycles.

Target yield includes correction/bonus tokens appended by Pre-Verify. Returned yield also clips the last cycle to the 128-token output limit. A zero Target contribution can result from an earlier-round rejection; it does not alone prove that this round has worse local quality. The trace writes synchronize and add I/O, so these runs do not establish performance or an optimal number of rounds. Existing Gemma AR repeat and speculative output parity limitations remain unresolved.

| Method | Round | Inner accepted | Target accepted | Returned accepted | Zero Target |
| --- | ---: | ---: | ---: | ---: | ---: |
| h4 | 1 | 2.370 | 3.065 | 2.913 | 17.4% |
| h4 | 2 | 2.543 | 2.891 | 2.565 | 30.4% |
| h4 | 3 | 2.696 | 2.739 | 2.413 | 37.0% |
| h4 | 4 | 2.630 | 2.652 | 2.239 | 43.5% |
| h6 | 1 | 2.425 | 3.200 | 3.150 | 12.5% |
| h6 | 2 | 2.575 | 3.075 | 2.800 | 25.0% |
| h6 | 3 | 3.075 | 3.400 | 3.025 | 27.5% |
| h6 | 4 | 3.175 | 3.300 | 2.825 | 32.5% |
| h8 | 1 | 2.676 | 3.676 | 3.618 | 0.0% |
| h8 | 2 | 3.088 | 4.000 | 3.853 | 2.9% |
| h8 | 3 | 2.882 | 3.824 | 3.471 | 2.9% |
| h8 | 4 | 2.706 | 3.647 | 3.118 | 2.9% |
| routing60 | 1 | 2.326 | 2.957 | 2.957 | 19.6% |
| routing60 | 2 | 2.217 | 2.522 | 2.522 | 41.3% |
| routing60 | 3 | 2.435 | 2.435 | 2.435 | 45.7% |
| routing60 | 4 | 2.783 | 2.304 | 2.196 | 52.2% |
| attention60 | 1 | 2.432 | 2.977 | 2.932 | 22.7% |
| attention60 | 2 | 2.523 | 2.795 | 2.682 | 31.8% |
| attention60 | 3 | 2.636 | 2.727 | 2.568 | 40.9% |
| attention60 | 4 | 2.955 | 2.705 | 2.455 | 45.5% |

Reproduce: `.venv/bin/python benchmarks/hierarchical/run_round_decay.py <fresh_directory> --gpu 1`, then `.venv/bin/python benchmarks/hierarchical/analyze_round_decay.py <fresh_directory>`. AI assistance was used.
