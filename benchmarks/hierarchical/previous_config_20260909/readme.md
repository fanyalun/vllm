# Previous-configuration measurement evidence

[results.md](results.md) describes the configuration, results and limitations.
The completion scope is performance and acceptance measurement; strict model
equivalence remains failed. [measurement_complete.json](measurement_complete.json)
records that scope explicitly.

- [summary.csv](summary.csv): all eight hierarchical configurations and counters.
- [comparison.csv](comparison.csv): the plotted historical and new values.
- [audit.json](audit.json): coverage, per-step count checks and raw-file hashes.
- [validation.json](validation.json): measurement tests and figure checks.
- `baselines/`: historical CSVs, AR token IDs and configuration snapshots.
- `figures/previous_comparison/`: the complete PNG, PDF and figure notes.
- `raw_evidence.tar.gz`: 24 cells, each containing its command, configuration,
  output token IDs, raw verification/timing records, log and completion marker.
  The interrupted pilot is excluded and remains in the local run directory.

From the repository root, verify and unpack the evidence:

```bash
(cd benchmarks/hierarchical/previous_config_20260909 && sha256sum -c checksums.sha256)
tar -xzf benchmarks/hierarchical/previous_config_20260909/raw_evidence.tar.gz \
  -C benchmarks/hierarchical/previous_config_20260909
```

Recompute the tables, audit and figure from the unpacked files:

```bash
.venv/bin/python benchmarks/hierarchical/summarize_previous.py \
  --run-dir benchmarks/hierarchical/previous_config_20260909
```

The live, uncompressed run is retained at
`benchmark_results/hierarchical_previous_config_20260909`. Reproduction of the
GPU measurements is documented in [results.md](results.md). AI assistance was
used for the measurement scripts and analysis.
