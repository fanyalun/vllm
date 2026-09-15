# Validation

All five cells completed four measured requests of exactly 128 output tokens.
The audit matched 210 final Target verification cycles and 840 inner rounds,
with identical cycle sets for round positions 1 through 4 within each method.
Each cell excludes four warmup requests and 16 inner rounds belonging to final
unused proposals. Every trace's scheduled and accepted counts match the saved
per-request metrics. Expert policy and model weights were not modified.

The interval accounting helper passed four CPU examples: accepting 6 out of
16 candidates in four equal blocks gives contributions `[4, 2, 0, 0]`; zero
acceptance gives all zeros; full acceptance gives `[4, 4, 4, 4]`; clipping full
acceptance to nine returned tokens gives `[4, 4, 1, 0]`. The real-data audit also
checks per-cycle sums and the 128-token request output boundary.

Applicable pre-commit hooks passed for both new scripts. The exported PNG was
visually checked, the PDF has one page, and its rendered image was checked.
No dependencies were installed. The raw diagnostic logs and traces are retained.

The staged-data spelling hook flags `daa` inside the generated request ID
`4-9cf41daa` in the raw trace and derived CSV. This is an identifier, not a
misspelling. Source spelling checks passed; the commit skips only the spelling
hook to preserve those exact diagnostic identifiers.

The sample is exploratory. In all four attention60 requests, round 4 inner
acceptance exceeds round 1. This does not establish a universal trend or explain
the unresolved Gemma output instability. These runs do not compare 1/2/3/4-round
execution policies and must not be used to claim an optimal stopping rule or
end-to-end acceleration.
