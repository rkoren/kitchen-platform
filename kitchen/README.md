# rkoren-kitchen 2.0 (alpha)

A small CLI harness that AI coding agents work inside for ML competitions and similar projects:
it records any run (job, script, heredoc) with git provenance, scores out-of-fold predictions
against frozen splits, compares runs with a paired cluster bootstrap, and gates promotion behind
a human. Install with `uv tool install rkoren-kitchen`; see the repository README.
