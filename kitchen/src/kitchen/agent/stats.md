# Reading kitchen verdicts

- A verdict compares two runs on the *same* frozen split rows with the track's metric. kitchen
  resamples clusters — seasons, patients, videos (the split's `cluster`, else its `group`, else
  rows) — 2,000 times, paired, and reports `delta = metric(A) - metric(B)`, its 95% CI and
  `p_better`.
- `better`: the whole CI is on the improving side and the gain meets the track's `min_delta`.
  `worse`: the whole CI is on the wrong side. `tie`: the CI straddles zero — no improvement has
  been shown, however good the point estimate looks.
- `incomparable`: different split or metric, partial coverage, missing outputs, or a quarantined
  run. Fix the cause; don't eyeball raw scores instead.
- Few clusters (e.g. ~20 seasons) mean wide CIs. That is the truth about the evidence, not a bug.
  Don't slice the data until something looks significant.
- Seed noise: re-run a config with more seeds (`-p seed=1,2,3 --op confirm`). With >= 3 seed
  replicates, promotion also requires |delta| > 2 x the seed SD.
- A suspiciously large gain usually means leakage (future data, target-derived features, test rows
  in training). Investigate before celebrating; `kitchen mark <run> --quarantine "<why>"` removes a
  leaky run from comparisons, and the brief keeps the warning visible.
- Leaderboard scores (Kaggle) are a second, noisy signal recorded next to CV — never a reason to
  promote on their own; public leaderboards are often tiny.
- Holdout splits (`scheme: holdout`) are for occasional confirmation, not for iterating against:
  every look leaks a little information into your choices.
