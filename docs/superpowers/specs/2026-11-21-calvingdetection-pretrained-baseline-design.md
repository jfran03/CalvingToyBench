# Add the authors' pretrained AE + RF as benchmark comparators (short design)

Date: 2026-11-21. Status: awaiting approval.

## Intent
The LLMs only beat our simple baselines by a small margin. To judge that margin we want a
reference from the dataset authors' own trained pipeline (Cao et al. 2025, CalvingDetection:
autoencoder (AE) anomaly detector, then random forest (RF) classifier of the anomalies),
scored by the same `score.py` on the same 735 windows. The user chose to use the pretrained
files as published, not to retrain.

## Goals / non-goals / anti-goals
- Goal: two new rows in `score.py` output, written as `results/*.jsonl` in the existing format
  (`id, cow, label, hours_to_calving, prob`), so `score.py` and `plot_results.py` need no changes.
- Non-goal: retraining, re-implementing the AE, tuning anything on our labels.
- Anti-goal: presenting these as a like-for-like "supervised vs zero-shot" without the caveats below.

## Verified facts (from the repo and the .pkl files)
- RF: 5 pickles (sklearn 1.9.1 loads them), 42 features: 6h/12h moving average and std, 1h/3h
  diffs of KNN2-5, Distance_moved, ActivityCount_Difference. Trained only on AE-flagged hours;
  label = hour within +-24h of calving. Fold k held out exactly 3 cows; I confirmed this by training-set fit
  (held-out cows score visibly worse) and the 5 folds cover all 15 cows once:
  F1 {09B,0C5,14C}, F2 {09F,0BB,0BD}, F3 {0BC,0BE,0E1}, F4 {0BA,13C,13D}, F5 {0E7,12A,12B}.
  So scoring each cow with its held-out fold is out-of-sample for the RF.
- AE: flags an hour as anomalous if its reconstruction error exceeds the 95th percentile of
  that cow's whole season (so ~5% flagged by construction). The saved `anomalies_from_AE/*.csv` are
  these flags (timestamps only, no error values). The AE trained on each cow's data up to 5 days before calving.

## Approaches considered
1. **Use the authors' saved anomaly CSVs + held-out RF folds (chosen).** Cheap, exactly their outputs.
2. Re-run the `.pth` AE ourselves to get continuous error. Needs the architecture and preprocessing
   reverse-engineered from the notebook; more risk, more code. Deferred.
3. Retrain under leave-one-cow-out. Rejected by the user (not "their trained models").

## Design
New `calvingdetection_baseline.py --repo <CalvingDetection clone>`, writes two files:
- `results/cd__rf_pretrained.jsonl`: for each window at time t, take the AE-flagged hours in the last
  `LOOKBACK_H` (default 12, fixed in advance, backward-looking only), compute the held-out RF fold's
  P(calving-related) at each, prob = max; no flagged hours gives prob 0.
- `results/cd__ae_only.jsonl`: prob = fraction of the last 12 hours that the AE flagged. Shows how much
  the RF adds over the unsupervised stage.
Features are built from the repo's `Processed_data` with the authors' feature code (rolling on the
full hourly series, backward-looking). Windows keep using our `windows.jsonl` ids/labels unchanged.
Print the fold to cow map and a check that no cow is scored by a fold that trained on it.

## Caveats to print in README (not hidden)
1. Label mismatch: RF trained on +-24h around birth including after; our positive is [t-6h, t+24h).
2. The AE threshold uses the cow's entire season (non-causal) and the AE saw each cow's days before -5 in training.
   Unsupervised, no labels, but not deployable as-is.
3. The RF only ever scores AE-flagged hours, so the window score is mostly 0 with ties; AUROC and
   the threshold-0.5 metrics are coarse. RF's own probabilities, not calibrated to our prior.
4. Cows in training folds of other cows' pickles are fine, but the paper's headline (per-cow cluster F1)
   is a different metric from ours; we do not reproduce it.

## Left to the builder
File layout, loading with warnings silenced for sklearn version drift, a mock-free smoke test:
per-cow row counts must match, and a sanity check that scoring a cow with a fold that trained on it gives
much higher AUROC than the held-out fold (proves the fold map is right).
