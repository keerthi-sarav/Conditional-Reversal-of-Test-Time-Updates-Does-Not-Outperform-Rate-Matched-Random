# Superseded - invalid data, retained for audit only

`random_revert_seed{1,2,3}.json` in this folder are the **pre-fix** logs of the rate-matched
random-revert control. They were produced by a version of the runner that used a hardcoded 0.5
accept probability instead of matching each seed's realized gate accept rate. Their realized accept
rates are approximately 0.499 / 0.479 / 0.466, clustered on that 0.5 default, rather than the gate's
per-seed rates of 0.350 / 0.247 / 0.306.

This bug is Addendum 5 of the pre-registration. It was found during analysis and the condition was
re-run correctly before any reported result was finalized, so **no published number comes from these
files**. The corrected re-run overwrote them on disk and its per-batch logs were then lost
(Addendum 7); only these pre-fix copies survived.

Do not use them as the random-revert control. An unmatched control confounds revert rate with revert
rule, which is exactly the confound the matched control exists to remove. The corrected condition's
surviving per-seed aggregates are in `../derived/main_grid_results.csv`.
