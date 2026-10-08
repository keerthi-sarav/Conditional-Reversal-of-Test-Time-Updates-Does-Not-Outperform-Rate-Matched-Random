# Data

## raw_logs/
One JSON per condition x seed. Each holds a `log` array with one record per batch carrying batch
accuracy, the corruption type in effect, and for the reverting conditions the accept/reject decision
and any rollback event; plus run-level fields including the realized accept rate and wall-clock time.

Present: `no_adapt`, `tent`, `cotta`, `gated_tent` at seeds 1-3 (12 files), and `_manifest.json`
recording conditions, seeds, hyperparameters and timings.

Absent: the corrected `random_revert` logs (Addendum 7). See `../README.md`.

Stream geometry: 945 batches of 32 images = 15 corruption blocks x 63 batches, one cycle, each block
presented consecutively to induce the non-IID collapse the study measures.

## derived/
| File | Contents |
|---|---|
| `main_grid_results.csv` | Per-seed, per-condition mean accuracy, robustness floor, realized accept rate. **Authoritative for the random-revert control.** |
| `corruption_validation_sweep.csv` | Accuracy by corruption x severity; the monotonicity check of step 2.3 |
| `per_corruption_breakdown.csv` | Per-corruption accuracy by condition (post hoc) |
| `new_analyses.csv` | Collapse windows, worst deficit, post-switch dip, recovery lag (post hoc) |
| `stats_new_analyses.csv` | Paired tests for the post hoc metrics; every row flagged not pre-registered |
| `gate_dynamics.csv` | Gate accept rate per corruption block, with the Spearman correlation |
| `ablation_results.csv` | Accept rate, revert rate, accuracy per tolerance multiplier / criterion / cadence |

Post hoc files exclude the random-revert control and therefore bear on no pre-registered hypothesis.

## superseded/
Invalid pre-fix control logs, retained for audit. See `DO_NOT_USE.md`.

Third-party inputs (not redistributed here): the public ten-class 32x32 image test split described in
the manuscript's reference 2, and the public pretrained residual-network checkpoints.
