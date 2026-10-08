# Code

| File | Protocol step | Purpose |
|---|---|---|
| `corruptions.py` | 2 | 17 re-implemented corruption functions, applied at severity 5 |
| `cifar_resnet.py` | 1.2 | 3-stage 16/32/64-channel residual network for 32x32 inputs |
| `stream_runner.py` | 3 | Cyclic non-IID stream constructor and per-condition run loop |
| `main_grid_runner.py` | 7 | Main grid: 5 conditions x 3 seeds, 945 batches each |
| `main_grid_runner_short.py` | 6.2 | Go/no-go pilot |
| `main_grid_rerun_random_revert.py` | 8.3 | Corrected rate-matched control re-run (Addendum 5) |
| `ablation_runner.py` | - | Tolerance / criterion / cadence sweep (Addendum 6) |
| `stats_analysis.py` | 8.2 | Pre-registered tests: paired Wilcoxon, rank-biserial, bootstrap CI |
| `figures/*.py` | - | Figure generation, one file per figure |

Not included: `tta_methods.py`, the module implementing the gate and the comparison policies. It was
lost before deposition (Addendum 7). The runners import it, so they will not execute as-is; the gate
must be reimplemented from the manuscript's protocol steps 4-6.

Environment: eight-core CPU, no accelerator. Python 3.12 with PyTorch, NumPy, SciPy, pandas,
Matplotlib and openpyxl.
