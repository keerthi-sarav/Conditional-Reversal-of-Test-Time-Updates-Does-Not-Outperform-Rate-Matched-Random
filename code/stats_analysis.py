"""
Statistical analysis exactly as frozen in preregistration.md Sec.6, run against
the corrected main grid (Addendum 5: random_revert accept-rate-matching bug fixed).

Primary endpoint: robustness floor = min over a 50-batch sliding window of mean
accuracy across the full stream (per condition, per seed).

Tests:
  1. Paired Wilcoxon signed-rank, gated_tent vs tent (ungated), n=3 seed pairs.
  2. Paired Wilcoxon signed-rank, gated_tent vs random_revert (matched control),
     n=3 seed pairs -- the decisive second test.
  3. Effect size: matched-pairs rank-biserial correlation, with a bootstrap 95%
     CI (2000 resamples) on the mean paired difference.

Success threshold (verbatim from preregistration.md Sec.6): gated TENT must beat
BOTH ungated TENT and the random-revert control on robustness floor, with the
paired difference's 95% CI excluding zero on both comparisons, in BOTH
architectures (ResNet-20 and ResNet-56), at n=5 seeds.

KNOWN, DISCLOSED SHORTFALL (Addendum 3): this run has n=3 seeds, ResNet-20 only.
The full success threshold as pre-registered cannot be reached from this data
regardless of which way the numbers fall -- reported honestly as a partial/
directionally-informative result per preregistration.md Sec.7, not reframed.
"""
import json, glob
import numpy as np
from scipy import stats

RESULTS_DIR = "results/main_grid"
SEEDS = [1, 2, 3]
WINDOW = 50
N_BOOT = 2000
RNG = np.random.default_rng(0)


def robustness_floor(accs, window=WINDOW):
    accs = np.array(accs)
    if len(accs) < window:
        window = len(accs)
    csum = np.cumsum(accs)
    csum = np.insert(csum, 0, 0)
    window_means = (csum[window:] - csum[:-window]) / window
    return float(window_means.min())


def load_floor(condition, seed):
    d = json.load(open(f"{RESULTS_DIR}/{condition}_seed{seed}.json"))
    accs = [r["acc"] for r in d["log"]]
    return robustness_floor(accs)


def rank_biserial(diffs):
    """Matched-pairs rank-biserial correlation from signed Wilcoxon ranks."""
    diffs = np.asarray(diffs)
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    r_plus = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    total = r_plus + r_minus
    return float((r_plus - r_minus) / total) if total > 0 else 0.0


def bootstrap_ci_mean_diff(diffs, n_boot=N_BOOT, rng=RNG):
    diffs = np.asarray(diffs)
    n = len(diffs)
    boots = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi), float(diffs.mean())


def paired_test(cond_a, cond_b, label):
    floors_a = np.array([load_floor(cond_a, s) for s in SEEDS])
    floors_b = np.array([load_floor(cond_b, s) for s in SEEDS])
    diffs = floors_a - floors_b
    try:
        stat, p = stats.wilcoxon(floors_a, floors_b)
    except ValueError as e:
        stat, p = np.nan, np.nan
        wilcoxon_note = str(e)
    else:
        wilcoxon_note = ""
    rb = rank_biserial(diffs)
    lo, hi, mean_diff = bootstrap_ci_mean_diff(diffs)
    ci_excludes_zero = (lo > 0) or (hi < 0)
    return {
        "comparison": label,
        "cond_a": cond_a, "cond_b": cond_b,
        "floors_a": floors_a.tolist(), "floors_b": floors_b.tolist(),
        "n_pairs": len(SEEDS),
        "wilcoxon_stat": float(stat) if stat == stat else None,
        "wilcoxon_p": float(p) if p == p else None,
        "wilcoxon_note": wilcoxon_note,
        "rank_biserial_r": rb,
        "mean_paired_diff": mean_diff,
        "bootstrap_ci_95": [lo, hi],
        "ci_excludes_zero": bool(ci_excludes_zero),
        "direction": "gated favored" if mean_diff > 0 else "control/baseline favored",
    }


if __name__ == "__main__":
    results = {}
    results["gated_vs_ungated"] = paired_test("gated_tent", "tent", "gated_tent vs ungated TENT")
    results["gated_vs_random_revert"] = paired_test("gated_tent", "random_revert",
                                                       "gated_tent vs random-revert control (matched, corrected)")

    both_exclude_zero = (results["gated_vs_ungated"]["ci_excludes_zero"] and
                          results["gated_vs_random_revert"]["ci_excludes_zero"])
    both_favor_gated = (results["gated_vs_ungated"]["mean_paired_diff"] > 0 and
                         results["gated_vs_random_revert"]["mean_paired_diff"] > 0)
    success_this_arch = both_exclude_zero and both_favor_gated

    verdict = {
        "success_threshold_as_preregistered": "NOT REACHABLE from this data: requires n=5 seeds "
            "in BOTH ResNet-20 and ResNet-56 (Sec.6); this run has n=3 seeds, ResNet-20 only "
            "(Addendum 3, disclosed compute-driven scope cut).",
        "resnet20_n3_directional_result": (
            "gated_tent beats both ungated TENT and the random-revert control on robustness floor, "
            "with both paired-difference 95% CIs excluding zero, at n=3 seeds, ResNet-20 only."
            if success_this_arch else
            "gated_tent does NOT clear both comparisons with both CIs excluding zero at n=3 seeds, "
            "ResNet-20 only -- reported as a partial/null result per preregistration.md Sec.7, not "
            "reframed."
        ),
        "both_ci_exclude_zero": both_exclude_zero,
        "both_favor_gated_direction": both_favor_gated,
    }

    out = {"results": results, "verdict": verdict}
    json.dump(out, open("stats_report_raw.json", "w"), indent=1)

    import csv
    with open("stats_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "cond_a", "cond_b", "n_pairs", "mean_paired_diff",
                    "wilcoxon_stat", "wilcoxon_p", "rank_biserial_r",
                    "bootstrap_ci_low", "bootstrap_ci_high", "ci_excludes_zero", "direction"])
        for r in results.values():
            w.writerow([r["comparison"], r["cond_a"], r["cond_b"], r["n_pairs"],
                        f"{r['mean_paired_diff']:.4f}", r["wilcoxon_stat"], r["wilcoxon_p"],
                        f"{r['rank_biserial_r']:.3f}", f"{r['bootstrap_ci_95'][0]:.4f}",
                        f"{r['bootstrap_ci_95'][1]:.4f}", r["ci_excludes_zero"], r["direction"]])

    print(json.dumps(out, indent=1))
