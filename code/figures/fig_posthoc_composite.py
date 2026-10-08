"""
Figure 2 - post hoc composite
Source: fig_posthoc_composite.png

Extracted verbatim from this study's execution record (the cell that produced the
figure). It ran inside an interactive session that already held the loaded logs and
the figure-style helpers, so it is a faithful record of what generated the figure
rather than a turnkey script; run it after loading the logs as the runners do.
"""

import json
import glob
import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib.pyplot as plt
import matplotlib as mpl

# skill:apply_figure_style kernel.py (auto-injected on skill load)
def apply_figure_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.dpi': 150,
    })

def set_frame(ax_or_fig, style=None):
    if isinstance(ax_or_fig, mpl.figure.Figure):
        return
    ax = ax_or_fig
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

META_GREY = "#888888"
GC = "#1a6fba"

def focal_palette(conditions, focal=None, focal_color="#1a6fba", other="muted"):
    base_colors = ["#e08214","#8073ac","#b2182b","#666666","#1a6fba","#4dac26","#d01c8b"]
    result = []
    other_idx = 0
    for c in conditions:
        if c == focal:
            result.append(focal_color)
        else:
            color = base_colors[other_idx % len(base_colors)]
            if color == focal_color:
                other_idx += 1
                color = base_colors[other_idx % len(base_colors)]
            result.append(color)
            other_idx += 1
    return result

apply_figure_style()

# Load main grid results
import shutil, os

def robustness_floor(accs, window=50):
    accs = np.array(accs, dtype=float)
    w = min(window, len(accs))
    csum = np.cumsum(accs)
    csum = np.insert(csum, 0, 0)
    window_means = (csum[w:] - csum[:-w]) / w
    return window_means.min(), window_means

# Load logs from artifact store
LOGS = {}
conds = ["no_adapt", "tent", "cotta", "gated_tent"]
seeds = [1, 2, 3]

log_files = {
    ("no_adapt", 1): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/d2e07725-1756-4b4d-bec0-f68f1ecdbdb2/v1ad9da3c_no_adapt_seed1.json",
    ("no_adapt", 2): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/b4ae8ed0-7f48-492d-8775-bd4ea121153a/v237b0ae0_no_adapt_seed2.json",
    ("no_adapt", 3): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/b5be8a48-d4bd-495e-a149-cc0901d40877/vdc5966a5_no_adapt_seed3.json",
    ("tent", 1): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/90648894-316a-4619-91bd-709e993d4366/ve8cb4814_tent_seed1.json",
    ("tent", 2): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/594e5ba1-b3e9-4b7e-bff5-f28bc009e4fd/v0121f9bd_tent_seed2.json",
    ("tent", 3): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/34b5a0e4-1aad-4a93-bb22-2964bbd59ac7/v6aca2ea7_tent_seed3.json",
    ("cotta", 1): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/99f98be4-25e5-433a-bf67-3a9f9482d573/v307fc1d3_cotta_seed1.json",
    ("cotta", 2): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/e9a73b2a-b445-47c5-9ff7-19c37269401b/v3e72fdb3_cotta_seed2.json",
    ("cotta", 3): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/96daeadd-99ac-4600-8fb7-355a7c91b667/v5e52e32b_cotta_seed3.json",
    ("gated_tent", 1): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/8deffa2e-0763-43f8-9533-4eef35abeb0b/v0e5c57be_gated_tent_seed1.json",
    ("gated_tent", 2): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/b8b5320f-1917-4d24-a9c3-644d0090c86e/v09613890_gated_tent_seed2.json",
    ("gated_tent", 3): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/c2d5f65d-69c5-42ce-911c-f9f8bece422a/v3d0b6892_gated_tent_seed3.json",
}

for c in conds:
    LOGS[c] = {}
    for s in seeds:
        LOGS[c][s] = json.load(open(log_files[(c, s)]))

BLOCK = 63
N_BLOCKS = 15
WIN = 50
VALID = ["no_adapt", "tent", "cotta", "gated_tent"]
VALID_LBL = {"no_adapt": "No adaptation", "tent": "Ungated TENT",
             "cotta": "CoTTA-style restore", "gated_tent": "Gated TENT (proposed)"}

def acc_arr(c, s):
    return np.array([r["acc"] for r in LOGS[c][s]["log"]], dtype=float)

def rolling(a, w=WIN):
    if len(a) < w:
        w = len(a)
    return np.convolve(a, np.ones(w)/w, mode="valid")

# per-corruption breakdown
rows = []
for c in VALID:
    for s in seeds:
        a = acc_arr(c, s)
        lg = LOGS[c][s]["log"]
        for bi in range(N_BLOCKS):
            sl = slice(bi*BLOCK, (bi+1)*BLOCK)
            ctype = lg[bi*BLOCK]["corruption"]
            rows.append({"condition": c, "seed": s, "block_index": bi,
                         "corruption": ctype, "block_acc": a[sl].mean()})
per_block = pd.DataFrame(rows)
per_corr_v = (per_block.groupby(["condition", "corruption"])["block_acc"]
              .agg(["mean", "std"]).reset_index()
              .rename(columns={"mean": "acc_mean", "std": "acc_sd"}))

# collapse windows
collapse_rows = []
for c in VALID:
    for s in seeds:
        base = rolling(acc_arr("no_adapt", s))
        cur = rolling(acc_arr(c, s))
        n_below = int((cur < base).sum())
        collapse_rows.append({"condition": c, "seed": s,
                               "n_windows": len(cur),
                               "collapse_windows": n_below,
                               "collapse_frac": n_below/len(cur),
                               "worst_deficit_pp": float((base - cur).max()*100)})
collapse_v = pd.DataFrame(collapse_rows)

# gate dynamics
gate_rows = []
for s in seeds:
    lg = LOGS["gated_tent"][s]["log"]
    for bi in range(N_BLOCKS):
        sl = lg[bi*BLOCK:(bi+1)*BLOCK]
        gate_rows.append({
            "condition": "gated_tent", "seed": s, "block_index": bi,
            "corruption": sl[0]["corruption"],
            "accept_rate": float(np.mean([r["accepted"] for r in sl])),
            "block_acc": float(np.mean([r["acc"] for r in sl])),
        })
gate_v = pd.DataFrame(gate_rows)

noad = per_corr_v[per_corr_v.condition == "no_adapt"][["corruption", "acc_mean"]].rename(
    columns={"acc_mean": "source_acc"})

gt = (gate_v.groupby("corruption")
      .agg(accept_rate_mean=("accept_rate", "mean")).reset_index())
gd = gt.merge(noad, on="corruption")
rho, p_rho = st.spearmanr(gd.source_acc, gd.accept_rate_mean)

# order corruptions by frozen-source accuracy (hardest -> easiest)
order = (per_corr_v[per_corr_v.condition == "no_adapt"]
         .sort_values("acc_mean")["corruption"].tolist())

fig = plt.figure(figsize=(13.0, 9.6))
gs_layout = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], wspace=0.30, hspace=0.42)
axA = fig.add_subplot(gs_layout[0, 0])
axB = fig.add_subplot(gs_layout[0, 1])
axC = fig.add_subplot(gs_layout[1, 0])
axD = fig.add_subplot(gs_layout[1, 1])

# --- Panel A: per-corruption heatmap ---
mat = (per_corr_v.pivot(index="corruption", columns="condition", values="acc_mean").loc[order, VALID])
disp = [VALID_LBL[c].replace(" (proposed)", "").replace("-style restore", "") for c in VALID]
im = axA.imshow(mat.values, cmap="viridis", aspect="auto", vmin=0, vmax=0.9)
axA.set_xticks(range(len(VALID)))
axA.set_xticklabels(disp, rotation=20, ha="right", fontsize=8)
axA.set_yticks(range(len(order)))
axA.set_yticklabels([c.replace("_", " ") for c in order], fontsize=7.5)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j]
        axA.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                 color="white" if v < 0.45 else "black")
axA.set_title("(A) Mean accuracy per corruption type", fontsize=10)
cb = fig.colorbar(im, ax=axA, fraction=0.035, pad=0.02)
cb.ax.tick_params(labelsize=7)

# --- Panel B: collapse windows ---
cw_piv = collapse_v.pivot(index="condition", columns="seed", values="collapse_windows").reindex(VALID)
xpos = np.arange(len(VALID))
for si, s in enumerate([1, 2, 3]):
    axB.scatter(xpos + (si-1)*0.12, cw_piv[s], s=45, color=GC, alpha=0.8, zorder=3,
                marker="o", edgecolor="white", linewidth=0.5)
axB.scatter(xpos, cw_piv.mean(axis=1), marker="_", s=500, color="#c0392b", linewidth=2, zorder=4)
axB.set_xticks(xpos)
axB.set_xticklabels(disp, rotation=20, ha="right", fontsize=8)
axB.set_ylabel("Collapse windows (of 896)", fontsize=9)
axB.set_title("(B) Windows below frozen-source model\n(lower better; 3 seeds + mean)", fontsize=10)
axB.set_ylim(-45, 950)
axB.set_yticks([0, 200, 400, 600, 800])
set_frame(axB)

# --- Panel C: gate accept rate by block ---
gv2 = (gate_v.groupby("block_index").agg(ar_mean=("accept_rate", "mean"), ar_sd=("accept_rate", "std"),
       corruption=("corruption", "first")).reset_index())
axC.bar(gv2.block_index, gv2.ar_mean, yerr=gv2.ar_sd, width=0.70, color=GC, alpha=0.88,
        error_kw=dict(lw=1.0, ecolor="#555", capsize=2.5), zorder=3)
overall2 = gate_v.accept_rate.mean()
axC.axhline(overall2, ls="--", lw=1.2, color=META_GREY, zorder=2)
axC.text(-0.45, 0.595, f"stream mean = {overall2:.2f}", fontsize=7, color=META_GREY, ha="left", va="top")
axC.set_xticks(gv2.block_index)
axC.set_xticklabels([c.replace("_", " ") for c in gv2.corruption], rotation=90, fontsize=6.5)
axC.set_ylabel("Fraction of updates accepted", fontsize=9)
axC.set_xlabel("Corruption block, in stream order", fontsize=9)
axC.set_ylim(0, 0.62)
axC.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
axC.set_xlim(-0.75, 14.75)
axC.set_title("(C) Gate accept rate by corruption block", fontsize=10)
set_frame(axC)

# --- Panel D: accept rate vs source accuracy scatter ---
xx = np.linspace(0.08, 0.96, 50)
sl_fit, ic = np.polyfit(gd.source_acc, gd.accept_rate_mean, 1)
axD.scatter(gd.source_acc, gd.accept_rate_mean, s=48, color=GC, zorder=3, edgecolor="white", linewidth=0.7)
axD.plot(xx, sl_fit*xx + ic, color=META_GREY, lw=1.3, zorder=2)
for _, rw in gd.iterrows():
    if rw.corruption in ("glass_blur", "brightness", "contrast", "zoom_blur"):
        dx, dy = (6, -10) if rw.corruption != "brightness" else (-8, -12)
        axD.annotate(rw.corruption.replace("_", " "), (rw.source_acc, rw.accept_rate_mean),
                     textcoords="offset points", xytext=(dx, dy), fontsize=6.5, color="#333",
                     ha="right" if rw.corruption == "brightness" else "left")
axD.set_xlabel("Frozen-source accuracy on that corruption", fontsize=9)
axD.set_ylabel("Gate accept rate", fontsize=9)
axD.set_title(f"(D) Accept rate vs. source accuracy\n(Spearman ρ={rho:.2f}, p={p_rho:.3f})", fontsize=10)
axD.set_xlim(0.08, 0.96)
axD.set_ylim(0.03, 0.52)
axD.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axD.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
set_frame(axD)

fig.suptitle("Post hoc secondary analyses — per-corruption accuracy, collapse frequency, and gate decision "
             "dynamics (random-revert control excluded, Addendum 7)", x=0.012, ha="left", fontsize=10)
fig.subplots_adjust(top=0.90)
fig.savefig("fig_posthoc_composite.png", dpi=200, bbox_inches="tight")
