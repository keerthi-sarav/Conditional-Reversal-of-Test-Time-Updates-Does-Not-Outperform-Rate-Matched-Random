"""
gate dynamics (component of Fig 2)
Source: fig_gate_dynamics.png

Extracted verbatim from this study's execution record (the cell that produced the
figure). It ran inside an interactive session that already held the loaded logs and
the figure-style helpers, so it is a faithful record of what generated the figure
rather than a turnkey script; run it after loading the logs as the runners do.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats as st

# skill:figure-style kernel.py (auto-injected on skill load)
META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def set_frame(ax, style="open"):
    show = {"open": (False, False, True, True),
            "boxed": (True, True, True, True),
            "none": (False, False, False, False)}[style]
    for side, vis in zip(("top", "right", "bottom", "left"), show):
        ax.spines[side].set_visible(vis)
        if vis:
            ax.spines[side].set_linewidth(0.6)
    ax.tick_params(direction="out", length=0 if style == "none" else 3, width=0.6)


apply_figure_style()

# Load data
SEEDS = (1, 2, 3)
BLOCK = 63
N_BLOCKS = 15
WIN = 50

LOGS = {}
seed_files = {
    ("no_adapt", 1): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/d2e07725-1756-4b4d-bec0-f68f1ecdbdb2/v1ad9da3c_no_adapt_seed1.json",
    ("no_adapt", 2): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/b4ae8ed0-7f48-492d-8775-bd4ea121153a/v237b0ae0_no_adapt_seed2.json",
    ("no_adapt", 3): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/b5be8a48-d4bd-495e-a149-cc0901d40877/vdc5966a5_no_adapt_seed3.json",
    ("gated_tent", 1): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/8deffa2e-0763-43f8-9533-4eef35abeb0b/v0e5c57be_gated_tent_seed1.json",
    ("gated_tent", 2): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/b8b5320f-1917-4d24-a9c3-644d0090c86e/v09613890_gated_tent_seed2.json",
    ("gated_tent", 3): "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/c2d5f65d-69c5-42ce-911c-f9f8bece422a/v3d0b6892_gated_tent_seed3.json",
}

for (c, s), path in seed_files.items():
    if c not in LOGS:
        LOGS[c] = {}
    LOGS[c][s] = json.load(open(path))

def acc_arr(c, s):
    return np.array([r["acc"] for r in LOGS[c][s]["log"]], dtype=float)

# Build per_corr for no_adapt (source accuracy per corruption)
conds_valid = ["no_adapt", "gated_tent"]
VALID_LBL = {"no_adapt": "No adaptation", "gated_tent": "Gated TENT (proposed)"}

import pandas as pd

rows = []
for c in conds_valid:
    for s in SEEDS:
        a = acc_arr(c, s)
        corr = [r["corruption"] for r in LOGS[c][s]["log"]]
        for bi in range(N_BLOCKS):
            sl = slice(bi * BLOCK, (bi + 1) * BLOCK)
            ctype = corr[bi * BLOCK]
            rows.append({"condition": c, "seed": s, "block_index": bi,
                         "corruption": ctype, "block_acc": a[sl].mean()})
per_block = pd.DataFrame(rows)
per_corr_v = (per_block.groupby(["condition", "corruption"])["block_acc"]
              .agg(["mean", "std"]).reset_index()
              .rename(columns={"mean": "acc_mean", "std": "acc_sd"}))

noad = (per_corr_v[per_corr_v.condition == "no_adapt"][["corruption", "acc_mean"]]
        .rename(columns={"acc_mean": "source_acc"}))

# Gate dynamics
gate_rows = []
for s in SEEDS:
    lg = LOGS["gated_tent"][s]["log"]
    for bi in range(N_BLOCKS):
        sl = lg[bi * BLOCK:(bi + 1) * BLOCK]
        gate_rows.append({
            "condition": "gated_tent", "seed": s, "block_index": bi,
            "corruption": sl[0]["corruption"],
            "accept_rate": float(np.mean([r["accepted"] for r in sl])),
            "block_acc": float(np.mean([r["acc"] for r in sl])),
        })
gate_df = pd.DataFrame(gate_rows)
gate_v = gate_df[gate_df.condition == "gated_tent"].copy()

gt = (gate_v.groupby("corruption")
      .agg(accept_rate=("accept_rate", "mean")).reset_index())
gt = gt.merge(noad, on="corruption")
rho, p_rho = st.spearmanr(gt.source_acc, gt.accept_rate)

gd = gate_v.groupby("corruption").agg(
    accept_rate_mean=("accept_rate", "mean"), accept_rate_sd=("accept_rate", "std"),
    block_acc_mean=("block_acc", "mean")).reset_index()
gd = gd.merge(noad, on="corruption").sort_values("source_acc")
gd["spearman_rho_vs_source_acc"] = rho
gd["spearman_p"] = p_rho

gv = (gate_v.groupby("block_index")
      .agg(ar_mean=("accept_rate", "mean"), ar_sd=("accept_rate", "std"),
           corruption=("corruption", "first")).reset_index())
overall = gate_v.accept_rate.mean()

sl_fit, ic = np.polyfit(gd.source_acc, gd.accept_rate_mean, 1)
xx = np.linspace(gd.source_acc.min() - 0.02, gd.source_acc.max() + 0.02, 50)

GC = "#1a6fba"

fig2 = plt.figure(figsize=(12.8, 5.2))
gs2 = fig2.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.24)
axC = fig2.add_subplot(gs2[0, 0])
axD = fig2.add_subplot(gs2[0, 1])

axC.bar(gv.block_index, gv.ar_mean, yerr=gv.ar_sd, width=0.70, color=GC,
        alpha=0.88, error_kw=dict(lw=1.0, ecolor="#555", capsize=2.5), zorder=3)
axC.axhline(overall, ls="--", lw=1.2, color=META_GREY, zorder=2)
axC.text(-0.45, 0.595, f"stream mean accept rate = {overall:.2f}", fontsize=7,
         color=META_GREY, ha="left", va="top")
axC.set_xticks(gv.block_index)
axC.set_xticklabels([c.replace("_", " ") for c in gv.corruption],
                    rotation=90, fontsize=6.5)
axC.set_ylabel("Fraction of updates accepted")
axC.set_xlabel("Corruption block, in stream order")
axC.set_ylim(0, 0.62)
axC.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
axC.set_xlim(-0.75, 14.75)
axC.set_title("Gate accept rate varies by corruption block\n(mean of 3 seeds; error bars = SD across seeds)")
set_frame(axC)

axD.scatter(gd.source_acc, gd.accept_rate_mean, s=52, color=GC, zorder=3,
            edgecolor="white", linewidth=0.7)
axD.plot(xx, sl_fit * xx + ic, color=META_GREY, lw=1.3, zorder=2)
for _, rw in gd.iterrows():
    if rw.corruption in ("glass_blur", "brightness", "contrast", "zoom_blur"):
        dx, dy = (6, -10) if rw.corruption != "brightness" else (-8, -12)
        axD.annotate(rw.corruption.replace("_", " "),
                     (rw.source_acc, rw.accept_rate_mean),
                     textcoords="offset points", xytext=(dx, dy), fontsize=7,
                     color="#333", ha="right" if rw.corruption == "brightness" else "left")
axD.set_xlabel("Frozen-source accuracy on that corruption")
axD.set_ylabel("Gate accept rate")
axD.set_title(f"Gate accepts more where the source model is\nalready accurate "
              f"(Spearman ρ={rho:.2f}, p={p_rho:.3f}, 15 corruptions)")
axD.set_xlim(0.08, 0.96)
axD.set_ylim(0.03, 0.52)
axD.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axD.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
set_frame(axD)

fig2.suptitle("Reversibility-gate decision dynamics — post hoc, gated TENT only, "
              "ResNet-20, n=3 seeds", x=0.012, ha="left", fontsize=9)
fig2.subplots_adjust(top=0.80)
fig2.savefig("fig_gate_dynamics.png", dpi=200, bbox_inches="tight")
