"""
per-corruption + collapse (component of Fig 2)
Source: fig_per_corruption.png

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


def focal_palette(labels, focal, focal_color, other="muted", base_colors=None):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    focal_set = {focal} if isinstance(focal, str) else set(focal)
    n = len(labels)
    if not focal_set & set(labels):
        raise ValueError(f"focal {focal!r} not found in labels")
    if base_colors is None:
        base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#444444"])
    base_colors = [base_colors[i % len(base_colors)] for i in range(n)]
    if other == "grey":
        rest = ["#BCBCBC"] * n
    elif other == "ordinal":
        nf = max(1, n - len(focal_set))
        ramp = [mcolors.to_hex((v, v, v)) for v in
                ([0.55] if nf == 1 else [0.80 - 0.35 * i / (nf - 1) for i in range(nf)])]
        rest, k = [], 0
        for l in labels:
            rest.append(ramp[min(k, nf - 1)]); k += (l not in focal_set)
    else:
        def mute(c):
            r, g, b = mcolors.to_rgb(c)
            m = (r + g + b) / 3
            return mcolors.to_hex((0.3 * r + 0.7 * m, 0.3 * g + 0.7 * m, 0.3 * b + 0.7 * m))
        rest = [mute(c) for c in base_colors]
    return [focal_color if l in focal_set else rest[i] for i, l in enumerate(labels)]


# Load data
LOGS = {}
files = {
    "no_adapt": {
        1: '/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/d2e07725-1756-4b4d-bec0-f68f1ecdbdb2/v1ad9da3c_no_adapt_seed1.json',
        2: '/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/b4ae8ed0-7f48-492d-8775-bd4ea121153a/v237b0ae0_no_adapt_seed2.json',
        3: '/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/b5be8a48-d4bd-495e-a149-cc0901d40877/vdc5966a5_no_adapt_seed3.json',
    },
    "tent": {
        1: '/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/90648894-316a-4619-91bd-709e993d4366/ve8cb4814_tent_seed1.json',
        2: '/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/594e5ba1-b3e9-4b7e-bff5-f28bc009e4fd/v0121f9bd_tent_seed2.json',
        3: '/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/34b5a0e4-1aad-4a93-bb22-2964bbd59ac7/v6aca2ea7_tent_seed3.json',
    },
    "cotta": {
        1: '/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/99f98be4-25e5-433a-bf67-3a9f9482d573/v307fc1d3_cotta_seed1.json',
        2: '/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/e9a73b2a-b445-47c5-9ff7-19c37269401b/v3e72fdb3_cotta_seed2.json',
        3: '/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/96daeadd-99ac-4600-8fb7-355a7c91b667/v5e52e32b_cotta_seed3.json',
    },
    "gated_tent": {
        1: '/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/8deffa2e-0763-43f8-9533-4eef35abeb0b/v0e5c57be_gated_tent_seed1.json',
        2: '/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/b8b5320f-1917-4d24-a9c3-644d0090c86e/v09613890_gated_tent_seed2.json',
        3: '/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/c2d5f65d-69c5-42ce-911c-f9f8bece422a/v3d0b6892_gated_tent_seed3.json',
    },
}

for cond, seed_files in files.items():
    LOGS[cond] = {}
    for s, path in seed_files.items():
        LOGS[cond][s] = json.load(open(path))

conds = ["no_adapt", "tent", "cotta", "gated_tent"]
SEEDS = (1, 2, 3)
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
    return np.convolve(a, np.ones(w) / w, mode="valid")


# Analysis 1: per-corruption-type accuracy
import pandas as pd

rows = []
for c in conds:
    for s in SEEDS:
        a = acc_arr(c, s)
        corr = [r["corruption"] for r in LOGS[c][s]["log"]]
        for bi in range(N_BLOCKS):
            sl = slice(bi * BLOCK, (bi + 1) * BLOCK)
            ctype = corr[bi * BLOCK]
            rows.append({"condition": c, "seed": s, "block_index": bi,
                         "corruption": ctype, "block_acc": a[sl].mean()})
per_block = pd.DataFrame(rows)
per_corr = (per_block.groupby(["condition", "corruption"])["block_acc"]
            .agg(["mean", "std"]).reset_index()
            .rename(columns={"mean": "acc_mean", "std": "acc_sd"}))
per_corr_v = per_corr[per_corr.condition.isin(VALID)].copy()

# Analysis 2: collapse events vs frozen-source baseline
coll = []
for c in conds:
    for s in SEEDS:
        base = rolling(acc_arr("no_adapt", s))
        cur = rolling(acc_arr(c, s))
        n_below = int((cur < base).sum())
        coll.append({"condition": c, "seed": s,
                     "n_windows": len(cur),
                     "collapse_windows": n_below,
                     "collapse_frac": n_below / len(cur),
                     "worst_deficit_pp": float((base - cur).max() * 100)})
collapse_df = pd.DataFrame(coll)
collapse_v = collapse_df[collapse_df.condition.isin(VALID)].copy()

noad = (per_corr_v[per_corr_v.condition == "no_adapt"][["corruption", "acc_mean"]]
        .rename(columns={"acc_mean": "source_acc"}))

# Gate dynamics
gate_rows = []
for c in ["gated_tent"]:
    for s in SEEDS:
        lg = LOGS[c][s]["log"]
        for bi in range(N_BLOCKS):
            sl = lg[bi * BLOCK:(bi + 1) * BLOCK]
            gate_rows.append({
                "condition": c, "seed": s, "block_index": bi,
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

# Plot
apply_figure_style()

order = (per_corr_v[per_corr_v.condition == "no_adapt"]
         .sort_values("acc_mean")["corruption"].tolist())
mat = (per_corr_v.pivot(index="corruption", columns="condition", values="acc_mean")
       .loc[order, VALID])
disp = [VALID_LBL[c].replace(" (proposed)", "").replace("-style restore", "") for c in VALID]

fig = plt.figure(figsize=(12.6, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.30)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])

im = axA.imshow(mat.values, cmap="viridis", aspect="auto", vmin=0, vmax=0.9)
axA.set_xticks(range(len(VALID)))
axA.set_xticklabels(disp, rotation=20, ha="right")
axA.set_yticks(range(len(order)))
axA.set_yticklabels([c.replace("_", " ") for c in order])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j]
        axA.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                 color="white" if v < 0.45 else "black")
axA.set_title("Accuracy by corruption type (mean of 3 seeds)\ncorruptions ordered hardest → easiest for the frozen model")
cb = fig.colorbar(im, ax=axA, fraction=0.040, pad=0.02)
cb.set_label("Mean accuracy in corruption block")
for s in axA.spines.values():
    s.set_visible(False)
axA.tick_params(length=0)

cols = dict(zip(VALID, focal_palette(VALID, focal="gated_tent", focal_color="#1a6fba")))
xs = np.arange(len(VALID))
for s in SEEDS:
    vals = [collapse_v[(collapse_v.condition == c) & (collapse_v.seed == s)]["collapse_windows"].iloc[0]
            for c in VALID]
    axB.scatter(xs + (s - 2) * 0.12, vals, s=42, color=[cols[c] for c in VALID],
                zorder=3, edgecolor="white", linewidth=0.6)
means = [collapse_v[collapse_v.condition == c]["collapse_windows"].mean() for c in VALID]
for i, m in enumerate(means):
    axB.plot([i - 0.24, i + 0.24], [m, m], color=cols[VALID[i]], lw=2.2, zorder=4)
axB.set_xticks(xs)
axB.set_xticklabels(disp, rotation=20, ha="right")
axB.set_ylabel("Collapse windows\n(of 896 sliding windows per seed)")
axB.set_title("Windows falling below the frozen-source model\n(lower = better; 3 seeds shown individually)")
axB.set_ylim(-45, 950)
axB.margins(x=0.10)
axB.set_yticks([0, 200, 400, 600, 800])
set_frame(axB)
fig.suptitle("Post hoc secondary analyses — ResNet-20, 15-corruption cyclic stream "
             "(random-revert control excluded: per-batch logs unavailable, Addendum 7)",
             x=0.012, ha="left", fontsize=9)
fig.subplots_adjust(top=0.84)
fig.savefig("fig_per_corruption.png", dpi=200, bbox_inches="tight")
