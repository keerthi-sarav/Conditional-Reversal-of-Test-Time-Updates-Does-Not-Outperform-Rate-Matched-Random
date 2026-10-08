"""
Figure 3 - ablation
Source: fig_ablation.png

Extracted verbatim from this study's execution record (the cell that produced the
figure). It ran inside an interactive session that already held the loaded logs and
the figure-style helpers, so it is a faithful record of what generated the figure
rather than a turnkey script; run it after loading the logs as the runners do.
"""

import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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


apply_figure_style()

# Load ablation result files
rows = []

files = {
    "criterion_disagreement_only": "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/a77e8eda-4aa6-41c4-b00e-00882a93081a/v4f6ffa37_criterion_disagreement_only.json",
    "criterion_entropy_only": "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/1bdc258a-8e45-468a-ac2e-7368fece96fa/vb0ed8c6f_criterion_entropy_only.json",
    "ref_resnet20_no_adapt": "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/47856f0f-f108-4483-8737-c7746644154e/vefef133b_ref_resnet20_no_adapt.json",
    "ref_resnet20_tent": "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/d13afd36-90cf-462e-920d-9f0266265cc5/vfdc87047_ref_resnet20_tent.json",
    "snap_every_100": "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/35ed82c6-537d-4a27-b6fe-9fe55b55ad4e/vbcdde1b8_snap_every_100.json",
    "snap_every_25": "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/b9c7caa7-5815-4fc1-98a8-fd1e6017c3e8/vfe19db84_snap_every_25.json",
    "threshold_mult_0.25": "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/610e6828-0ee6-4f0f-975b-05cf6a84f4ec/v65b4bf1a_threshold_mult_0.25.json",
    "threshold_mult_0.5": "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/1a76d473-c59e-4960-8c56-2ec0b2e10541/v362a8e32_threshold_mult_0.5.json",
    "threshold_mult_1.0": "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/e6633b92-8a0c-4f99-b9d4-c3c4021e78df/vc4e282a7_threshold_mult_1.0.json",
    "threshold_mult_2.0": "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/956ec4ac-ef74-4440-b89d-dca0b8cea3b5/v3f0d16f1_threshold_mult_2.0.json",
    "threshold_mult_4.0": "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/6818006c-32fc-4296-85bc-a8ae8d9ba167/vf19b602e_threshold_mult_4.0.json",
}

for job_id, path in files.items():
    d = json.load(open(path))
    rows.append({
        "job_id": d["job_id"], "arch": d["arch"], "condition": d["condition"],
        "tau_h": d["tau_h"], "tau_d": d["tau_d"], "snap_every": d["snap_every"],
        "accept_rate": d.get("accept_rate"), "mean_acc": d["mean_acc"],
    })

abl_df = pd.DataFrame(rows)
abl_df.to_csv("ablation_results.csv", index=False)

thr = abl_df[abl_df.job_id.str.startswith("threshold_mult_")].copy()
thr["mult"] = thr.job_id.str.replace("threshold_mult_", "").astype(float)
thr = thr.sort_values("mult")
thr["revert_rate"] = 1 - thr.accept_rate

c_sweep, c_ent, c_dis = focal_palette(["sweep","entropy_only","disagreement_only"], focal="sweep", focal_color="#1a6fba")

ent = abl_df[abl_df.job_id == "criterion_entropy_only"].iloc[0]
dis = abl_df[abl_df.job_id == "criterion_disagreement_only"].iloc[0]
no_adapt_acc = abl_df[abl_df.job_id == "ref_resnet20_no_adapt"].mean_acc.iloc[0]
tent_acc = abl_df[abl_df.job_id == "ref_resnet20_tent"].mean_acc.iloc[0]

snap = abl_df[abl_df.job_id.str.startswith("snap_every_") | (abl_df.job_id == "threshold_mult_1.0")].copy()
snap["cadence"] = snap.job_id.map({"snap_every_25": 25, "threshold_mult_1.0": 50, "snap_every_100": 100})
snap = snap.sort_values("cadence")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
axA, axB = axes

axA.plot(thr.revert_rate, thr.mean_acc, "-o", color=c_sweep, lw=1.8, ms=6, zorder=3)
offsets = {0.25: (0, -14), 0.5: (0, -14), 1.0: (-14, 9), 2.0: (0, -14), 4.0: (0, 9)}
for _, row in thr.iterrows():
    axA.annotate(f"×{row.mult:g}", (row.revert_rate, row.mean_acc), textcoords="offset points",
                 xytext=offsets[row.mult], fontsize=6.5, ha="center", color=c_sweep)

axA.scatter([1 - ent.accept_rate], [ent.mean_acc], color=c_ent, marker="^", s=70, zorder=4)
axA.annotate("Entropy only", (1 - ent.accept_rate, ent.mean_acc), textcoords="offset points",
             xytext=(-8, -16), fontsize=7, color=c_ent, ha="right")
axA.scatter([1 - dis.accept_rate], [dis.mean_acc], color=c_dis, marker="s", s=70, zorder=4)
axA.annotate("Disagreement only", (1 - dis.accept_rate, dis.mean_acc), textcoords="offset points",
             xytext=(10, 14), fontsize=7, color=c_dis, ha="left")

axA.axhline(no_adapt_acc, color=META_GREY, ls="--", lw=1)
axA.axhline(tent_acc, color=META_GREY, ls=":", lw=1)
axA.text(0.98, no_adapt_acc - 0.025, "No adaptation", fontsize=7, color=META_GREY, va="top", ha="right",
          transform=axA.get_yaxis_transform())
axA.text(0.98, tent_acc + 0.02, "Ungated TENT", fontsize=7, color=META_GREY, va="bottom", ha="right",
          transform=axA.get_yaxis_transform())

axA.set_xlabel("Revert rate (1 − accept rate)")
axA.set_ylabel("Mean accuracy (60-batch reduced stream)")
axA.set_title("Accuracy-vs-revert-rate frontier: gate tolerance ×0.25–×4, single seed")
axA.set_ylim(0.32, 0.72)
axA.margins(x=0.15)
set_frame(axA, "open")

axB.plot(snap.cadence, snap.mean_acc, "-o", color=c_sweep, lw=1.8, ms=7)
axB.set_ylim(snap.mean_acc.min() - 0.03, snap.mean_acc.max() + 0.03)
axB.set_xticks([25, 50, 100])
axB.set_xlabel("Rollback-check cadence (batches)")
axB.set_ylabel("Mean accuracy")
axB.set_title("Snapshot cadence: no measurable\neffect at this reduced scope")
axB.annotate("Rollback trigger never fired\nwithin 60 batches at any cadence\n(disclosed null sub-result)",
             xy=(50, snap.mean_acc.iloc[len(snap)//2]), xytext=(0.5, 0.22), textcoords="axes fraction",
             fontsize=6.5, ha="center", color=META_GREY,
             arrowprops=dict(arrowstyle="-", color=META_GREY, lw=0.6))
set_frame(axB, "open")

fig.suptitle("Ablation / sensitivity — reduced scope (n=1 seed, 60-batch stream, ResNet-20 only; Addendum 6)",
             fontsize=10, x=0.02, ha="left", y=0.995)
fig.subplots_adjust(wspace=0.4)
fig.savefig("fig_ablation.png", dpi=200, bbox_inches="tight")

r = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible()]
overlaps = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(texts) for b, bb in texts[i+1:] if ba.overlaps(bb)]
out_of_bounds = [t.get_text() for t, bt in texts if not fig.bbox.contains(bt.x0, bt.y0) or not fig.bbox.contains(bt.x1, bt.y1)]
print("overlaps:", overlaps)
print("out of bounds:", out_of_bounds)

axB.set_yticks([0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68])
fig.savefig("fig_ablation.png", dpi=200, bbox_inches="tight")
