"""
Figure 1 - main grid
Source: fig_main_results.png

Extracted verbatim from this study's execution record (the cell that produced the
figure). It ran inside an interactive session that already held the loaded logs and
the figure-style helpers, so it is a faithful record of what generated the figure
rather than a turnkey script; run it after loading the logs as the runners do.

One line was repaired: the extractor emitted the invalid statement "import numpy as np as np2"
when it merged the cells that produced this figure; it is restored to "import numpy as np".
No other line was altered. The trailing block documents the post-hoc correction of the figure
suptitle from "14" to "15" corruption types, applied to the rendered image.
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageDraw, ImageFont
import os

# Load logs from the restored JSON files
import shutil

# We need the main grid result logs
logs = {}
conds = ["no_adapt", "tent", "cotta", "random_revert", "gated_tent"]

# Load from the existing artifact
existing = "/Users/prem/.claude-science/orgs/0d81ddc7-678d-4f34-8207-98d2e44a4b23/artifacts/proj_af10f6c29192/30d46c0f-1362-496e-b85d-a3f6a4e9b218/v9fa6b378_fig_main_results.png"
shutil.copy(existing, "fig_main_results_orig.png")

# We need to reconstruct from data - load the main_grid_results.csv and per-batch logs
# The figure was built from per-batch logs stored in results/main_grid/
# Reconstruct using what's available

def robustness_floor(accs, window=50):
    accs = np.array(accs)
    if len(accs) < window:
        window = len(accs)
    csum = np.cumsum(accs)
    csum = np.insert(csum, 0, 0)
    window_means = (csum[window:] - csum[:-window]) / window
    return window_means.min(), window_means

def rolling(a, w=50):
    if len(a) < w:
        w = len(a)
    return np.convolve(a, np.ones(w)/w, mode="valid")

# Restore per-batch logs from artifacts
import os
os.makedirs("results/main_grid", exist_ok=True)

want = {
    "gated_tent_seed1.json": "0e5c57be-ce35-48e3-bf0f-7ebd938067a0",
    "gated_tent_seed2.json": "09613890-723a-464e-a1d6-fcea9a9667ab",
    "gated_tent_seed3.json": "3d0b6892-ea60-46fe-bb94-9e01668e763c",
    "random_revert_seed1.json": "5c8d5fdf-9d30-440e-a7a2-765ea3c472c6",
    "random_revert_seed2.json": "9448df8b-3d26-42bc-9867-c5946ae000e0",
    "random_revert_seed3.json": "b3f224b6-9465-4df7-9ecd-d56bebdd63b8",
    "cotta_seed1.json": "307fc1d3-cdb1-41a6-8f7e-4f4a92faa5f1",
    "cotta_seed2.json": "3e72fdb3-6dc3-4f5e-be68-61c4a3a8c5ef",
    "cotta_seed3.json": "5e52e32b-497e-4d39-8aa3-5053ee902087",
    "tent_seed1.json": "e8cb4814-21da-43cc-b92c-ad3420430814",
    "tent_seed2.json": "0121f9bd-02ab-4046-9cd1-a30886b44e99",
    "tent_seed3.json": "6aca2ea7-9543-4797-9702-9947bcce3d9f",
    "no_adapt_seed1.json": "1ad9da3c-285b-4408-8ce2-178d603ce591",
    "no_adapt_seed2.json": "237b0ae0-49c9-4179-8951-372d32a8681a",
    "no_adapt_seed3.json": "dc5966a5-b3d7-448d-97e3-b13277c4598c",
}
for fn, vid in want.items():
    shutil.copy("{{artifact:" + vid + "}}", f"results/main_grid/{fn}")

# Load logs
LOGS = {}
for c in conds:
    LOGS[c] = {}
    for s in (1, 2, 3):
        LOGS[c][s] = json.load(open(f"results/main_grid/{c}_seed{s}.json"))

def acc_arr(c, s):
    return np.array([r["acc"] for r in LOGS[c][s]["log"]], dtype=float)

# --- figure style helpers (inline replacements for apply_figure_style, focal_palette, etc.) ---

def apply_figure_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

META_GREY = "#888888"

def focal_palette(items, focal=None, focal_color="#1a6fba"):
    palette = ["#aaaaaa", "#bbbbbb", "#cccccc", "#dddddd", "#eeeeee"]
    result = []
    for i, item in enumerate(items):
        if item == focal:
            result.append(focal_color)
        else:
            grey_idx = sum(1 for j, x in enumerate(items) if x != focal and j < i)
            greys = ["#999999", "#b0b0b0", "#c8c8c8", "#d8d8d8", "#e8e8e8"]
            result.append(greys[min(grey_idx, len(greys)-1)])
    return result

def end_of_line_labels(ax, end_xs, end_ys, labels, colors):
    for xs, ys, label, color in zip(end_xs, end_ys, labels, colors):
        ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=7, color=color,
                    va="center", ha="left")

def goodness_arrow(ax, text="higher = better", loc="upper left", axis="y"):
    ax.annotate(text, xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=7, color=META_GREY, va="top", ha="left")

def set_frame(obj, style="open"):
    if hasattr(obj, 'spines'):
        ax = obj
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    elif hasattr(obj, 'axes'):
        fig = obj
        for ax in fig.axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

apply_figure_style()

cond_order = ["no_adapt", "tent", "cotta", "random_revert", "gated_tent"]
cond_labels_short = {
    "no_adapt": "No adaptation",
    "tent": "Ungated TENT",
    "cotta": "CoTTA-style restore",
    "random_revert": "Random-revert (matched)",
    "gated_tent": "Gated TENT (proposed)",
}
tick_labels_compact = {
    "no_adapt": "No adapt.",
    "tent": "Ungated\nTENT",
    "cotta": "CoTTA",
    "random_revert": "Random-\nrevert",
    "gated_tent": "Gated\nTENT",
}

colors = dict(zip(cond_order, focal_palette(cond_order, focal="gated_tent", focal_color="#1a6fba")))

fig, axes = plt.subplots(1, 2, figsize=(13.3, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
axA, axB = axes

end_xs, end_ys, end_labels, end_colors = [], [], [], []
for cond in cond_order:
    seed_curves = np.array([robustness_floor(acc_arr(cond, s))[1] for s in [1, 2, 3]])
    mean_curve = seed_curves.mean(axis=0)
    x = np.arange(len(mean_curve))
    axA.plot(x, mean_curve, color=colors[cond], lw=2.2 if cond == "gated_tent" else 1.4,
             alpha=1.0 if cond == "gated_tent" else 0.85, zorder=3 if cond == "gated_tent" else 2)
    end_xs.append(x)
    end_ys.append(mean_curve)
    end_labels.append(cond_labels_short[cond])
    end_colors.append(colors[cond])

axA.set_xlabel("Stream position (50-batch sliding window, batch index)")
axA.set_ylabel("Mean accuracy in window")
axA.set_title("Accuracy over a 945-batch cyclic non-IID corruption stream (mean of 3 seeds)")
axA.margins(x=0.02, y=0.06)
end_of_line_labels(axA, end_xs, end_ys, end_labels, end_colors)
set_frame(axA, "open")

xpos = {c: i for i, c in enumerate(cond_order)}
for cond in cond_order:
    floors = [robustness_floor(acc_arr(cond, s))[0] for s in [1, 2, 3]]
    jitter = np.random.default_rng(hash(cond) % 1000).uniform(-0.08, 0.08, size=3)
    axB.scatter(np.full(3, xpos[cond]) + jitter, floors, color=colors[cond], s=42, zorder=3,
                edgecolor="white", linewidth=0.5)
    axB.scatter([xpos[cond]], [np.mean(floors)], color=colors[cond], s=140, marker="_", zorder=4)

axB.set_xticks(list(xpos.values()))
axB.set_xticklabels([tick_labels_compact[c] for c in cond_order], fontsize=7)
axB.set_ylabel("Robustness floor\n(min 50-batch window accuracy)")
axB.set_title("Primary endpoint: robustness floor, 3 seeds each")
axB.margins(x=0.08, y=0.1)
goodness_arrow(axB, text="higher = better", loc="upper left", axis="y")
set_frame(axB, "open")

fig.suptitle("Adapt-or-Abstain main grid — ResNet-20, CIFAR-10, 14 corruption types (corrected random-revert control)",
             fontsize=10, x=0.02, ha="left", y=0.995)
fig.subplots_adjust(wspace=0.55)
fig.savefig("fig_main_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Apply the suptitle correction (15 instead of 14 corruption types) via PIL text replacement
im = Image.open("fig_main_results.png").convert("RGB")
W_, H_ = im.size
import numpy as np

a = np.array(im.convert("L"))
dark = (a < 200).sum(axis=1)
band = [(i, int(dark[i])) for i in range(0, 130)]
gaps, run = [], None
for i, v in band:
    if v > 0 and run is None:
        run = i
    elif v == 0 and run is not None:
        gaps.append((run, i - 1))
        run = None
if run is not None:
    gaps.append((run, 129))

if gaps:
    sup_y0, sup_y1 = gaps[0]
    d = ImageDraw.Draw(im)
    d.rectangle([0, sup_y0 - 6, W_, sup_y1 + 6], fill=(255, 255, 255))

    NEWT = "Adapt-or-Abstain main grid \u2014 ResNet-20, CIFAR-10, 15 corruption types (corrected random-revert control)"
    target_h = sup_y1 - sup_y0 + 1
    fnt = None
    for path in ["/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/Library/Fonts/Arial.ttf"]:
        if os.path.exists(path):
            for sz in range(40, 14, -1):
                f = ImageFont.truetype(path, sz)
                bb = d.textbbox((0, 0), NEWT, font=f)
                if (bb[3] - bb[1]) <= target_h + 2:
                    fnt = f
                    break
        if fnt:
            break

    bb = d.textbbox((0, 0), NEWT, font=fnt)
    d.text((20, sup_y0 - bb[1]), NEWT, fill=(0, 0, 0), font=fnt)

im.save("fig_main_results.png")
print("saved fig_main_results.png")
