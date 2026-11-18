"""Fault-distance figure of the paper: for each check family (panel) and code distance d, the code distance (open
marker; the S proxy reaches it) and the certified real-T fault distance (filled marker); the vertical gap is the
distance lost to real T gates. Ranges that are certified only from below are dotted. Style and colours follow
the co-author's figures (Helvetica-like sans serif, Okabe-Ito colours). Writes paper/fig_fd.pdf."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import font_manager
SANS = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "Liberation Sans"  # same metrics
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [SANS, "Helvetica", "DejaVu Sans"],
                     "mathtext.fontset": "custom", "mathtext.rm": SANS, "mathtext.it": SANS + ":italic",
                     "font.size": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42})
INK, GRAY, GRID = "#404040", "#9e9e9e", "#ececec"
TOLBLUE, HARM = "#0072b3", "#d55e00"

panels = [("GHZ fold", {3: 2, 5: 3, 7: 4, 9: 4, 11: 6, 13: 7}, {}),
          ("cat,\nnot verified", {5: 3, 7: 4}, {}),
          ("cat,\nverified", {3: 3, 5: 5, 7: 7, 9: 9}, {}),    # delta=7: with rounds (exact search) and without (late_ilp); 9: without
          ("NN cat,\ngrid", {3: 3, 5: 5, 7: 7, 9: 9}, {})]

DODGE = 0.45                            # horizontal offset of the GHZ-fold and RP^2 markers at delta = 3, 5
RP2 = {3: 2, 5: 3}                      # RP^2 surface-code cultivation of Chen et al. (contraction-tree check), panel 0
BOUND = {0: (11, 13)}                   # panel 0 at delta >= 11: lowest proven upper bound over the binary trees (hook bound);
                                        # at delta = 9 the lowest bound, 4, is certified by the late ILP without rounds (fold_nostab.py)
fig, axes = plt.subplots(1, 4, figsize=(3.375, 1.85), sharey=False, gridspec_kw={"width_ratios": [2.2, 1, 1.3, 1.3]})
for k, (ax, (title, real, upper)) in enumerate(zip(axes, panels)):
    for d in (3, 5, 7, 9, 11, 13):
        if d not in real:
            continue
        r = real[d]
        x = d - DODGE if k == 0 and d in RP2 else d          # side by side with the RP^2 markers
        if r < d:
            ax.plot([x, x], [r, d], color=HARM if d not in upper else TOLBLUE, lw=1.0, zorder=1,
                    ls=(0, (1, 1.2)) if d in upper else "-")
        ax.plot(x, d, "o", ms=4.2, mfc="white", mec=GRAY, mew=0.9, zorder=2)
        if d in BOUND.get(k, ()):
            ax.plot(x, r, "v", ms=4.6, color=TOLBLUE, mec="white", mew=0.5, zorder=3)
        else:
            ax.plot(x, r, "o", ms=4.2, color=TOLBLUE, mec="white", mew=0.5, zorder=3)
    if k == 0:
        for d, r in RP2.items():
            x = d + DODGE
            ax.plot([x, x], [r, d], color=HARM, lw=1.0, zorder=1)
            ax.plot(x, d, "s", ms=3.6, mfc="white", mec=GRAY, mew=0.9, zorder=2)
            ax.plot(x, r, "s", ms=3.6, color=TOLBLUE, mec="white", mew=0.5, zorder=3)
    ax.set_title(title, fontsize=6.5, color=INK, pad=2, linespacing=1.0)
    ax.text(-0.02, 1.30, "(" + "abcd"[k] + ")", transform=ax.transAxes, fontsize=7, fontweight="bold",
            va="bottom", ha="left")
    ax.set_xlim(2, 14 if k == 0 else 10 if k >= 2 else 8)
    ax.set_xticks([3, 5, 7, 9, 11, 13] if k == 0 else [3, 5, 7, 9] if k >= 2 else [3, 5, 7])
    ax.set_ylim(1.3, 13.7)
    ax.set_yticks([2, 4, 6, 8, 10, 12])
    if k:
        ax.set_yticklabels([])
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=6.5)
axes[0].set_ylabel("fault distance", color=INK, fontsize=7)
fig.text(0.5475, 0.095, r"code distance $\delta$", ha="center", va="bottom", color=INK, fontsize=7)
h_code, = axes[0].plot([], [], "o", ms=4.2, mfc="white", mec=GRAY, mew=0.9)
h_real, = axes[0].plot([], [], "o", ms=4.2, color=TOLBLUE, mec="white", mew=0.5)
h_bound, = axes[0].plot([], [], "v", ms=4.6, color=TOLBLUE, mec="white", mew=0.5)
h_rp2, = axes[0].plot([], [], "s", ms=3.6, color=TOLBLUE, mec="white", mew=0.5)
fig.legend([h_code, h_real, h_bound, h_rp2], ["code distance", "real T", "upper bound", "RP$^2$"], loc="lower center",
           bbox_to_anchor=(0.5475, -0.035), ncol=4, frameon=False, fontsize=5.6, handletextpad=0.05, columnspacing=0.5)
fig.subplots_adjust(left=0.1, right=0.995, top=0.76, bottom=0.27, wspace=0.18)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "fig_fd.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=300)
print("wrote", out)
