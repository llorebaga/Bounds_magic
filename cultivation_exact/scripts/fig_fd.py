"""Fault-distance figure of the paper: for each check family (panel) and code distance d, the code distance (open
marker; the S proxy reaches it) and the certified real-T fault distance (filled marker); the vertical gap is the
distance lost to real T gates. Ranges that are certified only from below are dotted. Style and colours follow
the co-author's figures (Helvetica-like sans serif, Okabe-Ito colours). Writes paper/fig_fd.pdf."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "font.size": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42})
INK, GRAY, GRID = "#404040", "#9e9e9e", "#ececec"
TOLBLUE, HARM = "#0072b3", "#d55e00"

panels = [("GHZ fold", {3: 2, 5: 3, 7: 4}, {}),
          ("cat, not verified", {5: 3, 7: 4}, {}),
          ("cat, verified", {3: 3, 5: 5, 7: 5}, {7: 7}),
          ("NN cat, grid", {3: 3, 5: 5, 7: 6}, {7: 7})]

fig, axes = plt.subplots(1, 4, figsize=(3.375, 1.62), sharey=True)
for k, (ax, (title, real, upper)) in enumerate(zip(axes, panels)):
    for d in (3, 5, 7):
        if d not in real:
            continue
        r = real[d]
        if r < d:
            ax.plot([d, d], [r, d], color=HARM if d not in upper else TOLBLUE, lw=1.0, zorder=1,
                    ls=(0, (1, 1.2)) if d in upper else "-")
        ax.plot(d, d, "o", ms=4.2, mfc="white", mec=GRAY, mew=0.9, zorder=2)
        ax.plot(d, r, "o", ms=4.2, color=TOLBLUE, mec="white", mew=0.5, zorder=3)
    ax.set_title(title, fontsize=6.5, color=INK, pad=2)
    ax.text(-0.02, 1.13, "(" + "abcd"[k] + ")", transform=ax.transAxes, fontsize=7, fontweight="bold",
            va="bottom", ha="left")
    ax.set_xlim(2, 8)
    ax.set_xticks([3, 5, 7])
    ax.set_ylim(1.3, 7.7)
    ax.set_yticks([2, 3, 4, 5, 6, 7])
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=6.5)
axes[0].set_ylabel("fault distance", color=INK, fontsize=7)
fig.text(0.33, 0.01, r"code distance $\delta$", ha="center", va="bottom", color=INK, fontsize=7)
h_code, = axes[0].plot([], [], "o", ms=4.2, mfc="white", mec=GRAY, mew=0.9)
h_real, = axes[0].plot([], [], "o", ms=4.2, color=TOLBLUE, mec="white", mew=0.5)
fig.legend([h_code, h_real], ["code distance", "real T"], loc="lower right", bbox_to_anchor=(0.995, -0.01),
           ncol=2, frameon=False, fontsize=6, handletextpad=0.1, columnspacing=0.8)
fig.subplots_adjust(left=0.1, right=0.995, top=0.82, bottom=0.27, wspace=0.18)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "fig_fd.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=300)
print("wrote", out)
