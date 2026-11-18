"""Figure 4 of the paper: (a) the d=3 ring check and (b) the d=5 ladder check on a square grid (src/nn_cat.py).
Writes ../../paper/fig_nn.pdf."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nn_cat

from matplotlib import font_manager
SANS = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "Liberation Sans"  # same metrics
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [SANS, "Helvetica", "DejaVu Sans"],
                     "mathtext.fontset": "custom", "mathtext.rm": SANS, "mathtext.it": SANS + ":italic",
                     "font.size": 7, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4,
                     "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "pdf.fonttype": 42})
# colours of the co-author's figures: tolblue (T states), skyblue (code), vermillion (faulty), greys
INK, MUTED, GRID, BLUE, ORANGE = "#404040", "#9e9e9e", "#ececec", "#0072b3", "#56b3e8"
HARM = "#d55e00"
HERE = os.path.dirname(os.path.abspath(__file__))


def layout(ax, L, title):
    path, data = L["path"], L["data_sites"]
    xs = [s[0] for s in path + data]
    ys = [s[1] for s in path + data]
    for x in range(min(xs), max(xs) + 1):          # the grid
        ax.plot([x, x], [min(ys), max(ys)], color=GRID, lw=0.5, zorder=0)
    for y in range(min(ys), max(ys) + 1):
        ax.plot([min(xs), max(xs)], [y, y], color=GRID, lw=0.5, zorder=0)
    for a, b in zip(path, path[1:]):              # cat path
        ax.plot([a[0], b[0]], [-a[1], -b[1]], color=BLUE, lw=1.4, zorder=1, solid_capstyle="round")
    for q, s in enumerate(data):                   # couplings
        c = path[L["cat_of"][q]]
        ax.plot([s[0], c[0]], [-s[1], -c[1]], color=INK, lw=0.6, zorder=1)
    anc = set()
    for (i, j), s in zip(L["ver_pairs"], L["ver_sites"]):
        sites = [s] if isinstance(s[0], int) else list(s)
        anc.update(sites)
        chain = [path[i]] + sites + [path[j]]
        for a, b in zip(chain, chain[1:]):
            ax.plot([a[0], b[0]], [-a[1], -b[1]], color=ORANGE, lw=0.6, zorder=1, ls=(0, (2, 1)))
    ax.scatter([s[0] for s in data], [-s[1] for s in data], s=16, color="#1a1a1a", zorder=3, linewidths=0)
    ax.scatter([s[0] for s in path], [-s[1] for s in path], s=16, color=BLUE, zorder=3, edgecolors="white",
               linewidths=0.5)
    r = path[L["root"]]
    ax.scatter([r[0]], [-r[1]], s=30, facecolors="white", edgecolors=BLUE, linewidths=1.0, zorder=4)
    ax.scatter([s[0] for s in anc], [-s[1] for s in anc], s=14, color=ORANGE, zorder=3, edgecolors="white",
               linewidths=0.5, marker="s")
    ax.set_aspect("equal")
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(-max(ys) - 0.6, -min(ys) + 0.6)
    ax.axis("off")
    ax._panel_title = title


# Fig. 3: the two layouts
fig = plt.figure(figsize=(3.375, 1.25))
gs = fig.add_gridspec(1, 2, width_ratios=[6.2, 11.2], wspace=0.08)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
layout(ax_a, nn_cat.RING3, r"(a) $\delta=3$, 17 qubits")
layout(ax_b, nn_cat.ladder(8), r"(b) $\delta=5$, 46 qubits")
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01)
fig.canvas.draw()                                  # apply the equal aspect, then put both titles on one baseline
y = max(ax.get_position().y1 for ax in (ax_a, ax_b)) + 0.03
for ax in (ax_a, ax_b):
    letter, desc = ax._panel_title.split(" ", 1)
    x0 = ax.get_position().x0
    fig.text(x0, y, letter, fontsize=7, fontweight="bold", color="black", va="bottom")
    fig.text(x0 + 0.045, y, desc, fontsize=7, color=INK, va="bottom")
out = os.path.join(HERE, "..", "..", "paper", "fig_nn.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=300)
print("wrote", out)
