"""Figure 3 of the paper: (a) the d=3 ring check and (b) the d=5 ladder check on a square grid (src/nn_cat.py),
and the appendix figure of the circuit run on hardware, real T against the S proxy under SI1000 noise
(results/google_mc.json). Writes ../../paper/fig_nn.pdf and ../../paper/fig_google.pdf."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nn_cat

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
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
    ax.set_title(title, fontsize=7, color=INK, pad=2, loc="left", fontweight="bold")


# Fig. 3: the two layouts
fig = plt.figure(figsize=(3.375, 1.25))
gs = fig.add_gridspec(1, 2, width_ratios=[6.2, 11.2], wspace=0.08)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
layout(ax_a, nn_cat.RING3, "(a) $d=3$, 17 qubits")
layout(ax_b, nn_cat.ladder(8), "(b) $d=5$, 46 qubits")
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01)
out = os.path.join(HERE, "..", "..", "paper", "fig_nn.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=300)
print("wrote", out)

# appendix figure: the circuit run on hardware, real T against the S proxy
fig, ax_c = plt.subplots(figsize=(3.375, 2.1))
rows = [r for r in json.load(open(os.path.join(HERE, "..", "results", "google_mc.json"))) if r["fig"] == "fig3"]
for magic, color, mfc, lab in (("T", HARM, HARM, "real T"), ("S", MUTED, "white", "S proxy")):
    rr = sorted((r["p"], r["eps"], r["se_eps"]) for r in rows if r["magic"] == magic)
    ax_c.errorbar([x[0] for x in rr], [x[1] for x in rr], yerr=[x[2] for x in rr], color=color, lw=1.0,
                  marker="o", ms=3.5, mfc=mfc, mec=color, mew=0.8, capsize=0, label=lab)
ax_c.axvline(2.3e-3, color=MUTED, lw=0.5, ls=(0, (1, 1.5)), zorder=0)
ax_c.set_xscale("log")
ax_c.set_yscale("log")
ax_c.set_xticks([1e-3, 1.5e-3, 2e-3, 3e-3])
ax_c.set_xticklabels(["1", "1.5", "2", "3"])
ax_c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax_c.set_xlabel(r"SI1000 noise $p$ ($10^{-3}$)", color=INK, labelpad=1)
ax_c.set_ylabel("logical error per accepted state", color=INK, labelpad=1, fontsize=6.5)
ax_c.grid(which="major", color=GRID, lw=0.5, zorder=0)
for s_ in ("top", "right"):
    ax_c.spines[s_].set_visible(False)
ax_c.tick_params(colors=INK, labelsize=6, which="both")
ax_c.legend(loc="lower right", frameon=False, fontsize=6, handletextpad=0.3, borderaxespad=0.1, labelspacing=0.3)
ylo, yhi = ax_c.get_ylim()
ax_c.text(2.3e-3 * 0.95, yhi * 0.6, "experiment", fontsize=5.5, color=MUTED, ha="right", va="top")
fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.18)
out = os.path.join(HERE, "..", "..", "paper", "fig_google.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=300)
print("wrote", out)
