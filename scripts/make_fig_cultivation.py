"""Writes paper/fig_cult_small.tex and paper/fig_cult_gsj.tex (TikZ, drawn at final size, 8 pt sans-serif,
CMYK Okabe-Ito colours) from results/cultivation/t9_circuit.json and results/cultivation/gsj_d3.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "cultivation")
PAPER = os.path.join(HERE, "..", "paper")

COLOURS = r"""\definecolor{tolblue}{cmyk}{1,0.36,0,0.30}      % T states, absorbed faults
\definecolor{skyblue}{cmyk}{0.63,0.23,0,0.09}   % code, trivial faults
\definecolor{harmver}{cmyk}{0,0.56,1,0.165}     % faulty states, harmful faults
\definecolor{rejgray}{cmyk}{0,0,0,0.62}         % rejected
\definecolor{inkgray}{cmyk}{0,0,0,0.75}         % annotation
\definecolor{pauX}{cmyk}{0,0.31,1,0.10}         % X
\definecolor{pauY}{cmyk}{1,0,0.27,0.38}         % Y
\definecolor{pauZ}{cmyk}{0,0.41,0.18,0.20}      % Z
"""
STYLE = r"""  font=\fontsize{8}{9.6}\fontfamily{phv}\selectfont\mathversion{figsans},
  line width=0.5pt,
  >={Stealth[length=1.6mm,width=1.2mm]},
  letter/.style={font=\fontsize{8}{9.6}\fontfamily{phv}\fontseries{b}\selectfont, anchor=north west, inner sep=0pt},
  small/.style={font=\fontsize{7}{8.4}\fontfamily{phv}\selectfont\mathversion{figsans}},
"""


def fig_small():
    r = json.load(open(os.path.join(RES, "t9_circuit.json")))
    plan = r["plan"]
    by = r["verification"]["single_fault_classes_by_location"]
    L = []
    L.append("% Figure 1 of cultivation.tex, written by scripts/make_fig_cultivation.py. Requires tikz (arrows.meta, calc,")
    L.append("% decorations.pathreplacing) and the math version figsans declared in the preamble.")
    L.append(COLOURS)
    L.append(r"\begin{tikzpicture}[")
    L.append(STYLE.rstrip("\n"))
    L.append(r"  pbox/.style={draw=none, fill=white, inner sep=0.6pt, minimum size=2.2mm, font=\fontsize{7}{8}\fontfamily{phv}\fontseries{b}\selectfont},")
    L.append(r"  brace/.style={decorate, decoration={brace, amplitude=2.2pt}},")
    L.append("]")
    # ---------------------------------------------------------------- (a)
    L.append(r"% ---------------------------------------------------------------- (a) a check seen on the equator")
    L.append(r"\node[letter] at (0,4.05) {(a)};")
    L.append(r"\coordinate (C) at (2.05,2.05);")
    L.append(r"\draw[black!25, line width=0.4pt] ($(C)+(-1.75,0)$) -- ($(C)+(1.75,0)$) node[right, inkgray, inner sep=1pt] {$x$};")
    L.append(r"\draw[black!25, line width=0.4pt] ($(C)+(0,-1.75)$) -- ($(C)+(0,1.75)$) node[above, inkgray, inner sep=1pt] {$y$};")
    L.append(r"\draw[inkgray] (C) circle (1.35);")
    L.append(r"\draw[tolblue!40, line width=1.8pt, line cap=round] ($(C)+(225:1.55)$) -- ($(C)+(45:1.55)$);")
    L.append(r"\node[tolblue, anchor=west, inner sep=1pt, align=left] at ($(C)+(45:1.6)+(0.02,-0.12)$) {check\\[-1pt]$TXT^\dagger$};")
    # reflections
    L.append(r"\draw[<->, harmver!80, dashed, line width=0.4pt, shorten <=2pt, shorten >=2pt] ($(C)+(45:1.35)$) -- ($(C)+(-45:1.35)$);")
    L.append(r"\draw[<->, harmver!80, dashed, line width=0.4pt, shorten <=2pt, shorten >=2pt] ($(C)+(45:1.35)$) -- ($(C)+(135:1.35)$);")
    # drop lines to the axis
    L.append(r"\draw[harmver, dotted, line width=0.7pt] ($(C)+(-45:1.35)$) -- (C);")
    L.append(r"\draw[harmver, dotted, line width=0.7pt] ($(C)+(135:1.35)$) -- (C);")
    L.append(r"\draw[harmver, line width=0.35pt] ($(C)+(-45:0.13)$) -- ++(45:0.13) -- ($(C)+(45:0.13)$);")
    L.append(r"\node[harmver, anchor=north, inner sep=1.5pt] at ($(C)+(0,-0.12)$) {$\tfrac12$};")
    L.append(r"\fill[tolblue] ($(C)+(45:1.35)$) circle (1.6pt);")
    L.append(r"\draw[rejgray, fill=white] ($(C)+(225:1.35)$) circle (1.6pt);")
    L.append(r"\fill[harmver] ($(C)+(-45:1.35)$) circle (1.6pt);")
    L.append(r"\fill[harmver] ($(C)+(135:1.35)$) circle (1.6pt);")
    L.append(r"\node[tolblue, anchor=south east, inner sep=2pt] at ($(C)+(45:1.35)$) {$|T\rangle$};")
    L.append(r"\node[rejgray, anchor=north east, inner sep=1.5pt, align=right] at ($(C)+(225:1.35)$) {$Z|T\rangle$\\[-1pt]rejected};")
    L.append(r"\node[harmver, anchor=north west, inner sep=2pt] at ($(C)+(-45:1.35)$) {$X|T\rangle$};")
    L.append(r"\node[harmver, anchor=south east, inner sep=2pt] at ($(C)+(135:1.35)$) {$Y|T\rangle$};")
    # ---------------------------------------------------------------- (b)
    L.append(r"% ---------------------------------------------------------------- (b) the nine-T circuit")
    x0 = 4.55
    L.append(r"\node[letter] at (%.2f,4.05) {(b)};" % x0)
    ys = [3.55 - 0.3 * i for i in range(9)]
    ya, yf = 0.78, 0.48
    xw0, xT, xm0, dx = x0 + 0.62, x0 + 0.95, x0 + 1.55, 0.6
    # the flagged first measurement shows its gates apart, so it gets extra room
    xm = [xm0 + 0.15] + [xm0 + 0.45 + dx * k for k in range(1, 8)]
    xdec = xm[-1] + 0.42
    L.append(r"\draw[brace] (%.2f,%.2f) -- (%.2f,%.2f);" % (x0 + 0.52, ys[-1] - 0.12, x0 + 0.52, ys[0] + 0.12))
    L.append(r"\node[anchor=east, inner sep=1pt] at (%.2f,%.2f) {$|{+}\rangle$};" % (x0 + 0.47, (ys[0] + ys[-1]) / 2))
    for y in ys:
        L.append(r"\draw (%.2f,%.2f) -- (%.2f,%.2f);" % (xw0, y, xdec, y))
    L.append(r"\draw (%.2f,%.2f) -- (%.2f,%.2f);" % (xm[0] - 0.3, ya, xm[-1] + 0.25, ya))
    L.append(r"\node[anchor=east, inner sep=1pt, small] at (%.2f,%.2f) {ancilla};" % (xm[0] - 0.33, ya))
    L.append(r"\draw (%.2f,%.2f) -- (%.2f,%.2f);" % (xm[0] - 0.3, yf, xm[0] + 0.3, yf))
    L.append(r"\node[anchor=east, inner sep=1pt, small] at (%.2f,%.2f) {flag};" % (xm[0] - 0.33, yf))
    for y in ys:
        L.append(r"\node[draw, fill=white, inner sep=0pt, minimum width=2.6mm, minimum height=2.4mm, small] at (%.2f,%.2f) {$T$};" % (xT, y))
    for k, (g, order, flag) in enumerate(plan):
        sup = [q for q in range(9) if g[1 + q] != "I"]
        # gate positions: one vertical line per measurement, except that the flagged measurement shows its first
        # and last gate apart, with the flag CNOTs between them
        pos = {q: xm[k] for q in sup}
        if flag:
            pos[order[0]], pos[order[-1]] = xm[k] - 0.24, xm[k] + 0.24
        L.append(r"\node[inkgray, small] at (%.2f,%.2f) {%d};" % (xm[k], ys[0] + 0.33, k + 1))
        for x in sorted(set(pos.values())):
            top = min(q for q in sup if pos[q] == x)
            L.append(r"\draw[inkgray] (%.2f,%.2f) -- (%.2f,%.2f);" % (x, ys[top], x, ya))
            L.append(r"\fill (%.2f,%.2f) circle (1.1pt);" % (x, ya))
        for q in sup:
            p = g[1 + q]
            L.append(r"\node[pbox, text=pau%s] at (%.2f,%.2f) {%s};" % (p, pos[q], ys[q], p))
            L.append(r"\node[anchor=south east, inner sep=0pt, inkgray, font=\fontsize{5}{6}\fontfamily{phv}\selectfont] at (%.2f,%.2f) {%d};" %
                     (pos[q] - 0.1, ys[q] + 0.01, order.index(q) + 1))
        L.append(r"\node[small, inkgray] at (%.2f,%.2f) {$%s$};" % (xm[k] + (0.38 if flag else 0.2), ya - 0.17, "+" if g[0] == "+" else "-"))
        if flag:
            for s in (-0.12, 0.12):
                L.append(r"\draw (%.2f,%.2f) -- (%.2f,%.2f);" % (xm[k] + s, ya, xm[k] + s, yf - 0.06))
                L.append(r"\fill (%.2f,%.2f) circle (0.9pt);" % (xm[k] + s, ya))
                L.append(r"\draw[fill=white] (%.2f,%.2f) circle (0.75mm);" % (xm[k] + s, yf))
                L.append(r"\draw (%.2f,%.2f) -- (%.2f,%.2f) (%.2f,%.2f) -- (%.2f,%.2f);" %
                         (xm[k] + s - 0.075, yf, xm[k] + s + 0.075, yf, xm[k] + s, yf - 0.075, xm[k] + s, yf + 0.075))
    L.append(r"\draw[fill=skyblue!18, draw=skyblue, rounded corners=1pt] (%.2f,%.2f) rectangle (%.2f,%.2f);" %
             (xdec, ys[-1] - 0.17, xdec + 0.42, ys[0] + 0.17))
    L.append(r"\node[rotate=90, small] at (%.2f,%.2f) {ideal decoder};" % (xdec + 0.21, (ys[0] + ys[-1]) / 2))
    L.append(r"\draw[->] (%.2f,%.2f) -- (%.2f,%.2f) node[right, inner sep=1pt] {$|T\rangle$};" %
             (xdec + 0.42, (ys[0] + ys[-1]) / 2, xdec + 0.72, (ys[0] + ys[-1]) / 2))
    # ---------------------------------------------------------------- (c)
    L.append(r"% ---------------------------------------------------------------- (c) single faults")
    xc = 13.15
    L.append(r"\node[letter] at (%.2f,4.05) {(c)};" % xc)
    cats = [("detected", "black!22", "black"), ("rejected", "rejgray", "white"),
            ("trivial", "skyblue!55", "black"), ("absorbed", "tolblue", "white")]
    rows = [("preparations", "preparation"), ("T gates", "T gate"), ("two-qubit gates", "two-qubit gate"),
            ("measurements", "measurement")]
    Lbar, h = 4.2, 0.27
    for n, (label, key) in enumerate(rows):
        yb = 3.25 - 0.66 * n
        d = by.get(key, {})
        tot = sum(d.values())
        L.append(r"\node[anchor=south west, inner sep=0pt] at (%.2f,%.2f) {%s (%d)};" % (xc + 0.05, yb + h / 2 + 0.05, label, tot))
        x = xc + 0.05
        for c, col, tcol in cats:
            v = d.get(c, 0)
            if not v:
                continue
            w = Lbar * v / tot
            L.append(r"\fill[%s] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (col, x, yb - h / 2, x + w, yb + h / 2))
            if w > 0.32:
                L.append(r"\node[small, text=%s] at (%.3f,%.3f) {%d};" % (tcol, x + w / 2, yb, v))
            x += w
    for n, (c, col, tcol) in enumerate(cats):
        xl = xc + 0.05 + (n % 2) * 2.1
        yl = 0.52 - (n // 2) * 0.32
        L.append(r"\fill[%s] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (col, xl, yl - 0.1, xl + 0.28, yl + 0.1))
        L.append(r"\node[anchor=west, inner sep=1.5pt] at (%.2f,%.2f) {%s};" % (xl + 0.3, yl, c))
    L.append(r"\end{tikzpicture}")
    open(os.path.join(PAPER, "fig_cult_small.tex"), "w").write("\n".join(L) + "\n")


def fig_gsj():
    r = json.load(open(os.path.join(RES, "gsj_d3.json")))
    table = r["published_vector_simulations"]
    fit = r["fit_excess_over_p2"]["0.005"]
    c2 = r["T_uniform_depolarizing"]["c2"]
    L = []
    L.append("% Figure 2 of cultivation.tex, written by scripts/make_fig_cultivation.py. Requires tikz and pgfplots.")
    L.append(COLOURS)
    L.append(r"\begin{tikzpicture}[")
    L.append(STYLE.rstrip("\n"))
    L.append("]")
    # (a) the colour-code triangle, scaled into the left half; (b) the plot beside it. All explanations are in
    # the caption, so the panels carry no sentences.
    L.append(r"\node[letter] at (0,3.05) {(a)};")
    L.append(r"\begin{scope}[shift={(0.55,-3.55)}, scale=0.78]")
    # colour-code triangle; qubit labels as in the circuit (coordinates of the published chunks)
    P = {10: (0.35, 4.75), 12: (3.95, 4.75), 13: (2.15, 7.87), 4: (2.15, 4.75), 6: (1.25, 6.31), 5: (3.05, 6.31),
         3: (2.15, 5.79)}
    faces = [((10, 4, 3, 6), "pauX!14"), ((4, 12, 5, 3), "pauY!14"), ((6, 3, 5, 13), "pauZ!14")]
    for f, col in faces:
        L.append(r"\fill[%s] %s -- cycle;" % (col, " -- ".join("(%.2f,%.2f)" % P[q] for q in f)))
    L.append(r"\draw[inkgray] (%.2f,%.2f) -- (%.2f,%.2f) -- (%.2f,%.2f) -- cycle;" % (P[10] + P[12] + P[13]))
    L.append(r"\draw[inkgray] (%.2f,%.2f) -- (%.2f,%.2f) -- (%.2f,%.2f) (%.2f,%.2f) -- (%.2f,%.2f);" %
             (P[4] + P[3] + P[6] + P[3] + P[5]))
    L.append(r"\draw[harmver, line width=1.6pt, line cap=round, opacity=0.55] (%.2f,%.2f) -- (%.2f,%.2f);" % (P[10] + P[13]))
    for q, (x, y) in P.items():
        L.append(r"\fill (%.2f,%.2f) circle (1.5pt);" % (x, y))
    # cat qubit 14 and the hook 6 -> 14 -> 10
    L.append(r"\draw[fill=white] (-0.07,5.28) rectangle (0.13,5.48);")
    L.append(r"\draw[->, harmver, dashed, line width=0.5pt] (%.2f,%.2f) to[out=200, in=60] (0.13,5.47);" % (P[6][0] - 0.06, P[6][1] - 0.04))
    L.append(r"\draw[->, harmver, dashed, line width=0.5pt] (0.03,5.27) to[out=-80, in=150] (%.2f,%.2f);" % (P[10][0] - 0.04, P[10][1] + 0.06))
    for q, lab, anc in ((10, "1", "north"), (6, "2", "south east"), (13, "3", "east")):
        L.append(r"\node[anchor=%s, inner sep=1.5pt, small] at (%.2f,%.2f) {%s};" % (anc, P[q][0], P[q][1], lab))
    # the two faults (stars) and the logical operator they complete
    for q, dx, dy in ((6, 0.30, 0.10), (13, 0.30, 0.0)):
        L.append(r"\node[star, star points=6, star point ratio=2.4, fill=harmver, draw=white, line width=0.3pt, "
                 r"inner sep=0pt, minimum size=2.8mm] at (%.2f,%.2f) {};" % (P[q][0] + dx, P[q][1] + dy))
    L.append(r"\node[harmver, anchor=south east, inner sep=1pt] at (%.2f,%.2f) {$\bar X$};" % (P[10][0] + 0.62 * (P[13][0] - P[10][0]) - 0.1, P[10][1] + 0.62 * (P[13][1] - P[10][1])))
    L.append(r"\end{scope}")
    # ---------------------------------------------------------------- (b)
    L.append(r"\node[letter] at (3.85,3.05) {(b)};")
    L.append(r"\begin{axis}[at={(4.95cm,0.62cm)}, anchor=south west, width=3.35cm, height=2.2cm, scale only axis,")
    L.append(r"  xmin=-0.4, xmax=10.5, ymin=0, ymax=3.0, xtick={0,5,10}, ytick={0,1,2,3},")
    L.append(r"  tick label style={font=\fontsize{8}{9.6}\fontfamily{phv}\selectfont}, label style={font=\fontsize{8}{9.6}\fontfamily{phv}\selectfont\mathversion{figsans}},")
    L.append(r"  xlabel={$p$ ($10^{-3}$)}, ylabel={$(\epsilon_T-\epsilon_S)/p^2$},")
    L.append(r"  axis line style={line width=0.5pt}, tick style={line width=0.5pt}, xlabel shift=-3pt, ylabel shift=-4pt,")
    L.append(r"  clip=false]")
    L.append(r"\addplot[inkgray, dashed, line width=0.5pt, domain=0:5.6] {%.4f + %.6f*x};" % (fit["a"], fit["b"] * 1e-3))
    pts = " ".join("(%.3f,%.4f) +- (0,%.4f)" % (t["p"] * 1e3, t["excess_over_p2"], t["sd_excess"]) for t in table)
    L.append(r"\addplot[only marks, mark=*, mark size=1.3pt, black, error bars/.cd, y dir=both, y explicit, error bar style={line width=0.5pt}] coordinates {%s};" % pts)
    L.append(r"\addplot[only marks, mark=diamond*, mark size=2.4pt, harmver] coordinates {(0,%.4f)};" % c2)
    L.append(r"\end{axis}")
    L.append(r"\end{tikzpicture}")
    open(os.path.join(PAPER, "fig_cult_gsj.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    fig_small()
    fig_gsj()
