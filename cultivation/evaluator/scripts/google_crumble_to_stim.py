"""Decode the Crumble links of Rosenfeld et al., "Magic state cultivation on a superconducting quantum processor"
(arXiv:2512.13908, HTML version) into stim circuits.

Download the HTML first (https://arxiv.org/html/2512.13908v1) and pass its path. Crumble's URL format writes one
instruction per ';', targets separated by '_', Q(x,y)i for qubit coordinates, DT / OI for DETECTOR /
OBSERVABLE_INCLUDE, and MARK / POLYGON annotations, which are dropped. Circuits are written in the order of the
links: 0 injection example, 1 cultivation example, 2-3 grafting examples, 4 = Fig. 2 cultivation,
5 = Fig. 3 kickback tomography, 6 = Fig. 4 grafting. external/google_circuits/ holds 4 and 5.
Usage: python google_crumble_to_stim.py PAPER.html OUTDIR"""
import html
import os
import re
import sys
import urllib.parse

import stim


def convert(body):
    names = sorted(stim.gate_data().keys(), key=len, reverse=True) + \
        ["DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "QUBIT_COORDS", "SHIFT_COORDS", "MPP"]
    out = []
    for ln in body.split(";"):
        if not ln:
            continue
        m = re.match(r"Q\(([^)]*)\)(\d+)$", ln)
        if m:
            out.append(f"QUBIT_COORDS({m.group(1)}) {m.group(2)}")
            continue
        if ln.startswith(("MARK", "POLYGON", "ERR")):
            continue
        if ln.startswith("DT"):
            ln = "DETECTOR" + ln[2:]
        elif ln.startswith("OI"):
            ln = "OBSERVABLE_INCLUDE" + ln[2:]
        name = next(g for g in names if ln.startswith(g) and (len(ln) == len(g) or ln[len(g)] in "_[("))
        m = re.match(r"(\[[^\]]*\])?(\([^)]*\))?_?(.*)$", ln[len(name):])
        tag, args, tg = m.group(1) or "", m.group(2) or "", m.group(3)
        out.append(f"{name}{tag}{args} {tg.replace('_', ' ')}".strip())
    return stim.Circuit("\n".join(out))


if __name__ == "__main__":
    src, outdir = sys.argv[1], sys.argv[2]
    s = open(src, encoding="utf8").read()
    links = sorted(set(re.findall(r"https?://algassert.com/crumble#circuit=[^\"<> ]*", s)), key=s.find)
    os.makedirs(outdir, exist_ok=True)
    for i, link in enumerate(links):
        c = convert(html.unescape(urllib.parse.unquote(link.split("circuit=", 1)[1])))
        path = os.path.join(outdir, f"crumble_{i}.stim")
        c.to_file(path)
        print(path, c.num_qubits, "qubits", c.num_detectors, "detectors", c.num_ticks, "ticks")
