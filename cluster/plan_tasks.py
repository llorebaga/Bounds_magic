"""Split a structure file for fast_enum.jl into array tasks of roughly equal work.
Work of a structure = 2^L * 2^m signed codes (L = size of its lift basis), at RATE codes per core-second.
Consecutive small structures are grouped ("L first last", 1-based lines); a structure larger than the
target is split over its lift range ("S line k K", run with SHARD=k NSHARDS=K).
Usage: python plan_tasks.py structs.txt m core_hours_per_task tasks_out
Prints: ntasks total_core_hours max_task_core_hours"""
import sys, math

RATE = 7.5e6
fname, m, target, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
work = []
for line in open(fname):
    if line.strip():
        L = len(line.split("|")[3].split())
        work.append((2 ** L) * (2 ** m) / RATE / 3600)

tasks, cur, cur_w, mx = [], None, 0.0, 0.0
def flush():
    global cur, cur_w, mx
    if cur:
        tasks.append(f"L {cur[0]} {cur[1]}"); mx = max(mx, cur_w)
    cur, cur_w = None, 0.0
for i, w in enumerate(work, 1):
    if w > target:
        flush()
        K = math.ceil(w / target)
        tasks += [f"S {i} {k} {K}" for k in range(K)]; mx = max(mx, w / K)
        continue
    if cur and cur_w + w > target: flush()
    cur = (cur[0], i) if cur else (i, i); cur_w += w
flush()
open(out, "w").write("\n".join(tasks) + ("\n" if tasks else ""))
print(len(tasks), f"{sum(work):.1f}", f"{mx:.2f}")
