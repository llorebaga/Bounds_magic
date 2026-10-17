"""Run check_t_survivors3.py in parallel on chunks of a large kernel log and merge the class counts.
Usage: python classify_parallel.py logfile n [workers]"""
import sys, os, json, subprocess, collections
from concurrent.futures import ThreadPoolExecutor

fname, n = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
lines = [l for l in open(fname) if "SURVIVOR" in l]
k = max(1, min(workers, len(lines)))
chunks = []
for i in range(k):
    part = f"{fname}.part{i}"
    open(part, "w").writelines(lines[i::k])
    chunks.append(part)

def run(part):
    out = subprocess.run([sys.executable, "check_t_survivors3.py", part, n], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])

total = collections.Counter(); examples = []
with ThreadPoolExecutor(k) as ex:
    for res in ex.map(run, chunks):
        total.update(res)
for part in chunks:
    if os.path.exists(part + ".nonstab.json"):
        examples += json.load(open(part + ".nonstab.json"))
        os.remove(part + ".nonstab.json")
    os.remove(part)
json.dump(examples, open(fname + ".nonstab.json", "w"), indent=1)
print(json.dumps(dict(total)))
