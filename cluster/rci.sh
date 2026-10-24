#!/bin/bash
# Runs ON the RCI cluster (in a MobaXterm SSH tab, or remotely through from_pc.sh).
#   bash cluster/rci.sh check            partitions, node sizes, available modules, chosen partitions
#   bash cluster/rci.sh setup            pick Python/Julia(/Gurobi), create the venv, self-test the kernel
#   bash cluster/rci.sh submit RUN       RUN = A1 | A2-export | A2 | B1 | B2 | A3-export | A3-low | A3-top
#   bash cluster/rci.sh status           your queue and the progress of every run
# Re-running `submit RUN` submits only the tasks of that run that have not finished (resume after a
# timeout or failure); it refuses while a previous array of the same run is still queued or running.
# Runs (see CLUSTER_RUNS.md):
#   A1         cubic CS, exclude n = 9: kernel on X-ranks 6 and 7 (~430 core-hours)
#   A2-export  two CCZ states, d = 2, n = 10: export the structures (Python, large memory)
#   A2         ... then the kernel on them (sized after the export)
#   B1         two-level SAT, cubic CCZ with 32 and quartic CCZ with 40 inputs (35 instances x 23 h, 1 core)
#   B2         two-CCZ in-class MIP, n = 11-13, s = 3-6 (needs a Gurobi module with a cluster license)
#   A3-export  cubic CCZ, n = 10: export the X-ranks 0-6 (Python, large memory)
#   A3-low     ... kernel on X-ranks 0-6 (~10% of the top sector)
#   A3-top     ... kernel on X-rank 7: ~86,000 core-hours. Needs `submit A3-top --yes`.
set -eo pipefail
MAGIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MAGIC_DIR
. "$MAGIC_DIR/cluster/env.sh"
cd "$MAGIC_DIR"
mkdir -p out

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- partitions
# Lines "name minutes cpus_per_node mem_MB" for CPU partitions (no gpu in the name).
partitions() {
    sinfo -h -o "%R|%l|%c|%m" | sort -u | awk -F'|' '
    function mins(t,   d, n, a) {
        if (t == "infinite" || t == "UNLIMITED") return 10^9
        d = 0; if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] }
        n = split(t, a, ":")
        if (n == 3) return d*1440 + a[1]*60 + a[2] + (a[3] > 0)
        if (n == 2) return d*1440 + a[1] + (a[2] > 0)
        return d*1440 + a[1]
    }
    tolower($1) !~ /gpu/ { gsub(/\+/, "", $2); gsub(/\+/, "", $3); gsub(/\+/, "", $4); print $1, mins($2), $3, $4 }'
}
# pick_partition MINUTES [MIN_MEM_MB] -> "name cpus mem"
pick_partition() {
    local need=$1 mem=${2:-0}
    if [ -n "$PARTITION" ]; then
        partitions | awk -v p="$PARTITION" '$1 == p { print $1, $3, $4; exit }'; return
    fi
    partitions | awk -v need="$need" -v mem="$mem" -v pref="$PART_PREFER" '
        $2 >= need && $4 >= mem {
            score = (index($1, pref) == 1 ? 0 : 1) * 10^10 + $2
            if (best == "" || score < bs) { best = $1 " " $3 " " $4; bs = score }
        }
        END { print best }'
}
fmt_minutes() { printf "%d-%02d:%02d:00" $(($1 / 1440)) $((($1 % 1440) / 60)) $(($1 % 60)); }

# ---------------------------------------------------------------- bookkeeping
# active_job OUTDIR -> job id if that run still has tasks in the queue
active_job() {
    local id
    [ -f "$1/jobid" ] || return 0
    id=$(cat "$1/jobid")
    squeue -h -j "$id" -o "%i" 2>/dev/null | head -1 | grep -q . && echo "$id"
    true
}
# missing_tasks OUTDIR NTASKS PATTERN -> comma list of task ids without a finished output
missing_tasks() {
    local out=$1 nt=$2 pat=$3 t list=""
    for ((t = 0; t < nt; t++)); do
        [ -s "$out/${pat//@/$t}" ] || list="$list,$t"
    done
    echo "${list#,}"
}
compress() {  # "0,1,2,5,7,8" -> "0-2,5,7-8"
    echo "$1" | tr ',' '\n' | awk 'NR==1{s=e=$1; next} $1==e+1{e=$1; next} {printf "%s%s,", s, (e>s?"-"e:""); s=e=$1} END{if (NR) printf "%s%s\n", s, (e>s?"-"e:"")}'
}
do_sbatch() {  # OUTDIR ARRAYSPEC SBATCH_ARGS... JOBFILE
    local out=$1 spec=$2; shift 2
    local id
    id=$(sbatch --parsable $SBATCH_ACCOUNT --array="$spec%$ARRAY_THROTTLE" --output="$out/slurm-%A_%a.out" "$@")
    echo "${id%%;*}" > "$out/jobid"
    echo "submitted job ${id%%;*} (tasks $spec) -> $out"
}

# ---------------------------------------------------------------- kernel runs
# submit_kernel NAME STRUCTS N D M
submit_kernel() {
    local name=$1 structs=$2 n=$3 d=$4 m=$5 out="$MAGIC_DIR/out/$1"
    [ -s "$structs" ] || { echo "skip $name: $structs missing or empty"; return 0; }
    mkdir -p "$out"
    [ -n "$(active_job "$out")" ] && { echo "skip $name: job $(cat "$out/jobid") still in the queue"; return 0; }
    [ -f "$out/tasks.txt" ] || python cluster/plan_tasks.py "$structs" "$m" "$TASK_CORE_HOURS" "$out/tasks.txt" > "$out/plan.txt"
    local nt total mx
    read -r nt total mx < "$out/plan.txt"
    local miss; miss=$(missing_tasks "$out" "$nt" "task_@.log")
    [ -z "$miss" ] && { echo "$name: all $nt tasks finished"; return 0; }
    # choose partition: time for the largest task on KERNEL_CPUS cores, safety factor 2, plus 30 min
    local cpus=$KERNEL_CPUS mins part pc pm
    for pass in 1 2; do
        mins=$(awk -v h="$mx" -v c="$cpus" 'BEGIN{printf "%d", h / c * 2 * 60 + 30}')
        read -r part pc pm <<< "$(pick_partition "$mins")"
        [ -n "$part" ] || die "no CPU partition allows $(fmt_minutes "$mins"); set PARTITION or a smaller TASK_CORE_HOURS"
        [ "$pc" -lt "$cpus" ] && cpus=$pc
    done
    echo "$name: $nt tasks ($total core-hours in total, at most $mx per task); partition $part, $cpus cores, $(fmt_minutes "$mins")"
    echo "file=$structs n=$n d=$d m=$m partition=$part cpus=$cpus time=$(fmt_minutes "$mins") submitted=$(date -Is)" >> "$out/meta.txt"
    do_sbatch "$out" "$(compress "$miss")" --partition="$part" --time="$(fmt_minutes "$mins")" \
        --cpus-per-task="$cpus" --job-name="mk-$name" \
        --export=ALL,MAGIC_DIR="$MAGIC_DIR",STRUCTS="$MAGIC_DIR/$structs",TASKS="$out/tasks.txt",N="$n",D="$d",M="$m",OUT="$out",GALOIS=1 \
        cluster/jobs/kernel.sbatch
}

# submit_export NAME R N D AMIN AMAX MEM_GB HOURS
submit_export() {
    local name=$1 r=$2 n=$3 d=$4 lo=$5 hi=$6 mem=$7 hours=$8 out="$MAGIC_DIR/out/$1" a miss=""
    mkdir -p "$out"
    [ -n "$(active_job "$out")" ] && { echo "skip $name: job $(cat "$out/jobid") still in the queue"; return 0; }
    for ((a = lo; a <= hi; a++)); do
        grep -q '^exported' "$out/export_a$a.log" 2>/dev/null || miss="$miss,$a"
    done
    miss=${miss#,}
    [ -z "$miss" ] && { echo "$name: all exports finished"; return 0; }
    local part pc pm
    read -r part pc pm <<< "$(pick_partition $((hours * 60)) $((mem * 1024)))"
    [ -n "$part" ] || die "no CPU partition with $hours h and $mem GB per node; set PARTITION"
    echo "$name: export X-ranks $miss on $part ($mem GB, $hours h)"
    do_sbatch "$out" "$(compress "$miss")" --partition="$part" --time="$(fmt_minutes $((hours * 60)))" \
        --mem="${mem}G" --job-name="mx-$name" \
        --export=ALL,MAGIC_DIR="$MAGIC_DIR",R="$r",N="$n",D="$d",OUT="$out" cluster/jobs/export.sbatch
}

# submit_list NAME LISTFILE CPUS MEM_GB JOBFILE
submit_list() {
    local name=$1 list=$2 cpus=$3 mem=$4 job=$5 out="$MAGIC_DIR/out/$1"
    mkdir -p "$out"
    [ -n "$(active_job "$out")" ] && { echo "skip $name: job $(cat "$out/jobid") still in the queue"; return 0; }
    local nt; nt=$(grep -v '^#' "$list" | grep -c .)
    local miss; miss=$(missing_tasks "$out" "$nt" "task_@.log")
    [ -z "$miss" ] && { echo "$name: all $nt instances finished"; return 0; }
    local mins=$((SOLVER_SECONDS / 60 + 60)) part pc pm
    read -r part pc pm <<< "$(pick_partition "$mins" $((mem * 1024)))"
    [ -n "$part" ] || die "no CPU partition allows $(fmt_minutes $mins); lower SOLVER_SECONDS or set PARTITION"
    echo "$name: $nt instances, $SOLVER_SECONDS s each, partition $part"
    do_sbatch "$out" "$(compress "$miss")" --partition="$part" --time="$(fmt_minutes $mins)" \
        --cpus-per-task="$cpus" --mem="${mem}G" --job-name="ms-$name" \
        --export=ALL,MAGIC_DIR="$MAGIC_DIR",LIST="$MAGIC_DIR/$list",OUT="$out",SOLVER_SECONDS="$SOLVER_SECONDS" "$job"
}

# ---------------------------------------------------------------- commands
cmd_check() {
    echo "== CPU partitions (name, time limit in minutes, cores per node, memory per node in MB)"
    partitions | column -t
    echo; echo "== chosen for: 4 h -> $(pick_partition 240) | 24 h -> $(pick_partition 1440) | 3 days -> $(pick_partition 4320)"
    echo "== MaxArraySize: $(scontrol show config 2>/dev/null | awk '/MaxArraySize/{print $3}')"
    echo; echo "== modules"
    for m in Python Julia Gurobi; do
        echo "-- $m:"; module -t avail "$m" 2>&1 | grep -i "^$m/" | tail -5 || true
    done
    echo; echo "== storage"; df -h "$HOME" | tail -1; quota -s 2>/dev/null | tail -2 || true
    echo; echo "== software chosen by setup:"; cat cluster/.software 2>/dev/null || echo "(setup not run yet)"
}

cmd_setup() {
    local py jl gu sw=cluster/.software
    py=$(module -t avail Python 2>&1 | grep -E '^Python/3\.(1[0-9])' | sort -V | tail -1 || true)
    jl=$(module -t avail Julia 2>&1 | grep -E '^Julia/1\.(9|1[0-9])' | sort -V | tail -1 || true)
    gu=$(module -t avail Gurobi 2>&1 | grep -E '^Gurobi/' | sort -V | tail -1 || true)
    : > "$sw"
    if [ -n "$py" ]; then echo "module load $py" >> "$sw"; module load "$py"
    else echo "no Python 3.10+ module found, using $(command -v python3)"; fi
    python3 -c 'import sys; assert sys.version_info >= (3, 9), sys.version' || die "Python >= 3.9 needed"
    [ -d venv ] || python3 -m venv venv
    echo ". \"\$MAGIC_DIR/venv/bin/activate\"" >> "$sw"
    . venv/bin/activate
    pip install -q --upgrade pip
    pip install -q numpy python-sat
    if [ -n "$jl" ]; then echo "module load $jl" >> "$sw"; module load "$jl"
    else
        if [ ! -x julia-1.10.5/bin/julia ]; then
            echo "no Julia module: downloading the official Julia 1.10.5 binary into $MAGIC_DIR/julia-1.10.5"
            curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.5-linux-x86_64.tar.gz | tar xz
        fi
        echo "export PATH=\"\$MAGIC_DIR/julia-1.10.5/bin:\$PATH\"" >> "$sw"
        export PATH="$MAGIC_DIR/julia-1.10.5/bin:$PATH"
    fi
    if [ -n "$gu" ]; then
        echo "GUROBI_MODULE=$gu" >> "$sw"
        module load "$gu"
        local v; v=$(echo "$gu" | sed -E 's|^Gurobi/([0-9]+\.[0-9]+(\.[0-9]+)?).*|\1|')
        python -c 'import gurobipy' 2>/dev/null || pip install -q "gurobipy==$v.*" || echo "WARNING: could not install gurobipy $v"
        python -c 'import gurobipy as g; m = g.Model(); print("gurobi ok", g.gurobi.version())' || echo "WARNING: Gurobi license not usable here (B2 will fail)"
    else
        echo "GUROBI_MODULE=" >> "$sw"; echo "no Gurobi module: run B2 is not available"
    fi
    echo "== $sw:"; cat "$sw"
    echo "== self-test (CCZ, n = 8, d = 3, X-rank 5); expected: signed candidates 536870912 survivors 0"
    (cd scripts && julia -t 4 fast_enum.jl ../results/structs_r3_n8_d3_a5.txt 8 3 5 | tail -1)
    python -c 'import numpy, pysat; print("python ok", numpy.__version__)'
}

cmd_status() {
    echo "== queue"; squeue -u "$USER" -o "%.12i %.14j %.10P %.8T %.10M %.10l %.5C %R" || true
    echo
    local out name
    for out in out/*/; do
        [ -d "$out" ] || continue
        name=$(basename "$out")
        if [ -f "$out/tasks.txt" ]; then
            local nt done
            nt=$(wc -l < "$out/tasks.txt"); done=$(ls "$out"/task_*.log 2>/dev/null | wc -l)
            awk -v name="$name" -v nt="$nt" -v done="$done" '/^RESULT/{s += $(NF-2); v += $NF} /^SURVIVOR/{sl++}
                END{printf "%-10s kernel  %d/%d tasks done  signed codes %.4g  survivors %d%s\n", name, done, nt, s, v, (v ? "  <-- check SURVIVOR lines" : "")}' \
                /dev/null "$out"/task_*.log
        elif ls "$out"/export_a*.log >/dev/null 2>&1; then
            for f in "$out"/export_a*.log; do printf "%-10s export  %s: %s\n" "$name" "$(basename "$f" .log)" "$(grep '^exported' "$f" || echo 'not finished')"; done
        elif ls "$out"/task_*.log >/dev/null 2>&1; then
            printf "%-10s solver  " "$name"
            cat "$out"/task_*.log 2>/dev/null | grep -o '"status": "[A-Z_()a-z]*"' | sort | uniq -c | tr '\n' ' '; echo
            grep -l '"status": "SAT"\|"status": "OPTIMAL"\|"status": "FEASIBLE"' "$out"/task_*.log 2>/dev/null | sed 's/^/    FOUND: /' || true
        fi
        ls "$out"/*.part 2>/dev/null | sed 's/^/    unfinished (running, failed or timed out): /' || true
    done
}

case "${1:-}" in
    check) cmd_check ;;
    setup) cmd_setup ;;
    status) cmd_status ;;
    submit)
        case "${2:-}" in
            A1) submit_kernel A1_a6 results/structs_r2_n9_d3_a6.txt 9 3 7
                submit_kernel A1_a7 results/structs_r2_n9_d3_a7.txt 9 3 7 ;;
            A2-export) submit_export A2_export 6 10 2 0 4 120 24 ;;
            A2) for a in 0 1 2 3 4; do submit_kernel "A2_a$a" "results/structs_r6_n10_d2_a$a.txt" 10 2 4; done ;;
            B1) submit_list B1_sat cluster/b1_sat_instances.txt 1 16 cluster/jobs/sat.sbatch ;;
            B2) [ -n "$GUROBI_MODULE" ] || die "no Gurobi module was found by setup; B2 needs Gurobi"
                submit_list B2_mip cluster/b2_mip_instances.txt 16 32 cluster/jobs/mip.sbatch ;;
            A3-export) submit_export A3_export 3 10 3 0 6 120 48 ;;
            A3-low) for a in 0 1 2 3 4 5 6; do submit_kernel "A3_a$a" "results/structs_r3_n10_d3_a$a.txt" 10 3 7; done ;;
            A3-top) [ "${3:-}" = "--yes" ] || die "A3-top needs about 86,000 core-hours (2.3e15 signed codes). Ask RCI first, then: submit A3-top --yes"
                submit_kernel A3_a7 results/structs_r3_n10_d3_a7.txt 10 3 7 ;;
            *) die "unknown run '${2:-}'. Runs: A1 A2-export A2 B1 B2 A3-export A3-low A3-top" ;;
        esac ;;
    *) sed -n '2,20p' "$0"; exit 1 ;;
esac
