#!/bin/bash
#SBATCH --job-name=magic-kernel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
# Array task of the exact kernel. Set by rci.sh: STRUCTS TASKS N D M OUT (and GALOIS, default 1).
# Each line of TASKS is "L first last" (whole structures) or "S line k K" (piece k of K of one structure).
set -eo pipefail
. "$MAGIC_DIR/cluster/env.sh"
t=$SLURM_ARRAY_TASK_ID
read -r kind x y z < <(sed -n "$((t + 1))p" "$TASKS")
tmp="${TMPDIR:-/tmp}/magic_${SLURM_JOB_ID}_${t}.txt"
if [ "$kind" = L ]; then
    sed -n "${x},${y}p" "$STRUCTS" > "$tmp"; export SHARD=0 NSHARDS=1
else
    sed -n "${x}p" "$STRUCTS" > "$tmp"; export SHARD=$y NSHARDS=$z
fi
export GALOIS="${GALOIS:-1}"
log="$OUT/task_$t.log"
{
    echo "TASK $t: $kind $x $y $z  file=$STRUCTS n=$N d=$D m=$M GALOIS=$GALOIS host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
    cd "$MAGIC_DIR/scripts"
    julia -t "$SLURM_CPUS_PER_TASK" fast_enum.jl "$tmp" "$N" "$D" "$M"
} > "$log.part" 2>&1
mv "$log.part" "$log"
rm -f "$tmp"
