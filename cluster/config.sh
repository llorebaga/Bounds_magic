# Settings for the RCI cluster (FEL CTU). Edit RCI_USER before the first use.
# Read by from_pc.sh (on your computer) and by rci.sh (on the cluster).

RCI_USER="${RCI_USER:-baladllo}"         # your RCI login (CTU username)
RCI_HOST="${RCI_HOST:-login.rci.cvut.cz}"   # accepts password login (needed once, for `keys`); login3 accepts keys only
REMOTE_DIR="${REMOTE_DIR:-magic_runs}"   # working directory on the cluster, relative to your home

# Partitions. Empty = chosen automatically from `sinfo`: the CPU partition with the shortest time limit
# that still fits the job, preferring names that start with PART_PREFER.
PART_PREFER="${PART_PREFER:-amd}"
PARTITION="${PARTITION:-}"

# Cores per kernel task (capped automatically at the cores of one node of the chosen partition).
KERNEL_CPUS="${KERNEL_CPUS:-64}"
# Work per kernel task, in core-hours at the measured 7.5e6 signed codes per core-second.
TASK_CORE_HOURS="${TASK_CORE_HOURS:-128}"
# SAT and MIP instances: wall-clock limit per instance in seconds (23 h fits a 24 h partition).
SOLVER_SECONDS="${SOLVER_SECONDS:-82800}"
# At most this many array tasks of one run at a time (be fair to other users; 20 x 64 cores = 1280 cores).
ARRAY_THROTTLE="${ARRAY_THROTTLE:-20}"
# Optional account (only if RCI asks you to charge a project): e.g. SBATCH_ACCOUNT="--account=xyz"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
