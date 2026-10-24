#!/bin/bash
# Runs on YOUR computer, in the MobaXterm local terminal ("Start local terminal") or in Git Bash.
# Needs only ssh and tar. Set RCI_USER in cluster/config.sh first.
#   bash cluster/from_pc.sh keys          create an SSH key (if needed) and install it on RCI (asks your password once)
#   bash cluster/from_pc.sh push          copy scripts/, cluster/ and the input structure files to the cluster
#   bash cluster/from_pc.sh check         partitions, node sizes, modules (runs rci.sh check)
#   bash cluster/from_pc.sh setup         push, then create the software environment and self-test
#   bash cluster/from_pc.sh submit RUN    push, then submit RUN (A1, A2-export, A2, B1, B2, A3-export, A3-low, A3-top)
#   bash cluster/from_pc.sh status        queue and progress of every run
#   bash cluster/from_pc.sh pull          copy out/ and new structure files to results/cluster/ here
#   bash cluster/from_pc.sh ssh           open a shell in the working directory on the cluster
# Typical order: keys, setup, check, submit A1, submit B1, submit A2-export, (later) submit A2, status, pull.
set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
. "$REPO/cluster/config.sh"
[ "$RCI_USER" = CHANGE_ME ] && { echo "Set RCI_USER in cluster/config.sh (or run: RCI_USER=name bash cluster/from_pc.sh ...)"; exit 1; }
H="$RCI_USER@$RCI_HOST"
remote() { ssh -t "$H" "cd ~/$REMOTE_DIR && bash cluster/rci.sh $*"; }

push() {
    cd "$REPO"
    local inputs="results/structs_r2_n9_d3_a6.txt results/structs_r2_n9_d3_a7.txt results/structs_r3_n10_d3_a7.txt results/structs_r3_n8_d3_a5.txt"
    tar czf - scripts cluster $inputs | ssh "$H" "mkdir -p ~/$REMOTE_DIR && cd ~/$REMOTE_DIR && tar xzf - && \
        find cluster -name '*.sh' -o -name '*.sbatch' -o -name '*.txt' -o -name '*.py' | xargs sed -i 's/\r\$//' && \
        echo pushed to \$(pwd)"
}

case "${1:-}" in
    keys)
        [ -f ~/.ssh/id_ed25519.pub ] || ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
        ssh "$H" 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys' < ~/.ssh/id_ed25519.pub
        echo "key installed; 'ssh $H' should no longer ask for the password" ;;
    push) push ;;
    check|status) remote "$1" ;;
    setup) push; remote setup ;;
    submit) [ -n "${2:-}" ] || { echo "usage: submit RUN [--yes]"; exit 1; }; push; remote submit "$2" "${3:-}" ;;
    pull)
        mkdir -p "$REPO/results/cluster"
        ssh "$H" "cd ~/$REMOTE_DIR && tar czf - out \$(ls results/structs_r6_n10_* results/structs_r3_n10_d3_a[0-6].txt 2>/dev/null)" \
            | tar xzf - -C "$REPO/results/cluster"
        echo "copied to $REPO/results/cluster" ;;
    ssh) ssh -t "$H" "cd ~/$REMOTE_DIR 2>/dev/null; exec bash -l" ;;
    *) sed -n '2,13p' "$0"; exit 1 ;;
esac
