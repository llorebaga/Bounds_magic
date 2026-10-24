# Sourced on the cluster by rci.sh and by every job: module system, settings, and the software chosen by
# `rci.sh setup` (stored in cluster/.software).
MAGIC_DIR="${MAGIC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MAGIC_DIR
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh /etc/profile.d/modules.sh \
             /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do
        [ -f "$f" ] && . "$f" && break
    done
fi
. "$MAGIC_DIR/cluster/config.sh"
[ -f "$MAGIC_DIR/cluster/.software" ] && . "$MAGIC_DIR/cluster/.software"
true
