#!/bin/bash
# Save the account you are logged into (interactive claude, $CLAUDE_CONFIG_DIR) as a slot for the autoloop.
#   claude /login            then      bash cap/auto/save_creds.sh buzhao
# The loop uses its own config dir (config-loop/) so your interactive session never overwrites its live token.
# Priority: edit config-loop/creds/order (one name per line, first = preferred).
set -euo pipefail
name="${1:?usage: save_creds.sh <name>}"
src="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
loop="${CAP_LOOP_CONFIG:-$(dirname "$src")/config-loop}"
mkdir -p "$loop/creds"; chmod 700 "$loop" "$loop/creds"
[ -f "$loop/.claude.json" ] || cp "$src/.claude.json" "$loop/.claude.json"       # onboarding state
[ -e "$loop/projects" ] || ln -s "$src/projects" "$loop/projects"                # share session transcripts
cp "$src/.credentials.json" "$loop/creds/$name.json"; chmod 600 "$loop/creds/$name.json"
if [ ! -f "$loop/.credentials.json" ]; then cp "$loop/creds/$name.json" "$loop/.credentials.json"; echo "$name" > "$loop/creds/current"; fi
grep -qE "^$name( |$)" "$loop/creds/order" 2>/dev/null || echo "$name" >> "$loop/creds/order"   # `name weekly Tue 5pm` = spend before its weekly reset
python3 - "$loop/creds/limits.json" "$name" <<'PY'
import json, sys, os
p, n = sys.argv[1], sys.argv[2]
d = json.load(open(p)) if os.path.exists(p) else {}
d.pop(n, None); json.dump(d, open(p, "w"), indent=1)      # a fresh login is not limited
PY
echo "loop config: $loop"; echo "slots: $(ls "$loop"/creds/*.json | xargs -n1 basename | grep -v limits | tr '\n' ' ')"
echo "order: $(tr '\n' ' ' < "$loop/creds/order")"; echo "live: $(cat "$loop/creds/current")"
