#!/bin/bash
# Save the currently logged-in Claude account as a named slot the autoloop can rotate to on a usage limit.
#   bash cap/auto/save_creds.sh tavia      (after `claude /login` as that account)
set -euo pipefail
name="${1:?usage: save_creds.sh <name>}"
cfg="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
mkdir -p "$cfg/creds"; chmod 700 "$cfg/creds"
cp "$cfg/.credentials.json" "$cfg/creds/$name.json"; chmod 600 "$cfg/creds/$name.json"
echo "$name" > "$cfg/creds/current"
echo "saved slots:"; ls "$cfg/creds"/*.json | xargs -n1 basename; echo "current: $name"
