#!/usr/bin/env python3
"""PreToolUse hook: the agent may only touch files inside its run directory."""
import json
import os
import re
import sys

RUN = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inside(p):
    p = os.path.realpath(os.path.join(RUN, os.path.expanduser(p)))
    return p == RUN or p.startswith(RUN + os.sep)


def deny(why):
    print(f"Blocked: {why}. You may only read and edit files inside {RUN}.", file=sys.stderr)
    sys.exit(2)


d = json.load(sys.stdin)
tool, inp = d.get("tool_name", ""), d.get("tool_input", {}) or {}
if tool in ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit"):
    p = inp.get("file_path") or inp.get("notebook_path") or ""
    if not inside(p):
        deny(f"{tool} outside the sandbox: {p}")
elif tool in ("Glob", "Grep"):
    p = inp.get("path") or "."
    if not inside(p):
        deny(f"{tool} outside the sandbox: {p}")
elif tool == "Bash":
    cmd = inp.get("command", "")
    for tok in re.findall(r"[^\s'\"|;&<>()]+", cmd):
        if tok.startswith(("/", "~")) or ".." in tok:
            if not inside(tok):
                deny(f"path outside the sandbox in shell command: {tok}")
sys.exit(0)
