#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="${TMPDIR:-/tmp}/chain-reaction-equivalence"

mkdir -p "$TMP_DIR"

TS_JSON="$TMP_DIR/fixture.ts.json"
CPP_JSON="$TMP_DIR/fixture.cpp.json"
CPP_BIN="$TMP_DIR/equivalence_main"

node "$ROOT_DIR/core/game.ts" > "$TS_JSON"
c++ -std=c++11 -Wall -Wextra -Werror "$ROOT_DIR/tests/equivalence_main.cpp" -o "$CPP_BIN"
"$CPP_BIN" > "$CPP_JSON"

if cmp -s "$TS_JSON" "$CPP_JSON"; then
    printf 'TypeScript and C++ fixtures match.\n'
else
    printf 'TypeScript and C++ fixtures differ.\n' >&2
    diff -u "$TS_JSON" "$CPP_JSON" >&2
    exit 1
fi
