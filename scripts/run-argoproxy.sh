#!/bin/bash
# run-argoproxy.sh — Run the precision-rewrite workflow against the locally
# installed `argo-proxy` daemon (the same one OpenCode's provider.argo uses).
#
# Unlike scripts/run-argo.sh, this needs no SSH tunnel and no per-session Duo:
# argo-proxy maintains its own persistent auth and serves an Anthropic-compatible
# /v1/messages route at http://127.0.0.1:52675/v1/.
#
# Prereq: `argo-proxy serve` is already running. Check with:
#   curl -sf http://127.0.0.1:52675/health
#
# Usage:
#   ./scripts/run-argoproxy.sh test-kernels/kokkos/mixed/nbody_force.cpp
#
# Environment overrides:
#   ARGOPROXY_PORT   argo-proxy port (default: 52675)

set -euo pipefail

PORT="${ARGOPROXY_PORT:-52675}"
BASE_URL="http://127.0.0.1:${PORT}/v1/"

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "argo-proxy is not responding on :${PORT}. Start it with: argo-proxy serve" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <kernel_file>" >&2
    echo "Example: $0 test-kernels/kokkos/mixed/nbody_force.cpp" >&2
    exit 2
fi

ANTHROPIC_BASE_URL="${BASE_URL}" \
    ANTHROPIC_AUTH_TOKEN="${USER}" \
    python3 -m workflow.run "$@"
