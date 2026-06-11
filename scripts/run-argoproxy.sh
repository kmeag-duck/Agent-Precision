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
#   ARGOPROXY_PORT                argo-proxy port (default: 52675)
#   AGENT_PRECISION_KOKKOS_ROOT   Kokkos install prefix used by the
#                                 compile_baseline_driver orchestrator tool.
#                                 Defaults to ${PWD}/kokkos (the bundled
#                                 install) if unset and that directory exists;
#                                 otherwise left unset and the compile tool
#                                 returns a non-fatal error.
#   AGENT_PRECISION_RUN_TIMEOUT_SEC
#                                 Wall-clock seconds the run_baseline_driver
#                                 tool will wait for the compiled driver to
#                                 finish. Defaults to 60 if unset; raise this
#                                 as kernels grow (e.g. once deployment-scale
#                                 inputs land).

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

# Default AGENT_PRECISION_KOKKOS_ROOT to the bundled install at ${PWD}/kokkos
# if the caller did not set it and that directory looks like a Kokkos prefix.
# This makes the compile_baseline_driver orchestrator tool actually work
# out of the box from the repo root, without forcing every CWD to have one.
if [ -z "${AGENT_PRECISION_KOKKOS_ROOT:-}" ] && [ -d "${PWD}/kokkos/include" ] && [ -d "${PWD}/kokkos/lib" ]; then
    export AGENT_PRECISION_KOKKOS_ROOT="${PWD}/kokkos"
    echo "AGENT_PRECISION_KOKKOS_ROOT defaulted to ${AGENT_PRECISION_KOKKOS_ROOT}"
fi

# Default AGENT_PRECISION_RUN_TIMEOUT_SEC to 60 if the caller did not set it.
# Matches the default baked into workflow.tools.run_baseline_driver; declared
# explicitly here so the value is visible at the wrapper-script level.
if [ -z "${AGENT_PRECISION_RUN_TIMEOUT_SEC:-}" ]; then
    export AGENT_PRECISION_RUN_TIMEOUT_SEC=60
fi

ANTHROPIC_BASE_URL="${BASE_URL}" \
    ANTHROPIC_AUTH_TOKEN="${USER}" \
    python3 -m workflow.run "$@"
