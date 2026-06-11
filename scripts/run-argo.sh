#!/bin/bash
# run-argo.sh — Run the precision-rewrite workflow with the Anthropic SDK
# routed through the Argo proxy on JLSE.
#
# Sets up (and reuses, if already running) two layers:
#   1. An SSH tunnel from local :8082 to apps.inside.anl.gov:443 via
#      homes.cels.anl.gov.
#   2. A local Anthropic-compatible shim (claude-argo-proxy.py) listening on
#      :8083 that forwards /argoapi/* to the tunnel.
#
# Then runs `python -m workflow.run "$@"` with ANTHROPIC_BASE_URL and
# ANTHROPIC_AUTH_TOKEN pointed at the shim. The `anthropic` Python SDK honors
# both env vars, so no workflow code changes are needed.
#
# If the tunnel or proxy were already up (e.g. started by another tool in this
# session), this script reuses them and does NOT tear them down on exit.
#
# Usage:
#   ./scripts/run-argo.sh test-kernels/kokkos/mixed/nbody_force.cpp
#
# Environment overrides:
#   ARGO_REMOTE_HOST              SSH host (default: homes.cels.anl.gov)
#   ARGO_TUNNEL_PORT              Local SSH-tunnel port (default: 8082)
#   ARGO_PROXY_PORT               Local shim port (default: 8083)
#   ARGO_PROXY_SCRIPT             Path to claude-argo-proxy.py
#                                 (default: ${HOME}/argo-shim-lite/claude-argo-proxy.py)
#   ARGO_USER                     Username passed as ANTHROPIC_AUTH_TOKEN
#                                 (default: $USER)
#   AGENT_PRECISION_KOKKOS_ROOT   Kokkos install prefix used by the
#                                 compile_baseline_driver orchestrator tool.
#                                 Defaults to ${PWD}/kokkos (the bundled
#                                 install) if unset and that directory exists;
#                                 otherwise left unset and the compile tool
#                                 returns a non-fatal error.

set -euo pipefail

REMOTE_HOST="${ARGO_REMOTE_HOST:-homes.cels.anl.gov}"
TUNNEL_LOCAL_PORT="${ARGO_TUNNEL_PORT:-8082}"
TUNNEL_REMOTE_HOST="apps.inside.anl.gov"
TUNNEL_REMOTE_PORT=443
PROXY_PORT="${ARGO_PROXY_PORT:-8083}"
PROXY_SCRIPT="${ARGO_PROXY_SCRIPT:-${HOME}/argo-shim-lite/claude-argo-proxy.py}"
CONTROL_PATH="/tmp/ssh-argo-workflow-$$"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

STARTED_TUNNEL=false
STARTED_PROXY=false
PROXY_PID=""

cleanup() {
    if [ "${STARTED_PROXY}" = true ] && [ -n "${PROXY_PID}" ]; then
        echo -e "\n${YELLOW}Stopping proxy (pid ${PROXY_PID})...${NC}"
        kill "${PROXY_PID}" 2>/dev/null || true
    fi
    if [ "${STARTED_TUNNEL}" = true ]; then
        echo -e "${YELLOW}Closing SSH tunnel...${NC}"
        ssh -O exit -o ControlPath="${CONTROL_PATH}" "${REMOTE_HOST}" 2>/dev/null || true
    fi
}
trap cleanup EXIT SIGINT SIGTERM

# ---------------------------------------------------------------------------
# Step 1: SSH tunnel (reuse if already bound)
# ---------------------------------------------------------------------------
if lsof -i :"${TUNNEL_LOCAL_PORT}" >/dev/null 2>&1; then
    echo -e "${GREEN}SSH tunnel already running on port ${TUNNEL_LOCAL_PORT} — reusing.${NC}"
else
    echo -e "${YELLOW}Starting SSH tunnel to ${TUNNEL_REMOTE_HOST} via ${REMOTE_HOST}...${NC}"
    echo -e "${YELLOW}(You may need to complete MFA / Duo)${NC}"
    ssh -f -N \
        -o ControlMaster=yes \
        -o ControlPath="${CONTROL_PATH}" \
        -L "${TUNNEL_LOCAL_PORT}:${TUNNEL_REMOTE_HOST}:${TUNNEL_REMOTE_PORT}" \
        "${REMOTE_HOST}"
    STARTED_TUNNEL=true
    echo -e "${GREEN}SSH tunnel established (port ${TUNNEL_LOCAL_PORT}).${NC}"
fi

# ---------------------------------------------------------------------------
# Step 2: Local Anthropic-compatible shim (reuse if already bound)
# ---------------------------------------------------------------------------
if lsof -i :"${PROXY_PORT}" >/dev/null 2>&1; then
    echo -e "${GREEN}Proxy already running on port ${PROXY_PORT} — reusing.${NC}"
else
    if [ ! -f "${PROXY_SCRIPT}" ]; then
        echo -e "${RED}Proxy script not found: ${PROXY_SCRIPT}${NC}"
        echo -e "${RED}Override with ARGO_PROXY_SCRIPT=/path/to/claude-argo-proxy.py${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Starting local Argo proxy (${PROXY_SCRIPT})...${NC}"
    python3 "${PROXY_SCRIPT}" &
    PROXY_PID=$!
    STARTED_PROXY=true
    sleep 2
    if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
        echo -e "${RED}Proxy failed to start. Is aiohttp installed? (pip install aiohttp)${NC}"
        exit 1
    fi
    echo -e "${GREEN}Proxy running (port ${PROXY_PORT}).${NC}"
fi

# ---------------------------------------------------------------------------
# Step 3: Run the workflow with the Anthropic SDK pointed at the shim
# ---------------------------------------------------------------------------
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage: $0 <kernel_file>${NC}"
    echo -e "${YELLOW}Example: $0 test-kernels/kokkos/mixed/nbody_force.cpp${NC}"
    exit 2
fi

# Default AGENT_PRECISION_KOKKOS_ROOT to the bundled install at ${PWD}/kokkos
# if the caller did not set it and that directory looks like a Kokkos prefix.
# This makes the compile_baseline_driver orchestrator tool actually work
# out of the box from the repo root, without forcing every CWD to have one.
if [ -z "${AGENT_PRECISION_KOKKOS_ROOT:-}" ] && [ -d "${PWD}/kokkos/include" ] && [ -d "${PWD}/kokkos/lib" ]; then
    export AGENT_PRECISION_KOKKOS_ROOT="${PWD}/kokkos"
    echo -e "${GREEN}AGENT_PRECISION_KOKKOS_ROOT defaulted to ${AGENT_PRECISION_KOKKOS_ROOT}${NC}"
fi

echo -e "${GREEN}Running workflow via Argo...${NC}"
ANTHROPIC_BASE_URL="http://127.0.0.1:${PROXY_PORT}/argoapi/" \
    ANTHROPIC_AUTH_TOKEN="${ARGO_USER:-$USER}" \
    python3 -m workflow.run "$@"
