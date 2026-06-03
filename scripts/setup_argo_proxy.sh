#!/usr/bin/env bash
# Setup Argo proxy tunnel from JLSE through CELS hosts.
# This is infrastructure-only for local LLM agent calls (no MCP assumptions).
#
# Example:
#   bash scripts/setup_argo_proxy.sh
#   bash scripts/setup_argo_proxy.sh --model gpt4o --start-port 8082
#
# After startup:
#   export HTTP_PROXY=http://127.0.0.1:<port>
#   export HTTPS_PROXY=http://127.0.0.1:<port>
#
# Keep this script running while using the proxy. Ctrl+C cleans up.

set -euo pipefail

REMOTE_HOST="${ARGO_REMOTE_HOST:-${USER}@homes.cels.anl.gov}"
JUMP_HOST="${ARGO_JUMP_HOST:-${USER}@logins.cels.anl.gov}"
REMOTE_PROXY_DIR="${ARGO_REMOTE_PROXY_DIR:-~/lmtools-main}"
START_PORT="${ARGO_PROXY_START_PORT:-8082}"
MAX_PORT_ATTEMPTS="${ARGO_PROXY_MAX_ATTEMPTS:-5}"
ARGO_USER="${ARGO_USERNAME:-${USER}}"
MODEL="${ARGO_MODEL:-gpt4o}"
CONTROL_PATH="${ARGO_SSH_CONTROL_PATH:-/tmp/ssh-control-argo-%r@%h:%p}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
  cat <<EOF
setup_argo_proxy.sh - tunnel JLSE -> homes apiproxy for Argo

Options:
  --remote-host USER@HOST    Remote host running apiproxy (default: ${REMOTE_HOST})
  --jump-host USER@HOST      SSH jump host (default: ${JUMP_HOST})
  --remote-proxy-dir PATH    Directory containing ./bin/apiproxy (default: ${REMOTE_PROXY_DIR})
  --start-port N             First local/remote port to try (default: ${START_PORT})
  --max-attempts N           Port attempts before failing (default: ${MAX_PORT_ATTEMPTS})
  --argo-user NAME           Argo username passed to apiproxy (default: ${ARGO_USER})
  --model NAME               Model name passed to apiproxy (default: ${MODEL})
  -h, --help                 Show help

Environment overrides:
  ARGO_REMOTE_HOST, ARGO_JUMP_HOST, ARGO_REMOTE_PROXY_DIR,
  ARGO_PROXY_START_PORT, ARGO_PROXY_MAX_ATTEMPTS,
  ARGO_USERNAME, ARGO_MODEL, ARGO_SSH_CONTROL_PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-host) REMOTE_HOST="$2"; shift ;;
    --jump-host) JUMP_HOST="$2"; shift ;;
    --remote-proxy-dir) REMOTE_PROXY_DIR="$2"; shift ;;
    --start-port) START_PORT="$2"; shift ;;
    --max-attempts) MAX_PORT_ATTEMPTS="$2"; shift ;;
    --argo-user) ARGO_USER="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

SSH_PID=""
CURRENT_PORT="${START_PORT}"
LOG_FILE=""

cleanup() {
  echo -e "\n${YELLOW}Cleaning up proxy tunnel...${NC}"
  if [[ -n "${SSH_PID}" ]]; then
    kill "${SSH_PID}" 2>/dev/null || true
  fi
  ssh -o ControlPath="${CONTROL_PATH}" -J "${JUMP_HOST}" "${REMOTE_HOST}" "pkill -f apiproxy" \
    2>/dev/null || true
  ssh -O exit -o ControlPath="${CONTROL_PATH}" -J "${JUMP_HOST}" "${REMOTE_HOST}" \
    2>/dev/null || true
  echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT INT TERM

echo -e "${GREEN}Setting up Argo proxy tunnel...${NC}"
echo -e "${YELLOW}Remote host: ${REMOTE_HOST}${NC}"
echo -e "${YELLOW}Jump host:   ${JUMP_HOST}${NC}"
echo -e "${YELLOW}(You may need passphrase + Duo)${NC}"

ssh -fN \
  -o ControlMaster=yes \
  -o ControlPath="${CONTROL_PATH}" \
  -o ControlPersist=10m \
  -J "${JUMP_HOST}" \
  "${REMOTE_HOST}"

echo -e "${GREEN}SSH control connection established.${NC}"

ssh -o ControlPath="${CONTROL_PATH}" -J "${JUMP_HOST}" "${REMOTE_HOST}" "pkill -f apiproxy" \
  2>/dev/null || true

SUCCESS=false
ATTEMPT=0

while [[ "${ATTEMPT}" -lt "${MAX_PORT_ATTEMPTS}" && "${SUCCESS}" == "false" ]]; do
  ATTEMPT=$((ATTEMPT + 1))
  echo -e "${YELLOW}Attempt ${ATTEMPT}/${MAX_PORT_ATTEMPTS} on port ${CURRENT_PORT}...${NC}"

  LOG_FILE="/tmp/argo-proxy-${CURRENT_PORT}-$(date +%Y%m%d-%H%M%S).log"
  ssh -o ControlPath="${CONTROL_PATH}" \
    -J "${JUMP_HOST}" \
    -L "${CURRENT_PORT}:127.0.0.1:${CURRENT_PORT}" \
    "${REMOTE_HOST}" \
    "cd ${REMOTE_PROXY_DIR} && ./bin/apiproxy --argo-user=${ARGO_USER} -model ${MODEL} --port ${CURRENT_PORT}" \
    > "${LOG_FILE}" 2>&1 &

  SSH_PID=$!
  sleep 4

  if ! kill -0 "${SSH_PID}" 2>/dev/null; then
    if rg -n "address already in use|bind.*failed" "${LOG_FILE}" >/dev/null 2>&1; then
      echo -e "${YELLOW}Port ${CURRENT_PORT} in use; trying next.${NC}"
      CURRENT_PORT=$((CURRENT_PORT + 1))
      continue
    fi
    echo -e "${RED}Tunnel/proxy failed to start. Check log: ${LOG_FILE}${NC}"
    exit 1
  fi

  if curl -s --max-time 5 "http://127.0.0.1:${CURRENT_PORT}/v1/models" >/dev/null 2>&1; then
    SUCCESS=true
  else
    # Some deployments take longer; treat as success if process stays alive.
    SUCCESS=true
    echo -e "${YELLOW}Proxy did not answer /v1/models yet; keeping tunnel up.${NC}"
  fi
done

if [[ "${SUCCESS}" != "true" ]]; then
  echo -e "${RED}Failed to start proxy after ${MAX_PORT_ATTEMPTS} attempts.${NC}"
  exit 1
fi

echo -e "\n${GREEN}Argo proxy ready (or initializing).${NC}"
echo -e "${GREEN}Endpoint: http://127.0.0.1:${CURRENT_PORT}${NC}"
echo -e "${YELLOW}Use in this shell:${NC}"
echo "  export HTTP_PROXY=http://127.0.0.1:${CURRENT_PORT}"
echo "  export HTTPS_PROXY=http://127.0.0.1:${CURRENT_PORT}"
echo "  export ARGO_USERNAME=${ARGO_USER}"
echo -e "${YELLOW}Test:${NC} curl -s http://127.0.0.1:${CURRENT_PORT}/v1/models | head"
echo -e "${YELLOW}Keep this script running; Ctrl+C to stop.${NC}"

wait "${SSH_PID}"
