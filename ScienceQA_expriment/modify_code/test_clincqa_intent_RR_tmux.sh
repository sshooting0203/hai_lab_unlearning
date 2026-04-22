#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-clincqa_rr}"
RUN_SCRIPT="${RUN_SCRIPT:-/dataset/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code/test_clincqa_intent_RR.sh}"
LOG_DIR="${LOG_DIR:-./logs_RR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${SESSION_NAME}_${TS}.log"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "[ERROR] run script not found: ${RUN_SCRIPT}"
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "[ERROR] tmux session already exists: ${SESSION_NAME}"
  echo "기존 세션 확인: tmux attach -t ${SESSION_NAME}"
  echo "세션 종료 후 재실행: tmux kill-session -t ${SESSION_NAME}"
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "bash -lc 'chmod +x \"${RUN_SCRIPT}\" && \"${RUN_SCRIPT}\" 2>&1 | tee \"${LOG_FILE}\"'"

echo "[OK] tmux session started: ${SESSION_NAME}"
echo "[OK] log file: ${LOG_FILE}"
echo
echo "attach  : tmux attach -t ${SESSION_NAME}"
echo "detach  : Ctrl-b then d"
echo "list    : tmux ls"
echo "monitor : tail -f ${LOG_FILE}"
echo "kill    : tmux kill-session -t ${SESSION_NAME}"