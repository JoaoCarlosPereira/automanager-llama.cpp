#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${PROJECT_DIR}/manager.pid"

if [[ -f "${PID_FILE}" ]]; then
    PID=$(cat "${PID_FILE}" 2>/dev/null || true)
    if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
        echo "Encerrando AutoManager (PID: ${PID})..."
        kill -15 "${PID}" 2>/dev/null || true
        for i in {1..10}; do
            if ! kill -0 "${PID}" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "${PID}" 2>/dev/null; then
            echo "Forçando encerramento..."
            kill -9 "${PID}" 2>/dev/null || true
        fi
        echo "AutoManager encerrado."
    else
        echo "AutoManager não está em execução."
    fi
    rm -f "${PID_FILE}"
else
    # Fallback caso não tenha pidfile
    pkill -f "llama_manager.py" || true
    echo "AutoManager encerrado."
fi
