#!/usr/bin/env bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"

if [[ ! -f "${VENV_PYTHON}" ]]; then
    echo "Erro: ambiente virtual não encontrado em ${PROJECT_DIR}/.venv"
    exit 1
fi

"${VENV_PYTHON}" - <<'PY'
import os
import subprocess
import sys
import time

project_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
pid_file = os.path.join(project_dir, "manager.pid")
stdout_log = os.path.join(project_dir, "logs", "manager_stdout.log")
python_bin = os.path.join(project_dir, ".venv", "bin", "python")
entrypoint = os.path.join(project_dir, "llama_manager.py")

# Check if already running
if os.path.isfile(pid_file):
    try:
        with open(pid_file, "r") as f:
            old_pid = int(f.read().strip())
        os.kill(old_pid, 0)
        print(f"AutoManager já está em execução (PID: {old_pid}).")
        sys.exit(0)
    except (ValueError, OSError):
        pass

os.makedirs(os.path.join(project_dir, "logs"), exist_ok=True)
log_handle = open(stdout_log, "a", encoding="utf-8")

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

proc = subprocess.Popen(
    [python_bin, "-u", entrypoint],
    cwd=project_dir,
    env=env,
    stdout=log_handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

with open(pid_file, "w") as f:
    f.write(str(proc.pid))

time.sleep(1.5)
try:
    os.kill(proc.pid, 0)
    print(f"AutoManager iniciado com sucesso! (PID: {proc.pid})")
    print("Acesse:")
    print("  Local: http://localhost:8000/")
    print("  Rede:  http://192.168.2.117:8000/")
except OSError:
    print("Falha ao iniciar AutoManager. Verifique logs/manager_stdout.log")
    sys.exit(1)
PY
