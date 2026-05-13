import subprocess
import os
import signal
import glob
import psutil
import json
import logging
import re
import uuid
import threading
import requests
import socket
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import time

# Configura log para o arquivo do gerenciador
logging.basicConfig(filename='/root/manager.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Automanager Llama.cpp")

MODELS_DIR = "/media/docker/models"
SERVER_LOG_PATH = "/root/llama_server.log"
CONFIG_PATH = "/root/automanager_config.json"

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

class GPUWeight(BaseModel):
    index: int
    weight: float
    name: str
    active: bool = True

class StartRequest(BaseModel):
    path: str
    mmproj_path: Optional[str] = None
    gpu_weights: List[GPUWeight]
    context_size: int = 131072

class DeleteRequest(BaseModel):
    path: str

class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None

class SetDefaultRequest(BaseModel):
    path: Optional[str] = None

class RenameRequest(BaseModel):
    path: str
    new_name: str

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except: return {}
    return {}

def save_config(config):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        logging.error(f"Erro ao salvar config: {e}")

def update_model_config(path, context_size, gpu_weights, mmproj_path=None):
    config = load_config()
    if "model_configs" not in config:
        config["model_configs"] = {}
    
    config["model_configs"][path] = {
        "context_size": context_size,
        "mmproj_path": mmproj_path,
        "gpu_weights": [w.model_dump() if hasattr(w, 'model_dump') else w for w in gpu_weights]
    }
    save_config(config)

# Estado dos downloads
downloads = {}

def download_model_task(download_id: str, url: str, filename: Optional[str]):
    try:
        if not filename:
            filename = url.split("/")[-1].split("?")[0]
            if not filename.endswith(".gguf"):
                filename += ".gguf"
        
        model_name_folder = filename.replace(".gguf", "")
        model_specific_dir = os.path.join(MODELS_DIR, model_name_folder)
        os.makedirs(model_specific_dir, exist_ok=True)
        path = os.path.join(model_specific_dir, filename)
        
        if os.path.exists(path):
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{int(time.time())}{ext}"
            path = os.path.join(model_specific_dir, filename)

        downloads[download_id] = {"filename": filename, "status": "downloading", "progress": 0}
        logging.info(f"Iniciando download: {url} -> {path}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        downloaded = 0
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192 * 4):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        downloads[download_id]["progress"] = round((downloaded / total_size) * 100, 2)
        
        downloads[download_id]["status"] = "completed"
        downloads[download_id]["progress"] = 100
        logging.info(f"Download concluído: {filename}")
    except Exception as e:
        logging.error(f"Erro no download {download_id}: {e}")
        downloads[download_id]["status"] = "failed"
        downloads[download_id]["error"] = str(e)

def get_gguf_models():
    files = glob.glob(os.path.join(MODELS_DIR, "**/*.gguf"), recursive=True)
    return sorted(files)

def get_gpu_info():
    try:
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        output = subprocess.check_output("llama-server --help 2>&1", shell=True, env=env).decode()
        pattern = r"Device (\d+): (.*?), compute capability.*?, VRAM: (\d+) MiB"
        matches = re.findall(pattern, output)
        gpus = []
        for match in matches:
            idx, name, vram = match
            gpus.append({"index": int(idx), "name": name.strip(), "vram": int(vram)})
        if not gpus:
            smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"]).decode()
            for line in smi_output.strip().split("\n"):
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({"index": int(parts[0]), "name": parts[1], "vram": int(parts[2])})
        return gpus
    except Exception as e:
        logging.error(f"Error getting GPU info: {e}")
        return []

def find_llama_server():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            name = proc.info['name'] or ""
            cmdline = proc.info['cmdline'] or []
            if 'llama-server' in name or (cmdline and 'llama-server' in cmdline[0]):
                model_name = None
                for i in range(len(cmdline)-1):
                    if cmdline[i] in ["-m", "--model"]:
                        model_name = os.path.basename(cmdline[i+1])
                        break
                if model_name:
                    return {
                        "running": True, 
                        "pid": proc.info['pid'], 
                        "model": model_name,
                        "start_time": proc.info['create_time']
                    }
        except (psutil.NoSuchProcess, psutil.AccessDenied): continue
    return {"running": False}

# Estado global para controle de auto-retry
last_start_request = None
recovery_state = {"active": False, "failed": False, "message": ""}
retry_lock = threading.Lock()

def monitor_oom():
    global last_start_request
    while True:
        try:
            if os.path.exists(SERVER_LOG_PATH):
                with open(SERVER_LOG_PATH, "r") as f:
                    f.seek(0, os.SEEK_END)
                    while True:
                        line = f.readline()
                        if not line:
                            if not find_llama_server()["running"] and not recovery_state["active"]:
                                break
                            time.sleep(1)
                            continue
                        
                        if "out of memory" in line.lower() or "failed to allocate" in line.lower():
                            logging.warning("OOM detectado no log! Iniciando ajuste de carga...")
                            handle_oom_retry()
                            break
            time.sleep(2)
        except Exception as e:
            logging.error(f"Erro no monitor de OOM: {e}")
            time.sleep(5)

def handle_oom_retry():
    global last_start_request, recovery_state
    with retry_lock:
        if not last_start_request:
            return
        
        recovery_state["active"] = True
        recovery_state["failed"] = False
        recovery_state["message"] = "OOM detectado. Reajustando pesos..."
        
        req = last_start_request
        weights = req.gpu_weights
        if len(weights) <= 1:
            logging.error("OOM em single GPU ou sem pesos. Impossível ajustar.")
            recovery_state["active"] = False
            recovery_state["failed"] = True
            recovery_state["message"] = "Capacidade insuficiente (Single GPU)."
            return

        max_w = max(w.weight for w in weights)
        if max_w <= 15:
            logging.error("Pesos já estão no limite de redistribuição. Falha total.")
            recovery_state["active"] = False
            recovery_state["failed"] = True
            recovery_state["message"] = "Incompatível com a capacidade atual."
            return

        main_gpu = max(weights, key=lambda x: x.weight)
        other_gpus = [w for w in weights if w != main_gpu]
        
        reduction = 10.0
        main_gpu.weight -= reduction
        share = reduction / len(other_gpus)
        for og in other_gpus:
            og.weight += share
        
        logging.info(f"Retentando com novos pesos: {[f'{w.index}:{w.weight}' for w in weights]}")
        update_model_config(req.path, req.context_size, req.gpu_weights, req.mmproj_path)
        execute_start(req)
        time.sleep(3)
        if recovery_state["active"]:
            recovery_state["active"] = False

def execute_start(req: StartRequest):
    global recovery_state
    stop_model()
    try:
        all_gpus = get_gpu_info()
        max_idx = max(g['index'] for g in all_gpus) if all_gpus else 0
        
        # Filtra apenas GPUs ativas para o cálculo
        active_weights = [gw for gw in req.gpu_weights if gw.active]
        weights_map = {gw.index: gw.weight for gw in active_weights}
        
        split = []
        total_user = sum(weights_map.values()) or 1
        for i in range(max_idx + 1):
            split.append(f"{weights_map.get(i, 0)/total_user:.4f}")
        
        # Main GPU deve ser a de maior peso entre as ATIVAS
        main_gpu = str(max(weights_map, key=weights_map.get)) if weights_map else "0"
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        
        cmd = [
            "llama-server", "-m", req.path, "-ngl", "99", "--flash-attn", "on", 
            "--host", "0.0.0.0", "--port", "8085", "--tools", "all", 
            "--parallel", "1", "--ctx-size", str(req.context_size), "--mlock", 
            "--main-gpu", main_gpu, "--tensor-split", ",".join(split)
        ]
        
        if req.mmproj_path:
            cmd.extend(["--mmproj", req.mmproj_path])
        else:
            cmd.append("--mmproj-auto")
            
        logging.info(f"START EXEC: {' '.join(cmd)}")
        with open(SERVER_LOG_PATH, "w") as f:
            f.write("")
            
        subprocess.Popen(cmd, stdout=open(SERVER_LOG_PATH, "a"), stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env)
        return True
    except Exception as e:
        logging.error(f"Execution Error: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=monitor_oom, daemon=True).start()
    config = load_config()
    default_model = config.get("default_model")
    model_configs = config.get("model_configs", {})
    
    if default_model and os.path.exists(default_model):
        if not find_llama_server().get("running"):
            logging.info(f"Auto-start: {default_model}")
            try:
                # Tenta carregar config salva para o modelo padrão
                saved_cfg = model_configs.get(default_model)
                
                if saved_cfg:
                    weights = [GPUWeight(**w) if isinstance(w, dict) else w for w in saved_cfg.get("gpu_weights", [])]
                    # Garante que temos todos os campos
                    for w in weights:
                        if not hasattr(w, 'active'): w.active = True
                    
                    context_size = saved_cfg.get("context_size", 131072)
                    mmproj_path = saved_cfg.get("mmproj_path")
                    req = StartRequest(path=default_model, gpu_weights=weights, context_size=context_size, mmproj_path=mmproj_path)
                else:
                    gpus = get_gpu_info()
                    weights = []
                    max_vram = max(g['vram'] for g in gpus) if gpus else 0
                    main_gpu_idx = next((g['index'] for g in gpus if g['vram'] == max_vram), -1)
                    for g in gpus:
                        val = 100.0 if g['index'] == main_gpu_idx else 0.0
                        weights.append(GPUWeight(index=g['index'], weight=val, name=g['name']))
                    req = StartRequest(path=default_model, gpu_weights=weights)
                
                global last_start_request
                last_start_request = req
                execute_start(req)
            except Exception as e:
                logging.error(f"Auto-start error: {e}")

@app.get("/metrics")
def get_metrics():
    try:
        gpu_output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"]).decode()
        gpus = []
        for line in gpu_output.strip().split("\n"):
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                idx, util, mem_used, mem_total, temp, power = parts
                mem_used_f = float(mem_used)
                mem_total_f = float(mem_total)
                vram_pct = round((mem_used_f / mem_total_f) * 100, 1) if mem_total_f > 0 else 0
                gpus.append({
                    "index": int(idx), "util": util, "mem_used": mem_used, "mem_total": mem_total,
                    "vram_pct": vram_pct, "temp": temp, "power": power.split('.')[0] if '.' in power else power
                })
        return {"cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent, "gpus": gpus}
    except: return {"cpu": 0, "ram": 0, "gpus": []}

@app.get("/logs")
def stream_logs():
    def generate():
        if not os.path.exists(SERVER_LOG_PATH):
            yield "Arquivo de log não encontrado.\n"
            return
        with open(SERVER_LOG_PATH, 'r') as f:
            lines = f.readlines()
            for line in lines[-100:]: yield line
            while True:
                line = f.readline()
                if not line: time.sleep(0.5); continue
                yield line
    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/", response_class=HTMLResponse)
def index():
    all_files = get_gguf_models()
    models = []
    projectors = []
    for f in all_files:
        name = os.path.basename(f).lower()
        item = {"path": f, "name": os.path.basename(f), "dir": os.path.dirname(f).replace(MODELS_DIR, "") or "/"}
        if any(x in name for x in ["mmproj", "clip", "vision", "projector"]):
            projectors.append(item)
        else:
            models.append(item)

    vision_options = '<option value="" class="bg-slate-900 italic">Auto-detectar / Nenhum</option>'
    for p in projectors:
        vision_options += f'<option value="{p["path"]}" class="bg-slate-900">{p["name"]}</option>'

    gpus = get_gpu_info()
    config = load_config()
    default_model = config.get("default_model")
    model_configs = config.get("model_configs", {})
    
    # Verifica se há um modelo rodando para pré-popular o UI
    status = find_llama_server()
    running_config = None
    if status.get("running") and last_start_request:
        running_config = {
            "path": last_start_request.path,
            "context_size": last_start_request.context_size,
            "gpu_weights": {w.index: w for w in last_start_request.gpu_weights},
            "mmproj_path": last_start_request.mmproj_path
        }

    model_items = ""
    for m in models:
        m_path = m["path"]
        m_js = m_path.replace("\\", "/")
        m_name = m["name"]
        m_dir = m["dir"]
        is_default = "checked" if m_path == default_model else ""
        
        # ID estável baseado no path (deve bater com o do /models)
        stable_id = f"model-item-{abs(sum(ord(c) << (i % 8) for i, c in enumerate(m_path))) % 1000000}"

        # Cache local para o JS inicial
        initial_cfg_js = ""
        if m_path in model_configs:
            initial_cfg_js = f"<script>window.modelConfigs['{m_js}'] = {json.dumps(model_configs[m_path])};</script>"

        # Se houver config salva, adiciona uma indicação visual
        has_config = "text-blue-400" if m_path in model_configs else "text-slate-100"
        
        model_items += f"""
        {initial_cfg_js}
        <div id="{stable_id}" class="model-item-container group flex items-center justify-between p-4 mb-3 bg-slate-800/40 backdrop-blur-md rounded-2xl hover:bg-slate-700/60 transition-all duration-300 border border-slate-700/50 hover:border-blue-500/50 shadow-lg" data-path="{m_js}">
            <div class="flex-1 min-w-0 mr-4 cursor-pointer" onclick="selectModel('{m_js}', '{stable_id}')" title="Clique para selecionar e carregar configurações">
                <div class="flex items-center gap-2 mb-1">
                    <i class="fas fa-cube text-blue-400 text-[10px]"></i>
                    <p class="model-name text-sm font-bold {has_config} break-all line-clamp-2">{m_name}</p>
                    { '<i class="fas fa-history text-[8px] text-blue-500/50 history-icon" title="Configuração salva disponível"></i>' if m_path in model_configs else '' }
                </div>
                <p class="text-[9px] text-slate-500 truncate uppercase tracking-tighter font-mono">{m_dir}</p>
            </div>
            <div class="flex items-center gap-3 md:gap-4">
                <div class="flex items-center gap-1">
                    <button onclick="renameModel('{m_js}')" class="rename-btn w-8 h-8 flex items-center justify-center rounded-lg hover:bg-blue-500/20 text-slate-600 hover:text-blue-500 transition-all" title="Renomear Modelo">
                        <i class="fas fa-edit text-[10px]"></i>
                    </button>
                    <button onclick="deleteModel('{m_js}')" class="delete-btn w-8 h-8 flex items-center justify-center rounded-lg hover:bg-red-500/20 text-slate-600 hover:text-red-500 transition-all" title="Excluir Modelo">
                        <i class="fas fa-trash-alt text-[10px]"></i>
                    </button>
                </div>
                <div class="flex flex-col items-center gap-1">
                    <span class="text-[8px] font-black text-slate-600 uppercase tracking-tighter">Padrão</span>
                    <input type="checkbox" class="model-default-checkbox w-4 h-4 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer" 
                           {is_default} onclick="setDefaultModel(this, '{m_js}')">
                </div>
                <div class="action-btn-container">
                    <button onclick="startModel('{m_js}', '{stable_id}')" class="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black rounded-xl active:scale-95 flex items-center gap-2 uppercase tracking-widest shadow-xl">
                        <i class="fas fa-play text-[8px]"></i> <span class="hidden sm:inline">CARREGAR</span><span class="sm:hidden">LOAD</span>
                    </button>
                </div>
            </div>
        </div>
        """
    
    gpu_rows = ""
    max_vram = max(g['vram'] for g in gpus) if gpus else 0
    main_gpu_idx = next((g['index'] for g in gpus if g['vram'] == max_vram), -1)
    
    for g in gpus:
        idx = g['index']
        if running_config and idx in running_config["gpu_weights"]:
            w_obj = running_config["gpu_weights"][idx]
            is_checked = "checked" if w_obj.active else ""
            weight_val = int(w_obj.weight)
        else:
            is_checked = "checked"
            weight_val = 100 if idx == main_gpu_idx else 0
            
        gpu_rows += f"""
        <tr class="gpu-row group border-b border-slate-800/50" data-index="{idx}">
            <td class="px-3 md:px-6 py-4 md:py-6 text-center">
                <div class="flex flex-col items-center gap-2">
                    <span class="gpu-util-val text-xs font-black text-blue-400 font-mono">0%</span>
                    <div class="w-12 h-1 bg-slate-800 rounded-full overflow-hidden"><div class="gpu-util-bar h-full bg-blue-500 transition-all duration-1000" style="width: 0%"></div></div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="flex items-center gap-2 md:gap-4">
                    <input type="checkbox" {is_checked} class="gpu-checkbox w-5 h-5 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer">
                    <div class="flex flex-col"><span class="text-[9px] font-black text-blue-400 uppercase tracking-widest mb-0.5">ID {idx}</span><span class="text-sm font-bold text-slate-100 whitespace-nowrap">{g['name']}</span></div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="flex flex-col md:flex-row gap-2 md:gap-6">
                    <div class="flex flex-col"><span class="text-[8px] font-black text-slate-500 uppercase mb-0.5">Temp</span><span class="gpu-temp-val text-xs font-bold text-slate-300 font-mono">--°C</span></div>
                    <div class="flex flex-col"><span class="text-[8px] font-black text-slate-500 uppercase mb-0.5">Power</span><span class="gpu-power-val text-xs font-bold text-slate-300 font-mono">--W</span></div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="flex flex-col gap-2 min-w-[100px] md:min-w-[160px]">
                    <div class="flex justify-between items-end"><span class="text-[8px] font-black text-slate-500 uppercase">VRAM</span><span class="gpu-vram-text text-[9px] font-mono text-blue-400">0 / {g['vram']} MB</span></div>
                    <div class="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden"><div class="gpu-vram-bar h-full bg-cyan-500 transition-all duration-1000" style="width: 0%"></div></div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="relative">
                    <input type="number" value="{weight_val}" min="0" max="100" class="gpu-weight w-24 pl-2 md:pl-4 pr-7 md:pr-9 py-2 bg-slate-900/80 border border-slate-700 rounded-xl text-sm font-black text-blue-400 outline-none transition-all" oninput="balanceWeights(this)">
                    <span class="absolute right-2 md:right-3 top-1/2 -translate-y-1/2 text-[10px] font-black text-slate-600">%</span>
                </div>
            </td>
        </tr>
        """
    
    # Prepara as opções de Vision com a seleção atual se rodando
    vision_options = '<option value="" class="bg-slate-900 italic">Auto-detectar / Nenhum</option>'
    for p in projectors:
        selected = 'selected' if running_config and running_config["mmproj_path"] == p["path"] else ''
        vision_options += f'<option value="{p["path"]}" class="bg-slate-900" {selected}>{p["name"]}</option>'
    
    # Contexto pré-selecionado
    ctx_2k = 'selected' if running_config and running_config["context_size"] == 2048 else ''
    ctx_4k = 'selected' if running_config and running_config["context_size"] == 4096 else ''
    ctx_8k = 'selected' if running_config and running_config["context_size"] == 8192 else ''
    ctx_16k = 'selected' if running_config and running_config["context_size"] == 16384 else ''
    ctx_32k = 'selected' if running_config and running_config["context_size"] == 32768 else ''
    ctx_64k = 'selected' if running_config and running_config["context_size"] == 65536 else ''
    ctx_128k = 'selected' if (running_config and running_config["context_size"] == 131072) or not running_config else ''
    ctx_256k = 'selected' if running_config and running_config["context_size"] == 262144 else ''
    ctx_512k = 'selected' if running_config and running_config["context_size"] == 524288 else ''
    ctx_1m = 'selected' if running_config and running_config["context_size"] == 1048576 else ''
    
    html_template = """
    <!DOCTYPE html>
    <html lang="pt-BR" class="dark">
    <head>
        <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Automanager Llama.cpp | Interface de IA</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
            :root { --bg-deep: #020617; --card-bg: rgba(15, 23, 42, 0.6); }
            body { font-family: 'Space Grotesk', sans-serif; background: radial-gradient(circle at 50% 0%, #1e3a8a 0%, #020617 100%); background-attachment: fixed; }
            .font-mono { font-family: 'JetBrains Mono', monospace; }
            .glass { background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); }
            .custom-scroll::-webkit-scrollbar { width: 6px; }
            .custom-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
            @keyframes pulse-glow { 0% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(37, 99, 235, 0); } 100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); } }
            .glow-online { animation: pulse-glow 2s infinite; }
            .terminal-line { animation: fadeIn 0.3s ease-out; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
            .model-item-container.active-selection { border-color: rgba(59, 130, 246, 0.8) !important; background-color: rgba(30, 41, 59, 0.8) !important; box-shadow: 0 0 15px rgba(59, 130, 246, 0.3); }
            .model-item-container.running-now { border-color: rgba(16, 185, 129, 0.5) !important; background-color: rgba(6, 78, 59, 0.2) !important; }
        </style>
    </head>
    <body class="min-h-screen text-slate-200 pb-16 selection:bg-blue-500/30">
        <div class="max-w-[1800px] mx-auto px-4 md:px-8 pt-6 md:pt-10">
            <header class="flex flex-col md:flex-row items-center justify-between mb-8 md:mb-10 glass p-4 md:p-5 rounded-3xl md:rounded-[2rem] gap-4">
                <div class="flex items-center gap-4 md:gap-6">
                    <div class="bg-blue-600 p-3 rounded-2xl shadow-xl shadow-blue-500/20"><i class="fas fa-brain text-white text-xl md:text-2xl"></i></div>
                    <div>
                        <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2 md:gap-3">Automanager <span class="text-blue-500 font-light">Llama.cpp</span></h1>
                        <p class="text-[10px] text-slate-500 font-mono tracking-wider uppercase">Interface Avançada de Computação Neural</p>
                    </div>
                </div>
                <div class="flex items-center gap-4 md:gap-8 w-full md:auto justify-center md:justify-end">
                    <div class="hidden sm:flex items-center gap-3 md:gap-5 px-4 md:px-6 py-2 bg-slate-900/50 rounded-xl border border-slate-800">
                        <div class="flex flex-col items-end"><span class="text-[9px] md:text-[10px] text-slate-500 font-black uppercase tracking-tighter">IP do Motor</span><span id="display-ip" class="text-xs font-mono text-blue-400">#FIXED_IP#</span></div>
                        <i class="fas fa-network-wired text-slate-600 text-sm md:text-base"></i>
                    </div>
                    <div id="status-badge" class="px-6 md:px-8 py-2 md:py-2.5 rounded-xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 glass border-slate-700/50 text-slate-500 uppercase transition-all duration-500"><div class="w-2 h-2 rounded-full bg-slate-600"></div>OFFLINE</div>
                </div>
            </header>
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 md:gap-10">
                <div class="lg:col-span-7 space-y-6 md:space-y-10">
                    <div class="grid grid-cols-2 md:grid-cols-2 gap-4 md:gap-6">
                        <div class="glass p-5 rounded-[1.5rem] border-l-4 border-blue-600">
                            <div class="flex justify-between items-start mb-4"><p class="text-[10px] font-black text-slate-500 uppercase tracking-widest font-mono">Processador (Host)</p><i class="fas fa-microchip text-slate-700 text-xs"></i></div>
                            <div class="flex items-end justify-between gap-4"><h3 id="cpu-val" class="text-3xl font-bold text-white leading-none">0%</h3><div class="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden"><div id="cpu-bar" class="h-full bg-blue-500 transition-all duration-700 shadow-[0_0_10px_rgba(37,99,235,0.5)]" style="width: 0%"></div></div></div>
                        </div>
                        <div class="glass p-5 rounded-[1.5rem] border-l-4 border-emerald-600">
                            <div class="flex justify-between items-start mb-4"><p class="text-[10px] font-black text-slate-500 uppercase tracking-widest font-mono">Memória RAM (Host)</p><i class="fas fa-memory text-slate-700 text-xs"></i></div>
                            <div class="flex items-end justify-between gap-4"><h3 id="ram-val" class="text-3xl font-bold text-white leading-none">0%</h3><div class="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden"><div id="ram-bar" class="h-full bg-emerald-500 transition-all duration-700" style="width: 0%"></div></div></div>
                        </div>
                    </div>
                    <div class="glass rounded-[2rem] overflow-hidden p-6 md:p-8">
                        <div class="flex flex-col md:flex-row md:items-center justify-between gap-6 md:gap-8 mb-8 border-b border-slate-800/50 pb-6">
                             <div><h3 class="font-bold text-lg text-white flex items-center gap-3"><i class="fas fa-microchip text-blue-500"></i>Recursos de GPU & Configuração</h3><p class="text-xs text-slate-500 mt-1 font-medium italic">Monitore e distribua a carga de processamento entre as GPUs</p></div>
                             <div class="flex flex-wrap items-center gap-4 md:gap-6 bg-slate-900/80 p-1.5 rounded-2xl border border-slate-800">
                                <div class="flex items-center gap-2">
                                    <label class="text-[9px] font-black uppercase text-slate-400 pl-3 md:pl-4 tracking-widest whitespace-nowrap">Contexto:</label>
                                    <select id="context-size" class="bg-blue-600/20 border border-blue-500/30 text-blue-300 rounded-xl px-4 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all cursor-pointer">
                                        <option value="2048" class="bg-slate-900" {ctx_2k}>2K</option>
                                        <option value="4096" class="bg-slate-900" {ctx_4k}>4K</option>
                                        <option value="8192" class="bg-slate-900" {ctx_8k}>8K</option>
                                        <option value="16384" class="bg-slate-900" {ctx_16k}>16K</option>
                                        <option value="32768" class="bg-slate-900" {ctx_32k}>32K</option>
                                        <option value="65536" class="bg-slate-900" {ctx_64k}>64K</option>
                                        <option value="131072" class="bg-slate-900" {ctx_128k}>128K</option>
                                        <option value="262144" class="bg-slate-900" {ctx_256k}>256K</option>
                                        <option value="524288" class="bg-slate-900" {ctx_512k}>512K</option>
                                        <option value="1048576" class="bg-slate-900" {ctx_1m}>1M</option>
                                    </select>
                                </div>
                                <div class="flex items-center gap-2 border-l border-slate-800 pl-4 md:pl-6">
                                    <label class="text-[9px] font-black uppercase text-slate-400 tracking-widest whitespace-nowrap"><i class="fas fa-eye text-blue-400 mr-2"></i>Vision:</label>
                                    <select id="mmproj-path" class="bg-slate-800 border border-slate-700 text-slate-300 rounded-xl px-4 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all cursor-pointer max-w-[200px]">
                                        #VISION_OPTIONS#
                                    </select>
                                </div>
                             </div>
                        </div>
                        <div class="overflow-x-auto"><table class="w-full text-left"><thead class="text-[9px] md:text-[10px] font-black text-slate-500 uppercase tracking-widest"><tr><th class="px-4 md:px-6 py-4 text-center">Uso</th><th class="px-4 py-4">Dispositivo</th><th class="px-4 py-4">Monitoramento</th><th class="px-4 py-4">VRAM Status</th><th class="px-4 py-4">Distribuição</th></tr></thead><tbody id="gpu-table-body" class="divide-y divide-slate-800/50">#GPU_ROWS#</tbody></table></div>
                        <div class="flex flex-col sm:flex-row justify-between items-center pt-8 gap-4"><div class="flex items-center gap-3 text-[10px] md:text-xs text-slate-500"><i class="fas fa-info-circle text-blue-500"></i>Distribua 100% da carga total entre as GPUs selecionadas</div><span id="total-percent" class="text-xs md:text-sm font-black tracking-widest px-4 py-2 rounded-xl transition-all duration-300">CARGA TOTAL: 100%</span></div>
                    </div>
                    <div id="active-card" class="bg-gradient-to-r from-blue-900/40 to-slate-900/40 backdrop-blur-xl p-6 md:p-10 rounded-[2rem] md:rounded-[2.5rem] border border-blue-500/30 hidden transition-all duration-700 animate-in fade-in zoom-in">
                        <div class="flex flex-col lg:flex-row items-center justify-between gap-8 md:gap-10">
                            <div class="flex items-center gap-5 md:gap-8 w-full">
                                <div class="w-16 h-16 rounded-3xl bg-blue-600 flex items-center justify-center text-white shadow-2xl shadow-blue-500/40 shrink-0"><i class="fas fa-robot text-2xl md:text-3xl"></i></div>
                                <div class="min-w-0">
                                    <p class="text-blue-400 text-[10px] font-black uppercase tracking-[0.3em] mb-2 font-mono">Motor de Computação Primário</p>
                                    <h2 id="active-model-name" class="text-xl md:text-2xl font-bold text-white truncate max-w-[200px] sm:max-w-md">--</h2>
                                    <div class="flex gap-4 mt-3"><div class="flex items-center gap-2 text-[10px] font-mono text-slate-400"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>Ativo há: <span id="uptime-val">Calculando...</span></div></div>
                                </div>
                            </div>
                            <div class="flex flex-col sm:flex-row gap-4 md:gap-6 w-full lg:w-auto">
                                <a id="chat-link" href="#" target="_blank" class="px-6 md:px-10 py-4 bg-blue-600 hover:bg-blue-500 text-white rounded-2xl text-[10px] md:text-xs font-black transition-all shadow-xl shadow-blue-600/30 active:scale-95 flex items-center justify-center gap-3 md:gap-4 uppercase tracking-widest whitespace-nowrap">
                                    <i class="fas fa-comments text-sm"></i> ABRIR CHAT
                                </a>
                                <button onclick="stopModel()" class="px-6 md:px-10 py-4 bg-red-600/10 hover:bg-red-600/20 text-red-500 border border-red-500/30 rounded-2xl text-[10px] md:text-xs font-black transition-all active:scale-95 uppercase tracking-widest whitespace-nowrap">
                                    ENCERRAR
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="glass rounded-[2rem] overflow-hidden shadow-2xl border border-slate-800">
                        <div class="px-6 md:px-10 py-4 bg-slate-900/60 border-b border-slate-800 flex justify-between items-center">
                            <div class="flex items-center gap-3 md:gap-4">
                                <div class="flex gap-1.5 md:gap-2"><div class="w-2 h-2 rounded-full bg-slate-700"></div><div class="w-2 h-2 rounded-full bg-slate-700"></div><div class="w-2 h-2 rounded-full bg-slate-700"></div></div>
                                <p class="text-slate-400 text-[9px] md:text-[10px] font-black uppercase tracking-widest font-mono ml-2 md:ml-4">Saída de logs do sistema</p>
                            </div>
                            <button onclick="document.getElementById('log-box').innerHTML=''" class="text-[9px] md:text-[10px] text-slate-600 hover:text-blue-400 font-bold uppercase transition-colors tracking-widest"><i class="fas fa-trash-alt mr-2"></i> Limpar</button>
                        </div>
                        <div id="log-box" class="custom-scroll p-6 md:p-10 h-[300px] md:h-[400px] overflow-y-auto font-mono text-[10px] md:text-xs text-slate-400 leading-relaxed whitespace-pre-wrap bg-slate-950/40"></div>
                    </div>
                </div>
                <div class="lg:col-span-5 space-y-6 md:space-y-10">
                    <div class="glass rounded-[2rem] border border-slate-800 flex flex-col h-auto md:h-[900px]">
                        <div class="p-8 border-b border-slate-800/50 flex items-center justify-between"><div class="flex items-center gap-4 md:gap-5"><div class="w-10 h-10 bg-slate-800 rounded-xl flex items-center justify-center border border-slate-700"><i class="fas fa-database text-amber-500 text-sm md:text-base"></i></div><h3 class="font-bold text-base md:text-lg text-white tracking-tight">Model Repository</h3></div><span class="text-[10px] bg-slate-800 text-slate-400 px-3 py-1 rounded-full font-mono border border-slate-700" id="model-count">0 UNIDADES</span></div>
                        <div class="p-8 border-b border-slate-800/30 bg-blue-600/5"><p class="text-[10px] font-black text-slate-500 uppercase mb-4 md:mb-6 tracking-widest">Ingerir GGUF via URL</p><div class="space-y-3 md:space-y-4"><div class="relative group"><i class="fas fa-link absolute left-4 top-1/2 -translate-y-1/2 text-slate-600 text-xs md:text-sm transition-colors group-focus-within:text-blue-500"></i><input type="text" id="download-url" placeholder="https://..." class="w-full pl-10 pr-4 py-3 bg-slate-900 border border-slate-700 rounded-2xl text-xs text-slate-200 focus:ring-2 focus:ring-blue-500/50 outline-none transition-all placeholder:text-slate-600"></div><button onclick="downloadModel()" class="w-full py-4 bg-slate-100 hover:bg-white text-slate-950 text-[10px] font-black rounded-2xl transition-all shadow-xl active:scale-[0.98] uppercase tracking-[0.2em] flex items-center justify-center gap-3 md:gap-4"><i class="fas fa-cloud-download-alt text-sm"></i> EXECUTAR DOWNLOAD</button></div><div id="download-status" class="mt-6 md:mt-8 space-y-3"></div></div>
                        <div id="model-list-container" class="p-6 flex-1 overflow-y-auto custom-scroll space-y-2">#MODEL_ITEMS#</div>
                        <div class="p-8 bg-slate-950/40 border-t border-slate-800 rounded-b-[2rem] md:rounded-b-[2.5rem]"><div class="flex flex-col gap-3"><div class="flex items-center justify-between"><p class="text-[9px] text-slate-500 font-black uppercase tracking-widest">Interface de API</p><span class="text-[8px] bg-emerald-500/10 text-emerald-500 px-2 py-0.5 rounded border border-emerald-500/20 uppercase font-black">Ativo</span></div><div class="flex items-center gap-3 md:gap-4 bg-slate-900 p-3 rounded-xl border border-slate-800 group"><code id="api-link" class="text-[10px] text-blue-400 font-mono flex-1 truncate"></code><button onclick="navigator.clipboard.writeText(document.getElementById('api-link').innerText)" class="text-slate-600 hover:text-white transition-colors"><i class="far fa-copy"></i></button></div></div></div>
                    </div>
                </div>
            </div>
        </div>
        <script>
            let logStream = null; let startTime = null; const fixedIp = "#FIXED_IP#";
            window.modelConfigs = window.modelConfigs || {}; 
            let currentSelectedModel = null;
            let currentRunningModelPath = null;

            document.getElementById('chat-link').href = `http://${fixedIp}:8085/`;
            document.getElementById('api-link').innerText = `http://${fixedIp}:8085/v1`;

            function resetToDefaults() {
                document.getElementById('context-size').value = "131072";
                document.getElementById('mmproj-path').value = "";
                document.querySelectorAll('.gpu-row').forEach((row, idx) => {
                    row.querySelector('.gpu-checkbox').checked = true;
                    // Tenta manter a lógica de maior VRAM ou apenas coloca 100 na primeira se não souber
                    row.querySelector('.gpu-weight').value = (idx === 0 ? "100" : "0");
                });
                updateTotal();
            }

            function selectModel(path, elementId) {
                currentSelectedModel = path;
                document.querySelectorAll('.model-item-container').forEach(el => {
                    el.classList.remove('active-selection');
                });
                const selectedEl = document.getElementById(elementId);
                if (selectedEl) selectedEl.classList.add('active-selection');
                
                if (window.modelConfigs[path]) {
                    applyModelConfig(path);
                } else {
                    resetToDefaults();
                }
            }

            function applyModelConfig(path) {
                const cfg = window.modelConfigs[path];
                if (!cfg) return;
                
                console.log("Aplicando config para:", path, cfg);

                // Aplica contexto
                if (cfg.context_size) document.getElementById('context-size').value = cfg.context_size;
                
                // Aplica mmproj
                if (cfg.mmproj_path !== undefined) {
                    const select = document.getElementById('mmproj-path');
                    // Verifica se a opção existe, senão adiciona temporariamente ou espera o refresh
                    let found = false;
                    for (let i = 0; i < select.options.length; i++) {
                        if (select.options[i].value === cfg.mmproj_path) {
                            select.value = cfg.mmproj_path;
                            found = true;
                            break;
                        }
                    }
                    if (!found && cfg.mmproj_path) {
                        const opt = document.createElement('option');
                        opt.value = cfg.mmproj_path;
                        opt.text = cfg.mmproj_path.split('/').pop() + " (Salvo)";
                        select.add(opt);
                        select.value = cfg.mmproj_path;
                    } else if (!cfg.mmproj_path) {
                        select.value = "";
                    }
                }
                
                // Aplica pesos e checkboxes
                if (cfg.gpu_weights) {
                    cfg.gpu_weights.forEach(w => {
                        const row = document.querySelector(`.gpu-row[data-index="${w.index}"]`);
                        if (row) {
                            const cb = row.querySelector('.gpu-checkbox');
                            const input = row.querySelector('.gpu-weight');
                            cb.checked = w.active !== undefined ? w.active : (w.weight > 0);
                            input.value = Math.round(w.weight);
                        }
                    });
                    updateTotal();
                }
                
                // Feedback visual na lista
                const nameEl = document.querySelector(`[data-path="${path}"] .model-name`);
                if (nameEl) {
                    nameEl.classList.add('text-emerald-400');
                    setTimeout(() => { nameEl.classList.remove('text-emerald-400'); }, 1000);
                }
            }

            function balanceWeights(changedInput) {
                const weights = Array.from(document.querySelectorAll('.gpu-weight'));
                const checkedWeights = weights.filter(w => w.closest('.gpu-row').querySelector('.gpu-checkbox').checked);
                if (checkedWeights.length <= 1) { if (checkedWeights.length === 1) checkedWeights[0].value = 100; updateTotal(); return; }
                let val = parseInt(changedInput.value) || 0;
                if (val > 100) { val = 100; changedInput.value = 100; }
                if (val < 0) { val = 0; changedInput.value = 0; }
                const otherInputs = checkedWeights.filter(w => w !== changedInput);
                let remaining = 100 - val;
                for (let i = 0; i < otherInputs.length; i++) {
                    if (i === otherInputs.length - 1) { otherInputs[i].value = Math.max(0, remaining); } 
                    else { let share = Math.min(remaining, Math.round(remaining / otherInputs.length)); otherInputs[i].value = share; remaining -= share; }
                }
                updateTotal();
            }

            function updateTotal() { 
                let sum = 0; 
                document.querySelectorAll('.gpu-weight').forEach(i => {
                    const isChecked = i.closest('.gpu-row').querySelector('.gpu-checkbox').checked;
                    if (isChecked) sum += parseInt(i.value || 0); else i.value = 0;
                }); 
                const badge = document.getElementById('total-percent'); badge.innerText = `CARGA TOTAL: ${sum}%`; 
                badge.className = sum === 100 ? 'text-sm font-black tracking-widest px-4 md:px-6 py-2.5 md:py-3 rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20' : 'text-sm font-black tracking-widest px-4 md:px-6 py-2.5 md:py-3 rounded-xl bg-red-500/10 text-red-500 border border-red-500/20'; 
            }

            async function updateMetrics() {
                try {
                    const res = await fetch('/metrics'); const data = await res.json();
                    document.getElementById('cpu-val').innerText = data.cpu + '%'; document.getElementById('cpu-bar').style.width = data.cpu + '%';
                    document.getElementById('ram-val').innerText = data.ram + '%'; document.getElementById('ram-bar').style.width = data.ram + '%';
                    data.gpus.forEach(g => {
                        const row = document.querySelector(`.gpu-row[data-index="${g.index}"]`);
                        if (row) {
                            row.querySelector('.gpu-util-val').innerText = g.util + '%'; row.querySelector('.gpu-util-bar').style.width = g.util + '%';
                            row.querySelector('.gpu-temp-val').innerText = (g.temp || '--') + '°C'; row.querySelector('.gpu-power-val').innerText = (g.power || '--') + 'W';
                            row.querySelector('.gpu-vram-text').innerText = `${g.mem_used} / ${g.mem_total} MB`; row.querySelector('.gpu-vram-bar').style.width = g.vram_pct + '%';
                        }
                    });
                } catch(e) {}
            }

            async function startLogs() {
                if (logStream) logStream.abort(); logStream = new AbortController(); const box = document.getElementById('log-box');
                box.innerHTML = '';
                try {
                    const response = await fetch('/logs', { signal: logStream.signal }); const reader = response.body.getReader(); const decoder = new TextDecoder();
                    while (true) {
                        const { value, done } = await reader.read(); if (done) break;
                        const formatted = decoder.decode(value).replace(/error/gi, '<span class="text-red-500 font-black px-1 rounded bg-red-500/10">ERRO</span>').replace(/warn/gi, '<span class="text-amber-500 font-black px-1 rounded bg-amber-500/10">AVISO</span>').replace(/info/gi, '<span class="text-blue-400 font-bold uppercase tracking-tighter">info</span>');
                        const line = document.createElement('div'); line.className = 'terminal-line mb-1 md:mb-2 border-l border-slate-800 md:border-l-2 pl-3 md:pl-4'; line.innerHTML = formatted; box.appendChild(line); box.scrollTop = box.scrollHeight; if (box.childNodes.length > 500) box.removeChild(box.firstChild);
                    }
                } catch(e) {}
            }

            function updateUptime(serverStartTime) { 
                let diff;
                if (serverStartTime) {
                    diff = Math.floor(Date.now() / 1000 - serverStartTime);
                } else if (startTime) {
                    diff = Math.floor((new Date() - startTime) / 1000);
                } else {
                    return;
                }
                document.getElementById('uptime-val').innerText = `${Math.floor(diff/3600)}h ${Math.floor((diff%3600)/60)}m ${diff%60}s`; 
            }
            
            async function updateStatus() {
                try {
                    const res = await fetch('/status'); 
                    const data = await res.json(); 
                    const badge = document.getElementById('status-badge'); 
                    const card = document.getElementById('active-card');
                    
                    if (data.config && data.config.gpu_weights && (!data.recovery || !data.recovery.active)) {
                        data.config.gpu_weights.forEach(w => {
                            const row = document.querySelector(`.gpu-row[data-index="${w.index}"]`);
                            if (row) {
                                const input = row.querySelector('.gpu-weight');
                                const cb = row.querySelector('.gpu-checkbox');
                                if (document.activeElement !== input) {
                                    const newWeight = Math.round(w.weight);
                                    if (parseInt(input.value) !== newWeight) {
                                        input.value = newWeight;
                                    }
                                }
                                if (w.active !== undefined) cb.checked = w.active;
                            }
                        });
                        updateTotal();
                        // Também restaura mmproj e context se estiver rodando e nada estiver selecionado
                        if (data.running && !currentSelectedModel) {
                            if (data.config.context_size) document.getElementById('context-size').value = data.config.context_size;
                            if (data.config.mmproj_path !== undefined) document.getElementById('mmproj-path').value = data.config.mmproj_path || "";
                        }
                    }

                    if (data.recovery && data.recovery.failed) {
                        badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-red-500/50 text-red-500 uppercase';
                        badge.innerHTML = `<i class="fas fa-exclamation-triangle mr-1"></i> FALHA: ${data.recovery.message.toUpperCase()}`;
                        return;
                    }

                    if (data.recovery && data.recovery.active) {
                        badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-amber-500/50 text-amber-500 uppercase';
                        badge.innerHTML = '<i class="fas fa-sync animate-spin mr-1"></i> REALOCANDO...';
                        return;
                    }

                    if (data.running) {
                        badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-emerald-500/30 text-emerald-500 uppercase glow-online'; 
                        badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-emerald-500 animate-pulse"></div> ONLINE'; 
                        card.classList.remove('hidden'); 
                        document.getElementById('active-model-name').innerText = data.model; 
                        if (!logStream) startLogs(); 
                        updateUptime(data.start_time);
                        currentRunningModelPath = data.model_path;
                        
                        // Marca seleção na lista se rodando
                        if (!currentSelectedModel && currentRunningModelPath) {
                            currentSelectedModel = currentRunningModelPath.replace(/\\\\/g, '/');
                        }
                    } else {
                        startTime = null; 
                        badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-slate-700/50 text-slate-500 uppercase'; 
                        badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-slate-600"></div> OFFLINE'; 
                        card.classList.add('hidden'); 
                        if (logStream) { logStream.abort(); logStream = null; }
                        currentRunningModelPath = null;
                    }
                    
                    // Sincroniza visual da lista de modelos
                    document.querySelectorAll('.model-item-container').forEach(el => {
                        const m_js = el.dataset.path;
                        const actionBtnContainer = el.querySelector('.action-btn-container');
                        const renameBtn = el.querySelector('.rename-btn');
                        const deleteBtn = el.querySelector('.delete-btn');
                        
                        // Normaliza para comparação robusta
                        const normalizedM = m_js.replace(/\\\\/g, '/');
                        const normalizedR = currentRunningModelPath ? currentRunningModelPath.replace(/\\\\/g, '/') : null;
                        const isRunning = normalizedR && normalizedM === normalizedR;
                        
                        if (isRunning) {
                            el.classList.add('running-now');
                            if (renameBtn) renameBtn.classList.add('hidden');
                            if (deleteBtn) deleteBtn.classList.add('hidden');
                        } else {
                            el.classList.remove('running-now');
                            if (renameBtn) renameBtn.classList.remove('hidden');
                            if (deleteBtn) deleteBtn.classList.remove('hidden');
                        }
                        
                        if (currentSelectedModel === m_js) {
                            el.classList.add('active-selection');
                        } else {
                            el.classList.remove('active-selection');
                        }
                        
                        const newButtonsHtml = getModelButtonsHtml(m_js, el.id, isRunning);
                        if (actionBtnContainer.innerHTML.trim() !== newButtonsHtml.trim()) {
                            actionBtnContainer.innerHTML = newButtonsHtml;
                        }
                    });
                } catch(e) { console.error("Error in updateStatus:", e); }
            }

            async function setDefaultModel(checkbox, path) { if (checkbox.checked) document.querySelectorAll('.model-default-checkbox').forEach(cb => { if (cb !== checkbox) cb.checked = false; }); try { await fetch('/set_default', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ path: checkbox.checked ? path : null }) }); } catch(e) { alert("Erro ao salvar configuração."); } }
            
            async function downloadModel() { const url = document.getElementById('download-url').value.trim(); if (!url) return; try { const res = await fetch('/download', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ url }) }); if (res.ok) { document.getElementById('download-url').value = ''; updateDownloads(); } } catch(e) {} }
            
            async function updateDownloads() {
                try {
                    const res = await fetch('/downloads'); const data = await res.json(); const container = document.getElementById('download-status'); const entries = Object.entries(data); if (entries.length === 0) { container.innerHTML = ''; return; }
                    container.innerHTML = entries.map(([id, d]) => `<div class="p-4 md:p-5 bg-slate-900 border border-slate-800 rounded-2xl"><div class="flex justify-between items-center mb-3 md:mb-4"><p class="text-xs md:text-sm font-bold truncate flex-1 mr-3 md:mr-4 text-slate-300 font-mono" title="${d.filename}">${d.filename}</p><span class="text-[8px] md:text-[10px] font-black uppercase px-2 md:px-3 py-0.5 md:py-1 rounded ${d.status === 'completed' ? 'bg-emerald-500/10 text-emerald-500' : d.status === 'failed' ? 'bg-red-500/10 text-red-500' : 'bg-blue-500/10 text-blue-500'}">${d.status === 'completed' ? 'concluído' : d.status === 'failed' ? 'falhou' : 'baixando'}</span></div><div class="w-full h-1.5 md:h-2 bg-slate-800 rounded-full overflow-hidden"><div class="h-full bg-blue-500 shadow-[0_0_10px_rgba(37,99,235,0.5)] transition-all duration-500" style="width: ${d.progress}%"></div></div></div>`).join(''); if (entries.some(([_, d]) => d.status === 'completed')) updateModels();
                } catch(e) {}
            }

            async function updateModels() {
                try {
                    const [res, cfgRes] = await Promise.all([fetch('/models'), fetch('/config')]); const data = await res.json(); const cfg = await cfgRes.json(); 
                    document.getElementById('model-count').innerText = `${data.models.length} UNIDADES`;
                    
                    // Preserva a seleção atual se possível
                    const oldContainer = document.getElementById('model-list-container');
                    
                    const newHtml = data.models.map(m => {
                        const m_js = m.path.replace(/\\\\/g, '/');
                        if (m.last_config) window.modelConfigs[m.path] = m.last_config;
                        const hasConfigClass = m.last_config ? 'text-blue-400' : 'text-slate-100';
                        const historyIcon = m.last_config ? '<i class="fas fa-history text-[8px] text-blue-500/50" title="Configuração salva disponível"></i>' : '';
                        const isRunning = currentRunningModelPath && m_js === currentRunningModelPath.replace(/\\\\/g, '/');
                        const isActive = currentSelectedModel === m_js ? 'active-selection' : '';
                        const runningClass = isRunning ? 'running-now' : '';
                        const hashId = m.id;
                        const buttonsHtml = getModelButtonsHtml(m_js, hashId, isRunning);

                        return `<div id="${hashId}" class="model-item-container group flex items-center justify-between p-4 md:p-5 mb-3 md:mb-4 bg-slate-800/40 backdrop-blur-md rounded-2xl hover:bg-slate-700/60 transition-all duration-300 border border-slate-700/50 hover:border-blue-500/50 shadow-lg ${isActive} ${runningClass}" data-path="${m_js}">
                            <div class="flex-1 min-w-0 mr-4 md:mr-6 cursor-pointer" onclick="selectModel('${m_js}', '${hashId}')">
                                <div class="flex items-center gap-2 md:gap-3 mb-1 md:mb-2">
                                    <i class="fas fa-cube text-blue-400 text-[10px] md:text-xs"></i>
                                    <p class="model-name text-sm md:text-base font-bold ${hasConfigClass} break-all line-clamp-2" title="${m.name}">${m.name}</p>
                                    ${historyIcon}
                                </div>
                                <p class="text-[9px] md:text-xs text-slate-500 truncate uppercase tracking-tighter font-mono">${m.dir}</p>
                            </div>
                            <div class="flex items-center gap-3 md:gap-6">
                                <div class="flex items-center gap-1">
                                    <button onclick="renameModel('${m_js}')" class="rename-btn w-10 h-10 flex items-center justify-center rounded-xl hover:bg-blue-500/20 text-slate-600 hover:text-blue-500 transition-all ${isRunning ? 'hidden' : ''}" title="Renomear Modelo">
                                        <i class="fas fa-edit text-[10px] md:text-xs"></i>
                                    </button>
                                    <button onclick="deleteModel('${m_js}')" class="delete-btn w-10 h-10 flex items-center justify-center rounded-xl hover:bg-red-500/20 text-slate-600 hover:text-red-500 transition-all ${isRunning ? 'hidden' : ''}" title="Excluir Modelo">
                                        <i class="fas fa-trash-alt text-[10px] md:text-xs"></i>
                                    </button>
                                </div>
                                <div class="flex flex-col items-center gap-1 md:gap-1.5">
                                    <span class="text-[8px] md:text-[10px] font-black text-slate-600 uppercase tracking-tighter">Padrão</span>
                                    <input type="checkbox" class="model-default-checkbox w-4 h-4 md:w-5 md:h-5 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer" ${m.path === cfg.default_model ? 'checked' : ''} onclick="setDefaultModel(this, '${m_js}')">
                                </div>
                                <div class="action-btn-container">
                                    ${buttonsHtml}
                                </div>
                            </div>
                        </div>`;
                    }).join('');
                    
                    if (oldContainer.innerHTML !== newHtml) oldContainer.innerHTML = newHtml;

                    const projSelect = document.getElementById('mmproj-path');
                    const currentVal = projSelect.value;
                    let projHtml = '<option value="" class="bg-slate-900 italic">Auto-detectar / Nenhum</option>';
                    data.projectors.forEach(p => {
                        projHtml += `<option value="${p.path}" class="bg-slate-900">${p.name}</option>`;
                    });
                    
                    // Só atualiza se a lista de opções mudou
                    if (projSelect.innerHTML.trim() !== projHtml.trim()) {
                        projSelect.innerHTML = projHtml;
                        // Tenta restaurar o valor, mas se sumiu (ex: deletado), volta pro auto
                        projSelect.value = currentVal;
                        if (projSelect.value !== currentVal) projSelect.value = "";
                    }
                } catch(e) {}
            }

            async function renameModel(path) {
                const currentName = path.split('/').pop().replace('.gguf', '');
                const newName = prompt("Digite o novo nome para o modelo:", currentName);
                if (!newName || newName === currentName) return;
                
                try {
                    const res = await fetch('/rename', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ path, new_name: newName })
                    });
                    if (res.ok) { 
                        updateModels(); 
                    } else { 
                        const err = await res.json(); 
                        alert("Erro ao renomear: " + (err.detail || "Erro desconhecido")); 
                    }
                } catch(e) { alert("Erro de rede ao renomear modelo."); }
            }

            async function deleteModel(path) {
                if (!confirm("TEM CERTEZA QUE DESEJA EXCLUIR ESTE MODELO DO DISCO?\\nEsta ação é irreversível.")) return;
                try {
                    const res = await fetch('/delete', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ path })
                    });
                    if (res.ok) { updateModels(); } else { const err = await res.json(); alert("Erro ao excluir: " + (err.detail || "Erro desconhecido")); }
                } catch(e) { alert("Erro de rede ao excluir modelo."); }
            }

            async function startModel(path, elementId) { 
                // Se o modelo ainda não está selecionado no UI, seleciona ele agora (o que aplica a config salva)
                if (currentSelectedModel !== path) {
                    selectModel(path, elementId);
                    // Pequeno delay para garantir que os valores do UI foram atualizados antes de ler
                    await new Promise(r => setTimeout(r, 100));
                }

                const weights = []; 
                document.getElementById('log-box').innerHTML = '';
                document.querySelectorAll('.gpu-row').forEach(r => { 
                    const isChecked = r.querySelector('.gpu-checkbox').checked;
                    weights.push({ 
                        index: parseInt(r.dataset.index), 
                        weight: parseInt(r.querySelector('.gpu-weight').value || 0), 
                        name: "GPU",
                        active: isChecked
                    }); 
                }); 
                
                if (!weights.some(w => w.active)) return alert("SELECIONE PELO MENOS UMA GPU"); 
                
                const mmprojPath = document.getElementById('mmproj-path').value;
                document.getElementById('status-badge').innerHTML = '<i class="fas fa-circle-notch animate-spin mr-2 md:mr-3 text-sm md:text-lg"></i> INICIALIZANDO...'; 
                
                try {
                    await fetch('/start', { 
                        method: 'POST', 
                        headers: {'Content-Type': 'application/json'}, 
                        body: JSON.stringify({ 
                            path, 
                            mmproj_path: mmprojPath || null,
                            gpu_weights: weights, 
                            context_size: parseInt(document.getElementById('context-size').value) 
                        }) 
                    }); 
                } catch(e) {
                    alert("Erro ao iniciar modelo.");
                }
                
                setTimeout(updateStatus, 2000); 
            }

            async function stopModel() { if (confirm("ENCERRAR PROCESSO?")) { await fetch('/stop', {method: 'POST'}); setTimeout(updateStatus, 1000); } }
            
            setInterval(updateMetrics, 2000); 
            setInterval(updateStatus, 3000); 
            setInterval(updateDownloads, 3000);
            setInterval(updateModels, 5000); // Refresh model list periodically

            updateStatus(); updateMetrics(); updateDownloads(); updateModels(); updateTotal();
        </script>
    </body>
    </html>
    """
    local_ip = get_local_ip()
    final_html = html_template.replace("#GPU_ROWS#", gpu_rows).replace("#MODEL_ITEMS#", model_items).replace("#FIXED_IP#", local_ip).replace("#VISION_OPTIONS#", vision_options)
    for c in ["ctx_2k", "ctx_4k", "ctx_8k", "ctx_16k", "ctx_32k", "ctx_64k", "ctx_128k", "ctx_256k", "ctx_512k", "ctx_1m"]:

        final_html = final_html.replace(f"{{{c}}}", locals()[c])
    return final_html

@app.get("/config")
def get_config():
    return load_config()

@app.post("/set_default")
def set_default(req: SetDefaultRequest):
    config = load_config()
    config["default_model"] = req.path
    save_config(config)
    return {"status": "ok"}

@app.post("/delete")
def delete_model(req: DeleteRequest):
    if not req.path.startswith(MODELS_DIR):
        raise HTTPException(status_code=403, detail="Acesso negado")
    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    status = find_llama_server()
    if status["running"] and status["model"] == req.path:
        stop_model()
    try:
        os.remove(req.path)
        logging.info(f"Modelo excluído: {req.path}")
        return {"status": "deleted"}
    except Exception as e:
        logging.error(f"Erro ao excluir modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rename")
def rename_model(req: RenameRequest):
    if not req.path.startswith(MODELS_DIR):
        raise HTTPException(status_code=403, detail="Acesso negado")
    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    
    # Verifica se o modelo está rodando
    status = find_llama_server()
    if status["running"]:
        # Se o last_start_request bater ou se o path estiver no cmdline
        is_running = False
        if last_start_request and last_start_request.path == req.path:
            is_running = True
        else:
            # Fallback robusto via cmdline
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    if proc.info['pid'] == status['pid']:
                        if any(req.path in arg for i, arg in enumerate(proc.info['cmdline'] or [])):
                            is_running = True
                        break
                except: pass
        
        if is_running:
            raise HTTPException(status_code=400, detail="Não é possível renomear um modelo em execução")

    # Prepara novo path
    dir_name = os.path.dirname(req.path)
    new_filename = req.new_name
    if not new_filename.endswith(".gguf"):
        new_filename += ".gguf"
    new_path = os.path.join(dir_name, new_filename)

    if os.path.exists(new_path):
        raise HTTPException(status_code=400, detail="Já existe um arquivo com este nome")

    try:
        os.rename(req.path, new_path)
        
        # Atualiza config se necessário
        config = load_config()
        updated = False
        
        # Atualiza modelo padrão
        if config.get("default_model") == req.path:
            config["default_model"] = new_path
            updated = True
            
        # Atualiza configs de modelo
        if "model_configs" in config and req.path in config["model_configs"]:
            config["model_configs"][new_path] = config["model_configs"].pop(req.path)
            updated = True
            
        if updated:
            save_config(config)
            
        logging.info(f"Modelo renomeado: {req.path} -> {new_path}")
        return {"status": "renamed", "new_path": new_path}
    except Exception as e:
        logging.error(f"Erro ao renomear modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def get_status():
    status = find_llama_server()
    status["recovery"] = recovery_state
    
    if status.get("running"):
        if last_start_request:
            status["model_path"] = last_start_request.path
            status["config"] = {
                "path": last_start_request.path,
                "context_size": last_start_request.context_size,
                "gpu_weights": [w.model_dump() if hasattr(w, "model_dump") else w for w in last_start_request.gpu_weights],
                "mmproj_path": last_start_request.mmproj_path
            }
        else:
            # Tenta encontrar o path completo no cmdline do processo
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['pid'] == status['pid']:
                        cmdline = proc.info['cmdline']
                        for i in range(len(cmdline)-1):
                            if cmdline[i] in ["-m", "--model"]:
                                status["model_path"] = cmdline[i+1]
                                break
                        break
                except: pass
    
    return status

@app.get("/models")
def list_models():
    all_files = get_gguf_models()
    models = []
    projectors = []
    config = load_config()
    model_configs = config.get("model_configs", {})
    
    for f in all_files:
        name = os.path.basename(f).lower()
        # ID estável baseado no path
        m_id = f"model-item-{abs(sum(ord(c) << (i % 8) for i, c in enumerate(f))) % 1000000}"
        item = {
            "id": m_id,
            "path": f,
            "name": os.path.basename(f),
            "dir": os.path.dirname(f).replace(MODELS_DIR, "") or "/",
            "last_config": model_configs.get(f)
        }
        
        if any(x in name for x in ["mmproj", "clip", "vision", "projector"]):
            projectors.append(item)
        else:
            models.append(item)
    return {"models": models, "projectors": projectors}

@app.get("/downloads")
def get_downloads():
    return downloads

@app.post("/download")
def start_download(req: DownloadRequest, background_tasks: BackgroundTasks):
    download_id = str(uuid.uuid4())
    background_tasks.add_task(download_model_task, download_id, req.url, req.filename)
    return {"download_id": download_id}

@app.post("/start")
def start_model(req: StartRequest):
    global last_start_request, recovery_state
    recovery_state = {"active": False, "failed": False, "message": ""}
    last_start_request = req
    update_model_config(req.path, req.context_size, req.gpu_weights, req.mmproj_path)
    if execute_start(req):
        return {"message": "Started"}
    else:
        raise HTTPException(status_code=500, detail="Erro ao iniciar processo")

@app.post("/stop")
def stop_model():
    subprocess.run(["pkill", "-9", "llama-server"])
    return {"message": "Stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
