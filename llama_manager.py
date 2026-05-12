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
FIXED_IP = "192.168.2.183"
CONFIG_PATH = "/root/automanager_config.json"

class GPUWeight(BaseModel):
    index: int
    weight: float
    name: str

class StartRequest(BaseModel):
    path: str
    gpu_weights: List[GPUWeight]
    context_size: int = 131072

class DeleteRequest(BaseModel):
    path: str

class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None

class SetDefaultRequest(BaseModel):
    path: Optional[str] = None

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
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
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
                    return {"running": True, "pid": proc.info['pid'], "model": model_name}
        except (psutil.NoSuchProcess, psutil.AccessDenied): continue
    return {"running": False}

# Estado global para controle de auto-retry
last_start_request = None
recovery_state = {"active": False, "message": ""}
retry_lock = threading.Lock()

def monitor_oom():
    global last_start_request
    while True:
        try:
            if os.path.exists(SERVER_LOG_PATH):
                with open(SERVER_LOG_PATH, "r") as f:
                    # Vai para o final do arquivo e monitora novas linhas
                    f.seek(0, os.SEEK_END)
                    while True:
                        line = f.readline()
                        if not line:
                            if not find_llama_server()["running"] and not recovery_state["active"]:
                                break # Se parou de rodar e não está recuperando, sai do loop interno
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
        recovery_state["message"] = "OOM detectado. Reajustando pesos..."
        
        req = last_start_request
        weights = req.gpu_weights
        if len(weights) <= 1:
            logging.error("OOM em single GPU ou sem pesos. Impossível ajustar.")
            recovery_state["active"] = False
            return

        # Encontra a GPU com maior peso
        max_w = max(w.weight for w in weights)
        if max_w <= 10:
            logging.error("Pesos já estão no mínimo. Falha total.")
            recovery_state["active"] = False
            return

        main_gpu = max(weights, key=lambda x: x.weight)
        other_gpus = [w for w in weights if w != main_gpu]
        
        # Reduz 10% da principal e distribui nos outros
        reduction = min(10.0, main_gpu.weight - 5.0)
        main_gpu.weight -= reduction
        
        share = reduction / len(other_gpus)
        for og in other_gpus:
            og.weight += share
        
        logging.info(f"Retentando com novos pesos: {[f'{w.index}:{w.weight}' for w in weights]}")
        execute_start(req)
        time.sleep(2) # Dá um tempo para o processo subir
        recovery_state["active"] = False

def execute_start(req: StartRequest):
    stop_model()
    try:
        all_gpus = get_gpu_info()
        max_idx = max(g['index'] for g in all_gpus) if all_gpus else 0
        weights_map = {gw.index: gw.weight for gw in req.gpu_weights}
        split = []
        total_user = sum(weights_map.values()) or 1
        for i in range(max_idx + 1):
            split.append(f"{weights_map.get(i, 0)/total_user:.4f}")
        
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
    # Inicia thread de monitoramento de log
    threading.Thread(target=monitor_oom, daemon=True).start()
    
    config = load_config()
    default_model = config.get("default_model")
    if default_model and os.path.exists(default_model):
        if not find_llama_server().get("running"):
            logging.info(f"Auto-start: {default_model}")
            try:
                gpus = get_gpu_info()
                weights = []
                max_vram = max(g['vram'] for g in gpus) if gpus else 0
                main_gpu_idx = next((g['index'] for g in gpus if g['vram'] == max_vram), -1)

                for g in gpus:
                    if len(gpus) == 1:
                        val = 100.0
                    elif g['index'] == main_gpu_idx:
                        val = 90.0
                    else:
                        val = 10.0 / (len(gpus) - 1)
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
    models = get_gguf_models()
    gpus = get_gpu_info()
    config = load_config()
    default_model = config.get("default_model")
    
    model_items = ""
    for m in models:
        m_js = m.replace("\\", "/")
        m_name = os.path.basename(m)
        m_dir = os.path.dirname(m).replace(MODELS_DIR, "")
        is_default = "checked" if m == default_model else ""
        model_items += f"""
        <div class="group flex items-center justify-between p-4 mb-3 bg-slate-800/40 backdrop-blur-md rounded-2xl hover:bg-slate-700/60 transition-all duration-300 border border-slate-700/50 hover:border-blue-500/50 shadow-lg">
            <div class="flex-1 min-w-0 mr-4">
                <div class="flex items-center gap-2 mb-1">
                    <i class="fas fa-cube text-blue-400 text-[10px]"></i>
                    <p class="text-sm font-bold text-slate-100 truncate" title="{m_name}">{m_name}</p>
                </div>
                <p class="text-[9px] text-slate-500 truncate uppercase tracking-tighter font-mono">{m_dir or "/"}</p>
            </div>
            <div class="flex items-center gap-3 md:gap-4">
                <button onclick="deleteModel('{m_js}')" class="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-red-500/20 text-slate-600 hover:text-red-500 transition-all" title="Excluir Modelo">
                    <i class="fas fa-trash-alt text-[10px]"></i>
                </button>
                <div class="flex flex-col items-center gap-1">
                    <span class="text-[8px] font-black text-slate-600 uppercase tracking-tighter">Padrão</span>
                    <input type="checkbox" class="model-default-checkbox w-4 h-4 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer" 
                           {is_default} onclick="setDefaultModel(this, '{m_js}')">
                </div>
                <button onclick="startModel('{m_js}')" class="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black rounded-xl active:scale-95 flex items-center gap-2 uppercase tracking-widest shadow-xl">
                    <i class="fas fa-play text-[8px]"></i> <span class="hidden sm:inline">CARREGAR</span><span class="sm:hidden">LOAD</span>
                </button>
            </div>
        </div>
        """
    
    gpu_rows = ""
    max_vram = max(g['vram'] for g in gpus) if gpus else 0
    # Encontra o índice da primeira GPU com o máximo de VRAM
    main_gpu_idx = next((g['index'] for g in gpus if g['vram'] == max_vram), -1)
    
    for g in gpus:
        if len(gpus) == 1:
            default_val = "100"
        elif g['index'] == main_gpu_idx:
            default_val = "90"
        else:
            default_val = str(round(10 / (len(gpus) - 1)))
        
        gpu_rows += f"""
        <tr class="gpu-row group border-b border-slate-800/50" data-index="{g['index']}">
            <td class="px-3 md:px-6 py-4 md:py-6 text-center">
                <div class="flex flex-col items-center gap-2">
                    <span class="gpu-util-val text-xs font-black text-blue-400 font-mono">0%</span>
                    <div class="w-12 h-1 bg-slate-800 rounded-full overflow-hidden"><div class="gpu-util-bar h-full bg-blue-500 transition-all duration-1000" style="width: 0%"></div></div>
                </div>
            </td>
            <td class="px-2 md:px-4 py-4 md:py-6">
                <div class="flex items-center gap-2 md:gap-4">
                    <input type="checkbox" checked class="gpu-checkbox w-5 h-5 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer">
                    <div class="flex flex-col"><span class="text-[9px] font-black text-blue-400 uppercase tracking-widest mb-0.5">ID {g['index']}</span><span class="text-sm font-bold text-slate-100 whitespace-nowrap">{g['name']}</span></div>
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
                    <input type="number" value="{default_val}" min="0" max="100" class="gpu-weight w-20 pl-2 md:pl-3 pr-6 md:pr-8 py-2 bg-slate-900/80 border border-slate-700 rounded-xl text-sm font-black text-blue-400 outline-none transition-all" oninput="balanceWeights(this)">
                    <span class="absolute right-2 md:right-3 top-1/2 -translate-y-1/2 text-[10px] font-black text-slate-600">%</span>
                </div>
            </td>
        </tr>
        """
    
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
        </style>
    </head>
    <body class="min-h-screen text-slate-200 pb-16 selection:bg-blue-500/30">
        <div class="max-w-[1600px] mx-auto px-4 md:px-8 pt-6 md:pt-10">
            <header class="flex flex-col md:flex-row items-center justify-between mb-8 md:mb-10 glass p-4 md:p-5 rounded-3xl md:rounded-[2rem] gap-4">
                <div class="flex items-center gap-4 md:gap-6">
                    <div class="bg-blue-600 p-3 rounded-2xl shadow-xl shadow-blue-500/20"><i class="fas fa-brain text-white text-xl md:text-2xl"></i></div>
                    <div>
                        <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2 md:gap-3">Automanager <span class="text-blue-500 font-light">Llama.cpp</span></h1>
                        <p class="text-[10px] text-slate-500 font-mono tracking-wider uppercase">Interface Avançada de Computação Neural</p>
                    </div>
                </div>
                <div class="flex items-center gap-4 md:gap-8 w-full md:w-auto justify-center md:justify-end">
                    <div class="hidden sm:flex items-center gap-3 md:gap-5 px-4 md:px-6 py-2 bg-slate-900/50 rounded-xl border border-slate-800">
                        <div class="flex flex-col items-end"><span class="text-[9px] md:text-[10px] text-slate-500 font-black uppercase tracking-tighter">IP do Motor</span><span id="display-ip" class="text-xs font-mono text-blue-400">#FIXED_IP#</span></div>
                        <i class="fas fa-network-wired text-slate-600 text-sm md:text-base"></i>
                    </div>
                    <div id="status-badge" class="px-6 md:px-8 py-2 md:py-2.5 rounded-xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 glass border-slate-700/50 text-slate-500 uppercase transition-all duration-500"><div class="w-2 h-2 rounded-full bg-slate-600"></div>OFFLINE</div>
                </div>
            </header>
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 md:gap-10">
                <div class="lg:col-span-8 space-y-6 md:space-y-10">
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
                             <div class="flex items-center gap-4 md:gap-6 bg-slate-900/80 p-1.5 rounded-2xl border border-slate-800"><label class="text-[9px] font-black uppercase text-slate-400 pl-3 md:pl-4 tracking-widest">Contexto:</label><select id="context-size" class="bg-blue-600/20 border border-blue-500/30 text-blue-300 rounded-xl px-4 py-2 text-xs md:text-sm font-bold focus:ring-2 focus:ring-blue-500/50 outline-none transition-all cursor-pointer"><option value="2048" class="bg-slate-900">2K</option><option value="4096" class="bg-slate-900">4K</option><option value="8192" class="bg-slate-900">8K</option><option value="16384" class="bg-slate-900">16K</option><option value="32768" class="bg-slate-900">32K</option><option value="65536" class="bg-slate-900">64K</option><option value="131072" selected class="bg-slate-900">128K</option><option value="262144" class="bg-slate-900">256K</option></select></div>
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
                <div class="lg:col-span-4 space-y-6 md:space-y-10">
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
            document.getElementById('chat-link').href = `http://${fixedIp}:8085/`;
            document.getElementById('api-link').innerText = `http://${fixedIp}:8085/v1`;
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
                try {
                    const response = await fetch('/logs', { signal: logStream.signal }); const reader = response.body.getReader(); const decoder = new TextDecoder();
                    while (true) {
                        const { value, done } = await reader.read(); if (done) break;
                        const formatted = decoder.decode(value).replace(/error/gi, '<span class="text-red-500 font-black px-1 rounded bg-red-500/10">ERRO</span>').replace(/warn/gi, '<span class="text-amber-500 font-black px-1 rounded bg-amber-500/10">AVISO</span>').replace(/info/gi, '<span class="text-blue-400 font-bold uppercase tracking-tighter">info</span>');
                        const line = document.createElement('div'); line.className = 'terminal-line mb-1 md:mb-2 border-l border-slate-800 md:border-l-2 pl-3 md:pl-4'; line.innerHTML = formatted; box.appendChild(line); box.scrollTop = box.scrollHeight; if (box.childNodes.length > 500) box.removeChild(box.firstChild);
                    }
                } catch(e) {}
            }
            function updateUptime() { if (!startTime) return; const diff = Math.floor((new Date() - startTime) / 1000); document.getElementById('uptime-val').innerText = `${Math.floor(diff/3600)}h ${Math.floor((diff%3600)/60)}m ${diff%60}s`; }
            async function updateStatus() {
                try {
                    const res = await fetch('/status'); const data = await res.json(); const badge = document.getElementById('status-badge'); const card = document.getElementById('active-card');
                    
                    // Atualiza pesos se houver mudança externa (auto-recovery)
                    if (data.current_weights) {
                        data.current_weights.forEach(w => {
                            const input = document.querySelector(`.gpu-row[data-index="${w.index}"] .gpu-weight`);
                            if (input && document.activeElement !== input) {
                                const newWeight = Math.round(w.weight);
                                if (parseInt(input.value) !== newWeight) {
                                    input.value = newWeight;
                                    updateTotal();
                                }
                            }
                        });
                    }

                    if (data.recovery && data.recovery.active) {
                        badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-amber-500/50 text-amber-500 uppercase';
                        badge.innerHTML = '<i class="fas fa-sync animate-spin mr-1"></i> REALOCANDO...';
                        return;
                    }

                    if (data.running) {
                        if (!startTime) startTime = new Date(); badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-emerald-500/30 text-emerald-500 uppercase glow-online'; badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-emerald-500 animate-pulse"></div> ONLINE'; card.classList.remove('hidden'); document.getElementById('active-model-name').innerText = data.model; if (!logStream) startLogs(); updateUptime();
                    } else {
                        startTime = null; badge.className = 'px-5 md:px-8 py-2 md:py-3 rounded-2xl text-[10px] md:text-xs font-black tracking-[0.2em] flex items-center gap-3 md:gap-4 glass border-slate-700/50 text-slate-500 uppercase'; badge.innerHTML = '<div class="w-2 md:w-2.5 h-2 md:h-2.5 rounded-full bg-slate-600"></div> OFFLINE'; card.classList.add('hidden'); if (logStream) { logStream.abort(); logStream = null; }
                    }
                } catch(e) {}
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
                    const [res, cfgRes] = await Promise.all([fetch('/models'), fetch('/config')]); const data = await res.json(); const cfg = await cfgRes.json(); document.getElementById('model-count').innerText = `${data.length} UNIDADES`;
                    document.getElementById('model-list-container').innerHTML = data.map(m => {
                        const m_js = m.path.replace(/\\\\/g, '/');
                        return `<div class="group flex items-center justify-between p-4 md:p-5 mb-3 md:mb-4 bg-slate-800/40 backdrop-blur-md rounded-2xl hover:bg-slate-700/60 transition-all duration-300 border border-slate-700/50 hover:border-blue-500/50 shadow-lg"><div class="flex-1 min-w-0 mr-4 md:mr-6"><div class="flex items-center gap-2 md:gap-3 mb-1 md:mb-2"><i class="fas fa-cube text-blue-400 text-[10px] md:text-xs"></i><p class="text-sm md:text-base font-bold text-slate-100 truncate" title="${m.name}">${m.name}</p></div><p class="text-[9px] md:text-xs text-slate-500 truncate uppercase tracking-tighter font-mono">${m.dir}</p></div><div class="flex items-center gap-3 md:gap-6"><button onclick="deleteModel('${m_js}')" class="w-10 h-10 flex items-center justify-center rounded-xl hover:bg-red-500/20 text-slate-600 hover:text-red-500 transition-all" title="Excluir Modelo"><i class="fas fa-trash-alt text-[10px] md:text-xs"></i></button><div class="flex flex-col items-center gap-1 md:gap-1.5"><span class="text-[8px] md:text-[10px] font-black text-slate-600 uppercase tracking-tighter">Padrão</span><input type="checkbox" class="model-default-checkbox w-4 h-4 md:w-5 md:h-5 bg-slate-900 border-slate-700 rounded text-blue-600 cursor-pointer" ${m.path === cfg.default_model ? 'checked' : ''} onclick="setDefaultModel(this, '${m_js}')"></div><button onclick="startModel('${m_js}')" class="px-4 md:px-6 py-2 md:py-3 bg-blue-600 hover:bg-blue-500 text-white text-[10px] md:text-xs font-black rounded-xl active:scale-95 flex items-center gap-2 md:gap-3 uppercase tracking-widest shadow-xl"><i class="fas fa-play text-[8px] md:text-[10px]"></i> <span class="hidden sm:inline">CARREGAR</span><span class="sm:hidden">LOAD</span></button></div></div>`;
                    }).join('');
                } catch(e) {}
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
            async function startModel(path) { const weights = []; document.querySelectorAll('.gpu-row').forEach(r => { if (r.querySelector('.gpu-checkbox').checked) weights.push({ index: parseInt(r.dataset.index), weight: parseInt(r.querySelector('.gpu-weight').value || 0), name: "GPU" }); }); if (!weights.length) return alert("SELECIONE UMA GPU"); document.getElementById('status-badge').innerHTML = '<i class="fas fa-circle-notch animate-spin mr-2 md:mr-3 text-sm md:text-lg"></i> INICIALIZANDO...'; await fetch('/start', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ path, gpu_weights: weights, context_size: parseInt(document.getElementById('context-size').value) }) }); setTimeout(updateStatus, 2000); }
            async function stopModel() { if (confirm("ENCERRAR PROCESSO?")) { await fetch('/stop', {method: 'POST'}); setTimeout(updateStatus, 1000); } }
            setInterval(updateMetrics, 2000); setInterval(updateStatus, 3000); setInterval(updateDownloads, 3000);
            updateStatus(); updateMetrics(); updateDownloads(); updateModels(); updateTotal();
        </script>
    </body>
    </html>
    """
    final_html = html_template.replace("#GPU_ROWS#", gpu_rows).replace("#MODEL_ITEMS#", model_items).replace("#FIXED_IP#", FIXED_IP)
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
    
    # Verifica se o modelo está em execução
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

@app.get("/status")
def get_status():
    status = find_llama_server()
    status["recovery"] = recovery_state
    if last_start_request:
        status["current_weights"] = [w.dict() for w in last_start_request.gpu_weights]
    return status

@app.get("/models")
def list_models():
    models = get_gguf_models()
    result = []
    for m in models:
        result.append({
            "path": m,
            "name": os.path.basename(m),
            "dir": os.path.dirname(m).replace(MODELS_DIR, "") or "/"
        })
    return result

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
    global last_start_request
    last_start_request = req
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
