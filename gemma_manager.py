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

app = FastAPI(title="Llama.cpp Model Manager")

MODELS_DIR = "/media/docker/models"
SERVER_LOG_PATH = "/root/gemma_server.log"
FIXED_IP = "192.168.2.183"

# Estado dos downloads
downloads = {}

def download_model_task(download_id: str, url: str, filename: Optional[str]):
    try:
        if not filename:
            filename = url.split("/")[-1].split("?")[0]
            if not filename.endswith(".gguf"):
                filename += ".gguf"
        
        # Remove a extensão para criar o nome da pasta
        model_name_folder = filename.replace(".gguf", "")
        model_specific_dir = os.path.join(MODELS_DIR, model_name_folder)
        
        # Garante que o diretório específico existe
        os.makedirs(model_specific_dir, exist_ok=True)
        path = os.path.join(model_specific_dir, filename)
        
        # Se o arquivo já existe dentro da pasta, adiciona um sufixo
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
        output = subprocess.check_output("llama-server --help 2>&1", shell=True).decode()
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

@app.get("/metrics")
def get_metrics():
    try:
        gpu_output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]).decode()
        gpus = []
        for line in gpu_output.strip().split("\n"):
            if not line.strip(): continue
            parts = line.split(", ")
            if len(parts) == 4:
                idx, util, mem_used, mem_total = parts
                gpus.append({"index": idx, "util": util, "mem_used": mem_used, "mem_total": mem_total})
        return {"cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent, "gpus": gpus}
    except: return {"cpu": 0, "ram": 0, "gpus": []}

@app.get("/logs")
def stream_logs():
    def generate():
        if not os.path.exists(SERVER_LOG_PATH):
            yield "Log file not found.\n"
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
    model_items = ""
    for m in models:
        m_name = os.path.basename(m)
        m_dir = os.path.dirname(m).replace(MODELS_DIR, "")
        model_items += f"""
        <div class="flex items-center justify-between p-3 mb-2 bg-gray-50 rounded-xl hover:bg-blue-50 transition-all border border-transparent hover:border-blue-100">
            <div class="flex-1 min-w-0">
                <p class="text-sm font-bold text-gray-800 truncate" title="{m_name}">{m_name}</p>
                <p class="text-[9px] text-gray-400 truncate uppercase">{m_dir or "/"}</p>
            </div>
            <button onclick="startModel('{m}')" class="ml-3 px-4 py-2 bg-blue-600 text-white text-[10px] font-black rounded-lg hover:bg-blue-700 transition-all shadow-md active:scale-95 uppercase">Iniciar</button>
        </div>
        """
    gpu_rows = ""
    for g in gpus:
        is_3090 = "3090" in g['name']
        default_val = "95" if is_3090 else "5"
        if len(gpus) == 1: default_val = "100"
        gpu_rows += f"""
        <tr class="gpu-row" data-index="{g['index']}" data-name="{g['name']}">
            <td class="px-6 py-3 text-center"><input type="checkbox" checked class="gpu-checkbox w-4 h-4 text-blue-600 rounded"></td>
            <td class="px-4 py-3 text-sm font-black text-gray-900">ID {g['index']}</td>
            <td class="px-4 py-3 text-sm font-medium text-gray-600">{g['name']}</td>
            <td class="px-4 py-3 text-xs font-mono text-gray-400">{g['vram']} MB</td>
            <td class="px-4 py-3">
                <div class="flex items-center gap-1">
                    <input type="number" value="{default_val}" min="0" max="100" class="gpu-weight w-16 px-2 py-1 text-sm font-black border border-gray-200 rounded-md focus:ring-2 focus:ring-blue-500 outline-none" onchange="updateTotal()">
                    <span class="text-[10px] font-bold text-gray-300">%</span>
                </div>
            </td>
        </tr>
        """
    
    html_template = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Llama Manager PRO</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
            body { font-family: 'Plus Jakarta Sans', sans-serif; }
            .custom-scroll::-webkit-scrollbar { width: 4px; }
            .custom-scroll::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 10px; }
        </style>
    </head>
    <body class="bg-[#f8fafc] min-h-screen text-slate-900 pb-20">
        <div class="max-w-7xl mx-auto px-6 pt-10">
            <nav class="flex items-center justify-between mb-10 bg-white p-5 rounded-2xl shadow-sm border border-slate-100">
                <div class="flex items-center gap-4">
                    <div class="bg-blue-600 p-2.5 rounded-xl shadow-lg shadow-blue-200"><i class="fas fa-bolt text-white"></i></div>
                    <h1 class="text-xl font-extrabold tracking-tight">Llama Manager <span class="text-blue-600">PRO</span></h1>
                </div>
                <div id="status-badge" class="px-5 py-2 rounded-full text-[10px] font-black tracking-widest flex items-center gap-2 bg-slate-100 text-slate-400 uppercase">OFFLINE</div>
            </nav>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                <div class="lg:col-span-8 space-y-8">
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div class="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm">
                            <p class="text-[10px] font-black text-slate-400 uppercase mb-2">CPU Usage</p>
                            <div class="flex items-end justify-between"><h3 id="cpu-val" class="text-2xl font-black">0%</h3><div class="w-12 h-1 bg-slate-100 rounded-full overflow-hidden"><div id="cpu-bar" class="h-full bg-blue-500 transition-all" style="width: 0%"></div></div></div>
                        </div>
                        <div class="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm">
                            <p class="text-[10px] font-black text-slate-400 uppercase mb-2">System RAM</p>
                            <div class="flex items-end justify-between"><h3 id="ram-val" class="text-2xl font-black">0%</h3><div class="w-12 h-1 bg-slate-100 rounded-full overflow-hidden"><div id="ram-bar" class="h-full bg-indigo-500 transition-all" style="width: 0%"></div></div></div>
                        </div>
                        <div id="gpu-stats-container" class="contents"></div>
                    </div>

                    <div class="bg-white rounded-3xl border border-slate-100 shadow-sm overflow-hidden p-6 space-y-6">
                        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-50 pb-4">
                             <div>
                                <h3 class="font-black text-sm uppercase tracking-wider text-slate-500">Hardware & Configuração</h3>
                                <p class="text-xs text-slate-400 font-medium mt-1">Configure o hardware antes de iniciar</p>
                             </div>
                             <div class="flex items-center gap-3">
                                <label class="text-[10px] font-black uppercase text-slate-400">Contexto:</label>
                                <select id="context-size" class="bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-xs font-black focus:ring-2 focus:ring-blue-500 outline-none">
                                    <option value="2048">2k</option><option value="4096">4k</option><option value="8192">8k</option><option value="16384">16k</option><option value="32768">32k</option><option value="65536" selected>64k (Padrão)</option><option value="98304">96k</option><option value="131072">128k</option><option value="262144">256k</option>
                                </select>
                             </div>
                        </div>
                        <div class="overflow-x-auto"><table class="w-full text-left"><tbody class="divide-y divide-slate-50">#GPU_ROWS#</tbody></table></div>
                        <div class="flex justify-end pt-2"><span id="total-percent" class="text-[10px] font-black bg-slate-900 text-white px-3 py-1 rounded-full">TOTAL: 100%</span></div>
                    </div>

                    <div id="active-card" class="bg-slate-900 p-8 rounded-3xl shadow-xl hidden border-b-8 border-blue-600 transition-all duration-300">
                        <div class="flex items-center justify-between gap-6">
                            <div class="min-w-0">
                                <p class="text-blue-400 text-[10px] font-black uppercase tracking-widest mb-1">Modelo Ativo</p>
                                <h2 id="active-model-name" class="text-xl font-black text-white truncate">--</h2>
                            </div>
                            <div class="flex gap-4">
                                <a id="chat-link" href="#" target="_blank" class="px-8 py-3 bg-blue-600 text-white rounded-xl text-xs font-black hover:bg-blue-700 transition-all shadow-lg active:scale-95 flex items-center gap-2">
                                    <i class="fas fa-comments"></i> ACESSAR CHAT
                                </a>
                                <button onclick="stopModel()" class="px-8 py-3 bg-red-600/20 text-red-500 rounded-xl text-xs font-black hover:bg-red-600 hover:text-white transition-all shadow-lg active:scale-95 border border-red-600/30 uppercase">Parar</button>
                            </div>
                        </div>
                    </div>

                    <div class="bg-slate-900 rounded-3xl overflow-hidden shadow-2xl">
                        <div class="px-6 py-4 bg-slate-800/40 border-b border-slate-800 flex justify-between items-center">
                            <p class="text-white text-[10px] font-black uppercase tracking-widest italic">Terminal Output</p>
                            <button onclick="document.getElementById('log-box').innerHTML=''" class="text-[9px] text-slate-500 hover:text-white font-bold uppercase">Limpar Console</button>
                        </div>
                        <div id="log-box" class="custom-scroll p-6 h-64 overflow-y-auto font-mono text-[10px] text-slate-400 leading-relaxed whitespace-pre-wrap bg-black/20"></div>
                    </div>
                </div>

                <div class="lg:col-span-4">
                    <div class="bg-white rounded-3xl border border-slate-100 shadow-sm flex flex-col h-[850px]">
                        <div class="p-6 border-b border-slate-50 flex items-center gap-3">
                            <i class="fas fa-folder text-amber-400"></i>
                            <h3 class="font-black text-sm uppercase tracking-wider text-slate-500">Modelos Disponíveis</h3>
                        </div>
                        
                        <div class="p-6 border-b border-slate-50 bg-slate-50/50">
                            <p class="text-[10px] font-black text-slate-400 uppercase mb-3 tracking-widest">Baixar Novo Modelo</p>
                            <div class="space-y-2">
                                <input type="text" id="download-url" placeholder="URL do arquivo .gguf" class="w-full px-4 py-2 text-xs border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all">
                                <button onclick="downloadModel()" class="w-full py-2.5 bg-green-600 text-white text-[10px] font-black rounded-xl hover:bg-green-700 transition-all shadow-lg shadow-green-100 active:scale-95 uppercase">
                                    <i class="fas fa-download mr-2"></i> Iniciar Download
                                </button>
                            </div>
                            <div id="download-status" class="mt-4 space-y-2"></div>
                        </div>

                        <div id="model-list-container" class="p-4 flex-1 overflow-y-auto custom-scroll">#MODEL_ITEMS#</div>
                        <div class="p-6 bg-slate-50 border-t border-slate-100 rounded-b-2xl text-center">
                             <p class="text-[9px] text-slate-400 font-black uppercase tracking-widest mb-1">OpenAI API Endpoint</p>
                             <p id="api-link" class="text-[10px] text-blue-600 font-mono font-bold"></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let logStream = null;
            const fixedIp = "#FIXED_IP#";
            document.getElementById('chat-link').href = `http://${fixedIp}:8085/`;
            document.getElementById('api-link').innerText = `http://${fixedIp}:8085/v1`;

            function updateTotal() {
                let sum = 0;
                document.querySelectorAll('.gpu-weight').forEach(i => sum += parseInt(i.value || 0));
                const badge = document.getElementById('total-percent');
                badge.innerText = `TOTAL: ${sum}%`;
                badge.className = sum === 100 ? 'text-[10px] font-black bg-slate-900 text-white px-3 py-1 rounded-full' : 'text-[10px] font-black bg-red-600 text-white px-3 py-1 rounded-full';
            }

            async function updateMetrics() {
                try {
                    const res = await fetch('/metrics');
                    const data = await res.json();
                    document.getElementById('cpu-val').innerText = data.cpu + '%';
                    document.getElementById('cpu-bar').style.width = data.cpu + '%';
                    document.getElementById('ram-val').innerText = data.ram + '%';
                    document.getElementById('ram-bar').style.width = data.ram + '%';
                    const container = document.getElementById('gpu-stats-container');
                    container.innerHTML = data.gpus.map(g => `
                        <div class="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm transition-all duration-500">
                            <p class="text-[10px] font-black text-slate-400 uppercase mb-2">GPU ${g.index} Utilization</p>
                            <div class="flex items-end justify-between"><h3 class="text-2xl font-black">${g.util}%</h3><div class="w-12 h-1 bg-slate-100 rounded-full overflow-hidden"><div class="h-full bg-green-500 transition-all duration-1000" style="width: ${g.util}%"></div></div></div>
                            <p class="text-[9px] font-bold text-slate-400 mt-2">VRAM: ${g.mem_used} / ${g.mem_total} MB</p>
                        </div>
                    `).join('');
                } catch(e) {}
            }

            async function startLogs() {
                if (logStream) logStream.abort();
                logStream = new AbortController();
                const box = document.getElementById('log-box');
                try {
                    const response = await fetch('/logs', { signal: logStream.signal });
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    while (true) {
                        const { value, done } = await reader.read();
                        if (done) break;
                        box.innerHTML += decoder.decode(value).replace(/error/gi, '<span class="text-red-500 font-bold">ERROR</span>');
                        box.scrollTop = box.scrollHeight;
                    }
                } catch(e) {}
            }

            async function updateStatus() {
                try {
                    const res = await fetch('/status');
                    const data = await res.json();
                    const badge = document.getElementById('status-badge');
                    const card = document.getElementById('active-card');
                    if (data.running) {
                        badge.className = 'px-5 py-2 rounded-full text-[10px] font-black tracking-widest flex items-center gap-2 bg-green-100 text-green-600';
                        badge.innerText = 'ONLINE';
                        card.classList.remove('hidden');
                        document.getElementById('active-model-name').innerText = data.model;
                        if (!logStream) startLogs();
                    } else {
                        badge.className = 'px-5 py-2 rounded-full text-[10px] font-black tracking-widest flex items-center gap-2 bg-slate-100 text-slate-400';
                        badge.innerText = 'OFFLINE';
                        card.classList.add('hidden');
                        if (logStream) { logStream.abort(); logStream = null; }
                    }
                } catch(e) {}
            }

            async function downloadModel() {
                const urlInput = document.getElementById('download-url');
                const url = urlInput.value.trim();
                if (!url) return alert("Por favor, insira a URL do modelo GGUF.");
                
                try {
                    const res = await fetch('/download', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ url })
                    });
                    if (res.ok) {
                        urlInput.value = '';
                        updateDownloads();
                    } else {
                        alert("Erro ao iniciar download.");
                    }
                } catch(e) {
                    alert("Erro de conexão.");
                }
            }

            async function updateDownloads() {
                try {
                    const res = await fetch('/downloads');
                    const data = await res.json();
                    const container = document.getElementById('download-status');
                    const entries = Object.entries(data);
                    
                    if (entries.length === 0) {
                        container.innerHTML = '';
                        return;
                    }

                    let hasCompleted = false;
                    container.innerHTML = entries.map(([id, d]) => {
                        if (d.status === 'completed') hasCompleted = true;
                        return `
                        <div class="p-3 bg-white rounded-xl border border-slate-100 shadow-sm">
                            <div class="flex justify-between items-center mb-2">
                                <p class="text-[9px] font-black truncate flex-1 mr-2 text-slate-700" title="${d.filename}">${d.filename}</p>
                                <span class="text-[8px] font-black uppercase px-2 py-0.5 rounded-full ${
                                    d.status === 'completed' ? 'bg-green-100 text-green-600' : 
                                    d.status === 'failed' ? 'bg-red-100 text-red-600' : 
                                    'bg-blue-100 text-blue-600'
                                }">${d.status}</span>
                            </div>
                            <div class="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                <div class="h-full bg-blue-500 transition-all duration-500" style="width: ${d.progress}%"></div>
                            </div>
                            ${d.error ? `<p class="text-[8px] text-red-500 mt-1 font-medium italic">${d.error}</p>` : ''}
                        </div>
                        `;
                    }).join('');
                    
                    if (hasCompleted) updateModels();
                } catch(e) {}
            }

            async function updateModels() {
                try {
                    const res = await fetch('/models');
                    const data = await res.json();
                    const container = document.getElementById('model-list-container');
                    container.innerHTML = data.map(m => `
                        <div class="flex items-center justify-between p-3 mb-2 bg-gray-50 rounded-xl hover:bg-blue-50 transition-all border border-transparent hover:border-blue-100">
                            <div class="flex-1 min-w-0">
                                <p class="text-sm font-bold text-gray-800 truncate" title="${m.name}">${m.name}</p>
                                <p class="text-[9px] text-gray-400 truncate uppercase">${m.dir}</p>
                            </div>
                            <button onclick="startModel('${m.path}')" class="ml-3 px-4 py-2 bg-blue-600 text-white text-[10px] font-black rounded-lg hover:bg-blue-700 transition-all shadow-md active:scale-95 uppercase">Iniciar</button>
                        </div>
                    `).join('');
                } catch(e) {}
            }

            async function startModel(path) {
                const weights = [];
                document.querySelectorAll('.gpu-row').forEach(r => {
                    if (r.querySelector('.gpu-checkbox').checked) {
                        weights.push({ index: parseInt(r.dataset.index), weight: parseInt(r.querySelector('.gpu-weight').value || 0), name: r.dataset.name });
                    }
                });
                if (!weights.length) return alert("Selecione uma GPU!");
                const ctxSize = document.getElementById('context-size').value;
                document.getElementById('status-badge').innerText = 'STARTING...';
                await fetch('/start', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ path, gpu_weights: weights, context_size: parseInt(ctxSize) }) });
                setTimeout(updateStatus, 3000);
            }

            async function stopModel() {
                if (confirm("Parar o servidor?")) await fetch('/stop', {method: 'POST'});
                setTimeout(updateStatus, 1500);
            }

            setInterval(updateMetrics, 2000);
            setInterval(updateStatus, 5000);
            setInterval(updateDownloads, 3000);
            updateStatus(); updateMetrics(); updateDownloads(); updateModels();
        </script>
    </body>
    </html>
    """
    
    final_html = html_template.replace("#GPU_ROWS#", gpu_rows)
    final_html = final_html.replace("#MODEL_ITEMS#", model_items)
    final_html = final_html.replace("#FIXED_IP#", FIXED_IP)
    
    return final_html

class GPUWeight(BaseModel):
    index: int
    weight: int
    name: str

class StartRequest(BaseModel):
    path: str
    gpu_weights: List[GPUWeight]
    context_size: int = 65536

class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None

@app.get("/status")
def get_status():
    return find_llama_server()

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
        env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        
        cmd = [
            "llama-server", "-m", req.path, "-ngl", "99", "--flash-attn", "on", 
            "--host", "0.0.0.0", "--port", "8085", "--tools", "all", 
            "--parallel", "1", "--ctx-size", str(req.context_size), "--mlock", 
            "--main-gpu", main_gpu, "--tensor-split", ",".join(split)
        ]
        
        logging.info(f"START: {' '.join(cmd)}")
        subprocess.Popen(cmd, stdout=open(SERVER_LOG_PATH, "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env)
        return {"message": "Started"}
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
def stop_model():
    subprocess.run(["pkill", "-9", "llama-server"])
    return {"message": "Stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
