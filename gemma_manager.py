import subprocess
import os
import signal
import glob
import psutil
import json
import logging
import re
from fastapi import FastAPI, HTTPException
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
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"running": False}

@app.get("/metrics")
def get_metrics():
    try:
        # GPU Metrics
        gpu_output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]).decode()
        gpus = []
        for line in gpu_output.strip().split("\n"):
            idx, util, mem_used, mem_total = line.split(", ")
            gpus.append({
                "index": idx,
                "util": util,
                "mem_used": mem_used,
                "mem_total": mem_total
            })
        
        return {
            "cpu": psutil.cpu_percent(),
            "ram": psutil.virtual_memory().percent,
            "gpus": gpus
        }
    except:
        return {"cpu": 0, "ram": 0, "gpus": []}

@app.get("/logs")
def stream_logs():
    def generate():
        if not os.path.exists(SERVER_LOG_PATH):
            yield "Log file not found.\n"
            return
        with open(SERVER_LOG_PATH, 'r') as f:
            lines = f.readlines()
            for line in lines[-100:]:
                yield line
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
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
            <!-- Top Navigation -->
            <nav class="flex items-center justify-between mb-10 bg-white p-5 rounded-2xl shadow-sm border border-slate-100">
                <div class="flex items-center gap-4">
                    <div class="bg-blue-600 p-2.5 rounded-xl shadow-lg shadow-blue-200"><i class="fas fa-bolt text-white"></i></div>
                    <h1 class="text-xl font-extrabold tracking-tight">Llama Manager <span class="text-blue-600">PRO</span></h1>
                </div>
                <div id="status-badge" class="px-5 py-2 rounded-full text-[10px] font-black tracking-widest flex items-center gap-2 bg-slate-100 text-slate-400">OFFLINE</div>
            </nav>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                <div class="lg:col-span-8 space-y-8">
                    <!-- System Monitor -->
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

                    <!-- Active Model -->
                    <div id="active-card" class="bg-slate-900 p-8 rounded-3xl shadow-xl hidden border-b-8 border-blue-600">
                        <div class="flex items-center justify-between gap-6">
                            <div class="min-w-0">
                                <p class="text-blue-400 text-[10px] font-black uppercase tracking-widest mb-1">Modelo Ativo</p>
                                <h2 id="active-model-name" class="text-xl font-black text-white truncate">--</h2>
                            </div>
                            <button onclick="stopModel()" class="px-8 py-3 bg-red-600 text-white rounded-xl text-xs font-black hover:bg-red-700 transition-all shadow-lg active:scale-95">PARAR SERVIDOR</button>
                        </div>
                    </div>

                    <!-- GPU Config -->
                    <div class="bg-white rounded-3xl border border-slate-100 shadow-sm overflow-hidden">
                        <div class="p-6 border-b border-slate-50 flex items-center justify-between">
                            <h3 class="font-black text-sm uppercase tracking-wider text-slate-500">Hardware NVIDIA</h3>
                            <span id="total-percent" class="text-[10px] font-black bg-slate-900 text-white px-3 py-1 rounded-full">TOTAL: 100%</span>
                        </div>
                        <table class="w-full text-left">
                            <tbody class="divide-y divide-slate-50">{{gpu_rows}}</tbody>
                        </table>
                    </div>

                    <!-- Console -->
                    <div class="bg-slate-900 rounded-3xl overflow-hidden shadow-2xl">
                        <div class="px-6 py-4 bg-slate-800/40 border-b border-slate-800 flex justify-between items-center">
                            <p class="text-white text-[10px] font-black uppercase tracking-widest italic">Terminal Output</p>
                            <button onclick="document.getElementById('log-box').innerHTML=''" class="text-[9px] text-slate-500 hover:text-white font-bold">CLEAR</button>
                        </div>
                        <div id="log-box" class="custom-scroll p-6 h-64 overflow-y-auto font-mono text-[10px] text-slate-400 leading-relaxed whitespace-pre-wrap bg-black/20"></div>
                    </div>
                </div>

                <!-- Models Sidebar -->
                <div class="lg:col-span-4">
                    <div class="bg-white rounded-3xl border border-slate-100 shadow-sm flex flex-col h-[700px]">
                        <div class="p-6 border-b border-slate-50 flex items-center gap-3">
                            <i class="fas fa-folder text-amber-400"></i>
                            <h3 class="font-black text-sm uppercase tracking-wider text-slate-500">Modelos Disponíveis</h3>
                        </div>
                        <div class="p-4 flex-1 overflow-y-auto custom-scroll">{{model_items}}</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let logStream = null;

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
                        <div class="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm">
                            <p class="text-[10px] font-black text-slate-400 uppercase mb-2">GPU ${g.index} Util</p>
                            <div class="flex items-end justify-between"><h3 class="text-2xl font-black">${g.util}%</h3><div class="w-12 h-1 bg-slate-100 rounded-full overflow-hidden"><div class="h-full bg-green-500" style="width: ${g.util}%"></div></div></div>
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
                        box.innerHTML += decoder.decode(value).replace(/error/gi, '<span class="text-red-500 font-bold">ERR</span>');
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

            async function startModel(path) {
                const weights = [];
                document.querySelectorAll('.gpu-row').forEach(r => {
                    if (r.querySelector('.gpu-checkbox').checked) {
                        weights.push({ index: parseInt(r.dataset.index), weight: parseInt(r.querySelector('.gpu-weight').value || 0), name: r.dataset.name });
                    }
                });
                if (!weights.length) return alert("Selecione uma GPU!");
                document.getElementById('status-badge').innerText = 'STARTING...';
                await fetch('/start', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ path, gpu_weights: weights }) });
                setTimeout(updateStatus, 3000);
            }

            async function stopModel() {
                if (confirm("Stop server?")) await fetch('/stop', {method: 'POST'});
                setTimeout(updateStatus, 1500);
            }

            setInterval(updateMetrics, 2000);
            setInterval(updateStatus, 5000);
            updateStatus(); updateMetrics();
        </script>
    </body>
    </html>
    """
    return html_template.replace("{{gpu_rows}}", gpu_rows).replace("{{model_items}}", model_items)

class GPUWeight(BaseModel):
    index: int
    weight: int
    name: str

class StartRequest(BaseModel):
    path: str
    gpu_weights: List[GPUWeight]

@app.get("/status")
def get_status():
    return find_llama_server()

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
        
        cmd = ["llama-server", "-m", req.path, "-ngl", "99", "--flash-attn", "on", "--host", "0.0.0.0", "--port", "8085", "--tools", "all", "--parallel", "1", "--ctx-size", "32768", "--mlock", "--main-gpu", main_gpu, "--tensor-split", ",".join(split)]
        
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
