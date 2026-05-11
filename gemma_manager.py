import subprocess
import os
import signal
import glob
import psutil
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

# Configura log para o arquivo do gerenciador
logging.basicConfig(filename='/root/manager.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Llama.cpp Model Manager")

MODELS_DIR = "/media/docker/models"

def get_gguf_models():
    """Scans the MODELS_DIR for all files with the .gguf extension."""
    files = glob.glob(os.path.join(MODELS_DIR, "**/*.gguf"), recursive=True)
    return sorted(files)

def get_gpu_info():
    """Retrieves information about available NVIDIA GPUs using nvidia-smi."""
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits"
        ]).decode()
        gpus = []
        for line in output.strip().split("\n"):
            idx, name, mem = line.split(", ")
            gpus.append({"index": int(idx), "name": name, "vram": int(mem)})
        return gpus
    except Exception as e:
        logging.error(f"Error getting GPU info: {e}")
        return []

def find_llama_server():
    """Searches for a running 'llama-server' process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = proc.info['name'] or ""
            cmdline = proc.info['cmdline'] or []
            if 'llama-server' in name or (cmdline and 'llama-server' in cmdline[0]):
                model_name = "Unknown Model"
                for i in range(len(cmdline)-1):
                    if cmdline[i] in ["-m", "--model"]:
                        model_name = os.path.basename(cmdline[i+1])
                        break
                return {"running": True, "pid": proc.info['pid'], "model": model_name}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"running": False}

@app.get("/", response_class=HTMLResponse)
def index():
    models = get_gguf_models()
    gpus = get_gpu_info()
    
    # Model list generation
    model_items = ""
    for m in models:
        m_name = os.path.basename(m)
        m_dir = os.path.dirname(m).replace(MODELS_DIR, "")
        model_items += f"""
        <div class="flex items-center justify-between p-4 mb-2 bg-gray-50 rounded-lg hover:bg-blue-50 transition-colors group">
            <div class="flex-1 min-w-0">
                <p class="text-sm font-semibold text-gray-900 truncate" title="{m_name}">{m_name}</p>
                <p class="text-xs text-gray-500 truncate">{m_dir}</p>
            </div>
            <button onclick="startModel('{m}')" 
                class="ml-4 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all">
                Iniciar
            </button>
        </div>
        """

    # GPU list generation
    gpu_rows = ""
    for g in gpus:
        # Sugestão padrão baseada no nome (ex: 3090 ganha 95% se houver outra)
        default_checked = "checked"
        default_val = "100"
        if len(gpus) > 1:
            if "3090" in g['name']:
                default_val = "95"
            else:
                default_val = "5"

        gpu_rows += f"""
        <tr class="hover:bg-gray-50 transition-colors gpu-row" data-index="{g['index']}">
            <td class="px-4 py-3 whitespace-nowrap text-center">
                <input type="checkbox" {default_checked} class="gpu-checkbox w-5 h-5 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 cursor-pointer" onchange="autoBalance()">
            </td>
            <td class="px-4 py-3 whitespace-nowrap text-sm font-bold text-gray-900">GPU {g['index']}</td>
            <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-600">{g['name']}</td>
            <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">{g['vram']} MB</td>
            <td class="px-4 py-3 whitespace-nowrap">
                <div class="flex items-center">
                    <input type="number" value="{default_val}" min="0" max="100" 
                        class="gpu-weight w-24 px-3 py-2 text-sm font-bold border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                        onchange="updateTotal()">
                    <span class="ml-2 text-sm font-bold text-gray-500">%</span>
                </div>
            </td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Llama.cpp Manager</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            body {{ font-family: 'Inter', sans-serif; }}
            ::-webkit-scrollbar {{ width: 8px; }}
            ::-webkit-scrollbar-track {{ background: #f1f1f1; }}
            ::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 4px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: #94a3b8; }}
        </style>
    </head>
    <body class="bg-slate-50 min-h-screen text-slate-800">
        <div class="max-w-6xl mx-auto px-4 py-10">
            <!-- Header -->
            <header class="flex flex-col md:flex-row items-center justify-between mb-10 bg-white p-8 rounded-3xl shadow-xl shadow-slate-200/50 border border-white">
                <div class="flex items-center gap-6 mb-6 md:mb-0">
                    <div class="bg-gradient-to-br from-blue-600 to-indigo-700 p-4 rounded-2xl shadow-lg shadow-blue-200">
                        <i class="fas fa-microchip text-white text-3xl"></i>
                    </div>
                    <div>
                        <h1 class="text-3xl font-extrabold text-slate-900 tracking-tight">Llama.cpp Manager</h1>
                        <p class="text-slate-500 font-medium">Controle de Modelos LLM e Hardware NVIDIA</p>
                    </div>
                </div>
                <div id="status-badge" class="px-6 py-3 rounded-2xl text-sm font-black flex items-center gap-3 shadow-inner transition-all duration-500">
                    <div class="w-3 h-3 rounded-full animate-ping bg-slate-400"></div>
                    DETECTANDO...
                </div>
            </header>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-10">
                <!-- Main Content -->
                <div class="lg:col-span-8 space-y-10">
                    <!-- Active Model Dashboard -->
                    <div id="status-card" class="bg-gradient-to-r from-blue-600 to-indigo-700 p-8 rounded-3xl shadow-2xl shadow-blue-200 hidden transform transition-all">
                        <div class="flex flex-col md:flex-row items-center justify-between gap-6">
                            <div class="text-center md:text-left">
                                <p class="text-blue-100 text-xs font-black uppercase tracking-[0.2em] mb-2">Processo Ativo</p>
                                <h2 id="current-model-name" class="text-2xl font-black text-white leading-tight">--</h2>
                            </div>
                            <button onclick="stopModel()" class="flex items-center gap-3 px-8 py-4 bg-white/10 hover:bg-white text-white hover:text-red-600 rounded-2xl font-black transition-all border border-white/20 hover:border-white shadow-lg backdrop-blur-md">
                                <i class="fas fa-stop-circle"></i>
                                ENCERRAR SERVIDOR
                            </button>
                        </div>
                    </div>

                    <!-- GPU Table -->
                    <div class="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden">
                        <div class="p-8 border-b border-slate-50 flex items-center justify-between bg-slate-50/30">
                            <h3 class="font-black text-slate-900 flex items-center gap-3">
                                <i class="fas fa-bolt text-amber-500"></i>
                                Configuração de VRAM e GPUs
                            </h3>
                            <div class="flex items-center gap-4">
                                <button onclick="resetWeights()" class="text-xs font-bold text-blue-600 hover:underline">Resetar Pesos</button>
                                <span class="text-sm bg-slate-900 text-white px-4 py-2 rounded-xl font-black shadow-lg shadow-slate-200" id="total-percent">Total: 100%</span>
                            </div>
                        </div>
                        <div class="overflow-x-auto">
                            <table class="w-full text-left">
                                <thead class="bg-slate-50/50 text-slate-400 text-[10px] font-black uppercase tracking-[0.15em]">
                                    <tr>
                                        <th class="px-8 py-5 text-center">Ativar</th>
                                        <th class="px-6 py-5">ID</th>
                                        <th class="px-6 py-5">Modelo</th>
                                        <th class="px-6 py-5">Memória Total</th>
                                        <th class="px-6 py-5">Carga (%)</th>
                                    </tr>
                                </thead>
                                <tbody class="divide-y divide-slate-50">
                                    {gpu_rows}
                                </tbody>
                            </table>
                        </div>
                        <div class="p-6 bg-blue-50/50 border-t border-blue-100 flex items-start gap-4">
                            <i class="fas fa-info-circle text-blue-500 mt-1"></i>
                            <p class="text-xs text-blue-800 font-medium leading-relaxed">
                                <strong>Dica:</strong> Para modelos grandes, priorize a GPU com mais memória ou a mais rápida (ex: RTX 3090). O sistema normaliza os pesos automaticamente se a soma não for 100%.
                            </p>
                        </div>
                    </div>
                </div>

                <!-- Sidebar -->
                <div class="lg:col-span-4 space-y-8">
                    <div class="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 flex flex-col h-full min-h-[500px]">
                        <div class="p-8 border-b border-slate-50 bg-slate-50/30">
                            <h3 class="font-black text-slate-900 flex items-center gap-3 text-lg">
                                <i class="fas fa-box-open text-indigo-500"></i>
                                Seus Modelos
                            </h3>
                        </div>
                        <div class="p-6 flex-1 overflow-y-auto max-h-[700px] space-y-3 custom-scrollbar">
                            {model_items}
                        </div>
                        <div class="p-8 border-t border-slate-50 bg-slate-50/30 rounded-b-3xl">
                            <div class="bg-slate-900 p-4 rounded-2xl">
                                <p class="text-[9px] text-slate-500 font-black uppercase tracking-widest mb-2">API Connection</p>
                                <p class="text-xs text-blue-400 font-mono break-all font-bold">http://localhost:8085/v1</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function autoBalance() {{
                const rows = document.querySelectorAll('.gpu-row');
                const checkedRows = Array.from(rows).filter(r => r.querySelector('.gpu-checkbox').checked);
                
                if (checkedRows.length === 0) {{
                    updateTotal();
                    return;
                }}

                const share = Math.floor(100 / checkedRows.length);
                const remainder = 100 % checkedRows.length;
                
                rows.forEach(r => {{
                    const input = r.querySelector('.gpu-weight');
                    const isChecked = r.querySelector('.gpu-checkbox').checked;
                    input.disabled = !isChecked;
                    if (!isChecked) input.value = 0;
                }});

                checkedRows.forEach((r, idx) => {{
                    const input = r.querySelector('.gpu-weight');
                    input.value = share + (idx === 0 ? remainder : 0);
                }});
                
                updateTotal();
            }}

            function resetWeights() {{
                autoBalance();
            }}

            function updateTotal() {{
                const inputs = document.querySelectorAll('.gpu-weight');
                let sum = 0;
                inputs.forEach(i => sum += parseInt(i.value || 0));
                const badge = document.getElementById('total-percent');
                badge.innerText = `Total: ${{sum}}%`;
                
                if (sum !== 100) {{
                    badge.classList.remove('bg-slate-900');
                    badge.classList.add('bg-red-600');
                }} else {{
                    badge.classList.remove('bg-red-600');
                    badge.classList.add('bg-slate-900');
                }}
            }}

            async function updateStatus() {{
                try {{
                    const res = await fetch('/status');
                    const data = await res.json();
                    const badge = document.getElementById('status-badge');
                    const statusCard = document.getElementById('status-card');
                    const modelName = document.getElementById('current-model-name');
                    
                    if (data.running) {{
                        badge.className = 'px-6 py-3 rounded-2xl text-sm font-black flex items-center gap-3 bg-green-100 text-green-700 shadow-inner';
                        badge.innerHTML = '<div class="w-3 h-3 rounded-full bg-green-500"></div> ONLINE';
                        statusCard.classList.remove('hidden');
                        modelName.innerText = data.model;
                    }} else {{
                        badge.className = 'px-6 py-3 rounded-2xl text-sm font-black flex items-center gap-3 bg-slate-100 text-slate-500 shadow-inner';
                        badge.innerHTML = '<div class="w-3 h-3 rounded-full bg-slate-400"></div> OFFLINE';
                        statusCard.classList.add('hidden');
                    }}
                } catch (e) {{
                    console.error("Status check failed", e);
                }}
            }}

            async function startModel(path) {{
                const gpuData = [];
                document.querySelectorAll('.gpu-row').forEach(r => {{
                    if (r.querySelector('.gpu-checkbox').checked) {{
                        gpuData.push({{
                            index: parseInt(r.dataset.index),
                            weight: parseInt(r.querySelector('.gpu-weight').value || 0)
                        }});
                    }}
                }});

                if (gpuData.length === 0) {{
                    alert("⚠️ Erro: Selecione pelo menos uma GPU para carregar o modelo.");
                    return;
                }}

                const sum = gpuData.reduce((a, b) => a + b.weight, 0);
                if (sum === 0) {{
                     alert("⚠️ Erro: A distribuição de carga não pode ser 0% para todas as GPUs selecionadas.");
                     return;
                }}

                document.getElementById('status-badge').innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> INICIANDO...';
                
                try {{
                    const response = await fetch('/start', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ 
                            path: path, 
                            gpu_weights: gpuData 
                        }})
                    }});
                    if (!response.ok) throw new Error("Falha ao iniciar modelo");
                    setTimeout(updateStatus, 3000);
                }} catch (err) {{
                    alert("Erro ao iniciar o servidor: " + err.message);
                    updateStatus();
                }}
            }}

            async function stopModel() {{
                if (!confirm("⚠️ Deseja realmente encerrar o processo do servidor LLM?")) return;
                await fetch('/stop', {{method: 'POST'}});
                setTimeout(updateStatus, 1500);
            }}

            setInterval(updateStatus, 5000);
            updateStatus();
            // Inicialização opcional: autoBalance() se quiser resetar no load
        </script>
    </body>
    </html>
    """
    return html_content

class GPUWeight(BaseModel):
    index: int
    weight: int

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
        env = os.environ.copy()
        env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        
        all_gpus = get_gpu_info()
        max_idx = max(g['index'] for g in all_gpus) if all_gpus else 0
        weights = [0.0] * (max_idx + 1)
        
        total_weight = sum(gw.weight for gw in req.gpu_weights)
        if total_weight == 0:
             raise HTTPException(status_code=400, detail="Peso total inválido.")

        # Identifica a GPU principal (primeira com peso > 0)
        main_gpu = str(req.gpu_weights[0].index)
        
        for gw in req.gpu_weights:
            weights[gw.index] = gw.weight / total_weight

        tensor_split_str = ",".join([f"{w:.4f}" for w in weights])

        cmd = [
            "llama-server",
            "-m", req.path,
            "-ngl", "99",
            "--flash-attn", "on",
            "--host", "0.0.0.0",
            "--port", "8085",
            "--tools", "all",
            "--parallel", "4",
            "--ctx-size", "32768",
            "--mlock",
            "--main-gpu", main_gpu,
            "--tensor-split", tensor_split_str
        ]
        
        logging.info(f"Running command: {' '.join(cmd)}")
        
        subprocess.Popen(
            cmd, 
            stdout=open("/root/gemma_server.log", "w"), 
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=env
        )
        return {"message": "Model starting", "tensor_split": tensor_split_str}
    except Exception as e:
        logging.error(f"Start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
def stop_model():
    logging.info("Stopping all llama-server instances")
    subprocess.run(["pkill", "-9", "llama-server"])
    return {"message": "Model stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
