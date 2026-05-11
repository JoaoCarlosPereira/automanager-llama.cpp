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
import uvicorn

# Configura log para o arquivo do gerenciador
logging.basicConfig(filename='/root/manager.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Llama.cpp Model Manager")

MODELS_DIR = "/media/docker/models"

def get_gguf_models():
    \"\"\"
    Scans the MODELS_DIR for all files with the .gguf extension.
    Returns a sorted list of absolute paths to the discovered models.
    \"\"\"
    files = glob.glob(os.path.join(MODELS_DIR, "**/*.gguf"), recursive=True)
    logging.info(f"Found {len(files)} models in {MODELS_DIR}")
    return sorted(files)

def get_gpu_info():
    \"\"\"
    Retrieves information about available NVIDIA GPUs using nvidia-smi.
    Returns a list of dictionaries containing index, name, and total VRAM.
    \"\"\"
    try:
        # Detecta GPUs usando nvidia-smi
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits"
        ]).decode()
        gpus = []
        for line in output.strip().split("\n"):
            idx, name, mem = line.split(", ")
            gpus.append({"index": int(idx), "name": name, "vram": int(mem)})

        # Ordem de detecção do llama-server pode ser diferente.
        # Geralmente segue o CUDA_VISIBLE_DEVICES ou ordem do bus.
        # No llama.cpp compilado com CUDA, 3090 foi Device 0 nos logs anteriores.
        return gpus
    except Exception as e:
        logging.error(f"Error getting GPU info: {e}")
        return []

def find_llama_server():
    \"\"\"
    Searches for a running 'llama-server' process.
    If found, extracts the model name from the command line arguments.
    Returns a dictionary indicating if it's running, the PID, and the model name.
    \"\"\"
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Verifica se o processo é llama-server
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
    \"\"\"
    Renders the main dashboard of the Model Manager.
    Displays available GPU hardware, allows setting the tensor split ratio,
    and lists discovered GGUF models with options to start them.
    \"\"\"
    models = get_gguf_models()
    gpus = get_gpu_info()
    
    # Prioridade solicitada pelo usuário: 95% na 3090, 5% na P100
    # No llama-server, 3090 é o Device 0 e P100 é o Device 1
    suggested_split = "0.95,0.05"

    model_list_html = ""
    for m in models:
        m_name = os.path.basename(m)
        m_dir = os.path.dirname(m).replace(MODELS_DIR, "")
        model_list_html += f"""
        <div class="model-item">
            <div>
                <strong>{m_name}</strong><br>
                <small style="color: #666">{m_dir}</small>
            </div>
            <div class="actions">
                <button class="start" onclick="startModel('{m}')">Start</button>
            </div>
        </div>
        """
    
    gpu_list_html = ""
    for g in gpus:
        is_3090 = " (Main GPU Candidate)" if "3090" in g["name"] else ""
        gpu_list_html += f"<li><strong>GPU {g['index']}</strong>: {g['name']} - {g['vram']} MB{is_3090}</li>"

    html_content = f"""
    <html>
        <head>
            <title>Llama.cpp Manager</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; padding: 2rem; color: #333; }}
                .container {{ max-width: 900px; margin: auto; background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }}
                h1 {{ color: #1a73e8; border-bottom: 2px solid #e8f0fe; padding-bottom: 1rem; }}
                #status-box {{ background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; text-align: center; border: 1px solid #dee2e6; }}
                #status {{ font-size: 1.2rem; margin-bottom: 1rem; }}
                .active {{ color: #28a745; font-weight: bold; border: 2px solid #28a745; padding: 0.2rem 0.5rem; border-radius: 4px; }}
                .config-section {{ background: #f1f3f4; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; }}
                .model-item {{ display: flex; justify-content: space-between; align-items: center; padding: 1rem; border-bottom: 1px solid #eee; transition: background 0.2s; }}
                .model-item:hover {{ background: #f8f9fa; }}
                button {{ padding: 0.6rem 1.2rem; cursor: pointer; border: none; border-radius: 6px; font-weight: 600; transition: all 0.2s; }}
                .start {{ background: #1a73e8; color: white; }}
                .start:hover {{ background: #1557b0; transform: translateY(-1px); }}
                .stop {{ background: #d93025; color: white; width: 100%; }}
                .stop:hover {{ background: #a50e0e; }}
                input {{ width: 100%; padding: 0.8rem; border: 1px solid #dadce0; border-radius: 6px; margin: 0.5rem 0 1rem 0; font-size: 1rem; }}
                .gpu-list {{ list-style: none; padding: 0; margin: 1rem 0; }}
                .gpu-list li {{ padding: 0.5rem; background: white; margin-bottom: 0.5rem; border-radius: 4px; border-left: 4px solid #1a73e8; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Llama.cpp Model Manager</h1>
                
                <div id="status-box">
                    <div id="status">Detecting status...</div>
                    <button id="stop-btn" class="stop" style="display:none" onclick="stopModel()">STOP SERVER</button>
                </div>

                <div class="config-section">
                    <h3>GPU Hardware & Split</h3>
                    <ul class="gpu-list">
                        {gpu_list_html}
                    </ul>
                    <label><strong>Tensor Split Ratio:</strong></label>
                    <input type="text" id="tensor_split" value="{suggested_split}">
                    <div style="display: flex; gap: 10px;">
                        <button style="background: #5f6368; color: white; flex: 1;" onclick="setSplit('{suggested_split}')">Balance by VRAM</button>
                        <button style="background: #1a73e8; color: white; flex: 1;" onclick="setSplit('0,1')">Prioritize RTX 3090 (if GPU 1)</button>
                        <button style="background: #1a73e8; color: white; flex: 1;" onclick="setSplit('1,0')">Prioritize RTX 3090 (if GPU 0)</button>
                    </div>
                    <p><small>* Note: llama-server order may vary. Check console logs if the split fails.</small></p>
                </div>

                <h3>Available Models (.gguf)</h3>
                <div id="model-list">
                    {model_list_html}
                </div>
                
                <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee; font-size: 0.9rem; color: #70757a;">
                    API: <code>http://{psutil.net_if_addrs().get('eth0', [type('obj', (object,), {'address': 'localhost'})])[0].address}:8085/v1</code>
                </div>
            </div>

            <script>
                async function updateStatus() {{
                    try {{
                        const res = await fetch('/status');
                        const data = await res.json();
                        const statusDiv = document.getElementById('status');
                        const stopBtn = document.getElementById('stop-btn');
                        
                        if (data.running) {{
                            statusDiv.innerHTML = 'Current Model: <span class="active">' + data.model + '</span>';
                            stopBtn.style.display = 'block';
                        }} else {{
                            statusDiv.innerHTML = 'Status: <span style="color: #d93025">Offline</span>';
                            stopBtn.style.display = 'none';
                        }}
                    }} catch (e) {{
                        console.error("Status check failed", e);
                    }}
                }}

                function setSplit(val) {{
                    document.getElementById('tensor_split').value = val;
                }}

                async function startModel(path) {{
                    const split = document.getElementById('tensor_split').value;
                    document.getElementById('status').innerHTML = '<i>Starting model... please wait...</i>';
                    await fetch('/start', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ path: path, tensor_split: split }})
                    }});
                    setTimeout(updateStatus, 2000);
                }}

                async function stopModel() {{
                    document.getElementById('status').innerText = 'Stopping server...';
                    await fetch('/stop', {{method: 'POST'}});
                    setTimeout(updateStatus, 1500);
                }}

                setInterval(updateStatus, 3000);
                updateStatus();
            </script>
        </body>
    </html>
    """
    return html_content

class StartRequest(BaseModel):
    path: str
    tensor_split: str = None

@app.get("/status")
def get_status():
    return find_llama_server()

@app.post("/start")
def start_model(req: StartRequest):
    \"\"\"
    Stops any currently running llama-server instance and starts a new one
    with the specified model path and tensor split ratio.
    \"\"\"
    stop_model()
    try:
        env = os.environ.copy()
        env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        
        # Identifica a GPU principal (RTX 3090)
        # Nos logs anteriores a 3090 era o Device 0
        main_gpu = "0"
        gpus = get_gpu_info()
        for g in gpus:
            if "3090" in g["name"]:
                main_gpu = str(g["index"])
                break

        cmd = [
            "llama-server",
            "-m", req.path,
            "-ngl", "99",
            "--flash-attn", "on",
            "--host", "0.0.0.0",
            "--port", "8085",
            "--tools", "all",
            "--parallel", "1",
            "--ctx-size", "32768",
            "--mlock",
            "--main-gpu", main_gpu
        ]
        
        if req.tensor_split and "," in req.tensor_split:
            cmd.extend(["--tensor-split", req.tensor_split])
        
        logging.info(f"Running command: {' '.join(cmd)}")
        
        subprocess.Popen(
            cmd, 
            stdout=open("/root/gemma_server.log", "w"), 
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=env
        )
        return {"message": "Model starting"}
    except Exception as e:
        logging.error(f"Start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
def stop_model():
    \"\"\"
    Terminates all running llama-server processes using pkill.
    \"\"\"
    logging.info("Stopping all llama-server instances")
    subprocess.run(["pkill", "-9", "llama-server"])
    return {"message": "Model stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
