#!/usr/bin/env python3
import os
import socket
import subprocess
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from panel_model_manager import PanelModelManager
from model_registry import ModelRegistry

ROOT = Path(os.getenv("JETSON_FRIEND_ROOT", Path(__file__).resolve().parents[1])).resolve()
app = FastAPI(title="MILO Control")
manager = PanelModelManager()
registry = ModelRegistry()

class Selection(BaseModel):
    model_id: str

HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MILO Control</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;max-width:1000px;margin:30px auto;padding:0 16px;background:#111;color:#eee}
.card{background:#1b1b1b;border:1px solid #333;border-radius:14px;padding:18px;margin:14px 0}
button,select{padding:10px 12px;border-radius:10px;border:1px solid #444;background:#242424;color:#eee;margin:4px}
button{cursor:pointer}
pre{white-space:pre-wrap;background:#090909;padding:12px;border-radius:10px;max-height:300px;overflow:auto}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
.small{opacity:.75;font-size:.92em}
</style>
</head>
<body>
<h1>MILO Control</h1>
<div class="grid">
<div class="card">
<h2>Runtime</h2>
<div id="runtime"></div>
<button onclick="act('start')">Start</button>
<button onclick="act('stop')">Stop</button>
<button onclick="act('restart')">Restart</button>
</div>
<div class="card">
<h2>System</h2>
<div id="system"></div>
</div>
</div>

<div class="card">
<h2>Models</h2>
<div class="grid">
<div>
<h3>Text LLM</h3>
<select id="textModel"></select>
<button onclick="selectModel('text')">Select</button>
</div>
<div>
<h3>Vision VLM</h3>
<select id="vlmModel"></select>
<button onclick="selectModel('vlm')">Select</button>
</div>
</div>
<p class="small">Text and VLM are selected separately. MILO can keep the text LLM as default and load the VLM only for visual questions.</p>
</div>

<div class="card">
<h2>Model catalog</h2>
<div id="catalog"></div>
</div>

<div class="card">
<h2>Logs</h2>
<button onclick="refreshLogs()">Refresh logs</button>
<pre id="logs"></pre>
</div>

<script>
async function j(url,opts){let r=await fetch(url,opts); if(!r.ok) throw new Error(await r.text()); return await r.json()}
async function refresh(){
 const s=await j('/api/status');
 runtime.innerHTML=`MILO: <b>${s.running?'running':'stopped'}</b><br>Text: ${s.current.text_model||s.current.text_model_path}<br>VLM: ${s.current.vlm_model||s.current.vlm_model_path}`;
 system.innerHTML=`CPU ${s.cpu_percent}%<br>RAM ${s.ram_used_gb}/${s.ram_total_gb} GB<br>Host ${s.hostname}`;
 const models=await j('/api/models');
 textModel.innerHTML=''; vlmModel.innerHTML=''; catalog.innerHTML='';
 for(const m of models){
   const line=document.createElement('div');
   line.innerHTML=`<b>${m.name||m.id}</b> — ${m.type} — ${m.installed?'installed':'not installed'} ${m.notes?'<br><span class="small">'+m.notes+'</span>':''}`;
   if(!m.installed){
     const b=document.createElement('button'); b.textContent='Download'; b.onclick=()=>downloadModel(m.id); line.appendChild(b);
   }
   catalog.appendChild(line);
   if(m.type==='text'){let o=document.createElement('option');o.value=m.id;o.textContent=m.id+(m.installed?'':' (not installed)');o.disabled=!m.installed;textModel.appendChild(o)}
   if(m.type==='vision'){let o=document.createElement('option');o.value=m.id;o.textContent=m.id+(m.installed?'':' (not installed)');o.disabled=!m.installed;vlmModel.appendChild(o)}
 }
}
async function act(a){await j('/api/runtime/'+a,{method:'POST'}); setTimeout(refresh,700)}
async function selectModel(kind){let id=kind==='text'?textModel.value:vlmModel.value; await j('/api/select/'+kind,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_id:id})}); await refresh()}
async function downloadModel(id){catalog.innerHTML='Downloading '+id+'...'; await j('/api/download/'+encodeURIComponent(id),{method:'POST'}); await refresh()}
async function refreshLogs(){logs.textContent=await (await fetch('/api/logs')).text()}
refresh(); refreshLogs(); setInterval(refresh,5000);
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

@app.get("/api/status")
def status():
    vm = psutil.virtual_memory()
    return {
        "running": manager.milo_running(),
        "current": manager.current(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_gb": round((vm.total-vm.available)/(1024**3), 2),
        "ram_total_gb": round(vm.total/(1024**3), 2),
        "hostname": socket.gethostname(),
    }

@app.get("/api/models")
def models():
    return registry.list()

@app.post("/api/select/text")
def select_text(body: Selection):
    try:
        result = manager.select_text(body.model_id)
        result["restart_required"] = manager.milo_running()
        return result
    except Exception as exc:
        raise HTTPException(400, str(exc))

@app.post("/api/select/vlm")
def select_vlm(body: Selection):
    try:
        result = manager.select_vlm(body.model_id)
        result["restart_required"] = manager.milo_running()
        return result
    except Exception as exc:
        raise HTTPException(400, str(exc))

@app.post("/api/runtime/start")
def start():
    return {"result": manager.start_milo()}

@app.post("/api/runtime/stop")
def stop():
    return {"result": manager.stop_milo()}

@app.post("/api/runtime/restart")
def restart():
    return {"result": manager.restart_milo()}

@app.post("/api/download/{model_id}")
def download(model_id: str):
    result = subprocess.run(
        [str(ROOT/".venv/bin/python"), str(ROOT/"src/model_downloader.py"), "download", model_id],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise HTTPException(500, result.stderr or result.stdout)
    return {"result": result.stdout}

@app.get("/api/logs", response_class=PlainTextResponse)
def logs():
    path = ROOT/"milo_runtime.log"
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-250:])

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("SETTINGS_HOST", "0.0.0.0")
    port = int(os.getenv("SETTINGS_PORT", "8765"))
    uvicorn.run(app, host=host, port=port, log_level="info")
