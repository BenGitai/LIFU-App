"""
app.py -- web UI for the LIFU pipeline.

Local web app: point it at a folder of scans + device STLs, it auto-detects
which file is which, you pick an atlas target by name or label number, confirm
the target + transducer placement, then run the k-Wave sim and get the charts
+ the interactive 3-D pressure viewer.

    conda activate lifu
    python -m pip install flask          # once
    python app.py                        # opens http://localhost:5000

Runs the sim on the local GPU, so start it on the machine with the GPU.
Outputs go to <folder>/pipeline_out.
"""
import os, sys, csv, re, glob, uuid, threading, subprocess, webbrowser
import numpy as np, nibabel as nib
from flask import Flask, request, jsonify, send_from_directory, Response

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lifu_pipeline import resolve_files            # noqa: E402

app = Flask(__name__)
STATE = {"data": None, "out": None}
JOBS = {}
REQUIRED = ["ct", "mri", "skull", "brain", "subcort", "cort", "adapter", "base"]


def find_csv(folder, atlas):
    cands = glob.glob(os.path.join(folder, "*.csv"))
    key = "subcort" if atlas == "subcortical" else "cort"

    def ok(c):
        n = os.path.basename(c).lower()
        return key in n and (atlas == "subcortical" or "subcort" not in n)

    hits = [c for c in cands if ok(c)]
    if hits:
        return hits[0]
    for c in cands:                       # fallback: any CSV that parses to labels
        if parse_labels(c):
            return c
    return None


_NL = re.compile(r"^\s*(-?\d+)\s*:\s*(.+?)\s*$")   # "91: subthalamic_nucleus (STh)"


def parse_labels(path):
    """Handle both a plain 'number,name' CSV and a hierarchical atlas whose cells are
    'N: name (abbr)' across Level columns. Returns unique [{'num','name'}]."""
    out = {}
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            hit = False
            for cell in row:                        # format A: "N: name" inside cells
                m = _NL.match(cell)
                if m:
                    out.setdefault(int(m.group(1)), m.group(2).strip()); hit = True
            if hit:
                continue
            nums = [c for c in row if c.strip().lstrip("-").isdigit()]   # format B: number + name cols
            if nums:
                names = [c.strip() for c in row if c.strip() and not c.strip().lstrip("-").isdigit()]
                out.setdefault(int(nums[0]), max(names, key=len) if names else str(int(nums[0])))
    rows = [{"num": k, "name": v} for k, v in out.items()]
    rows.sort(key=lambda r: r["name"].lower())
    return rows


@app.route("/")
def index():
    r = Response(PAGE, mimetype="text/html")
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"   # always serve the latest UI
    return r


@app.route("/scan", methods=["POST"])
def scan():
    folder = request.get_json(force=True).get("folder", "").strip().strip('"')
    if not os.path.isdir(folder):
        return jsonify(ok=False, error="Not a folder: " + folder), 400
    F = resolve_files(folder)
    out = os.path.join(folder, "pipeline_out")
    os.makedirs(out, exist_ok=True)
    STATE["data"], STATE["out"] = folder, out
    detected = {k: (os.path.basename(v) if v else None) for k, v in F.items()}
    missing = [k for k in REQUIRED if not F.get(k)]
    return jsonify(ok=True, detected=detected, missing=missing)


@app.route("/labels")
def labels():
    folder = STATE.get("data")
    if not folder:
        return jsonify([])
    atlas = request.args.get("atlas", "subcortical")
    rows = parse_labels(find_csv(folder, atlas))
    path = resolve_files(folder).get("subcort" if atlas == "subcortical" else "cort")   # keep only present labels
    if path and rows:
        try:
            present = set(int(v) for v in np.unique(np.asarray(nib.load(path).dataobj)) if v > 0)
            rows = [r for r in rows if r["num"] in present] or rows
        except Exception:
            pass
    return jsonify(rows)


def _cmd(extra):
    return [sys.executable, os.path.join(HERE, "lifu_pipeline.py"),
            "--data", STATE["data"], "--out", STATE["out"]] + extra


def _adj(j):                    # device/sim adjustment options -> pipeline args
    return ["--atlas", j["atlas"], "--target-label", str(j["label"]), "--target-name", j.get("name", ""),
            "--target-offset", j.get("offset", "0,0,0"),
            "--freq-mhz", str(j.get("freq", 1.0)), "--diam-mm", str(j.get("diam", 19)),
            "--pressure-kpa", str(j.get("pressure", 100)), "--cycles", str(int(j.get("cycles", 5))),
            "--lens-c", str(j.get("lensc", 2500))]


@app.route("/preview", methods=["POST"])
def preview():
    j = request.get_json(force=True)
    p = subprocess.run(_cmd(["--preview"] + _adj(j)), capture_output=True, text=True)
    return jsonify(ok=p.returncode == 0, log=(p.stdout + p.stderr)[-4000:])


@app.route("/run", methods=["POST"])
def run():
    j = request.get_json(force=True)
    jid = uuid.uuid4().hex[:8]
    cmd = _cmd(["--run-sim"] + _adj(j) + ["--dx-sim", str(j.get("dx", 0.4)),
                                          "--lease-gb", str(j.get("lease", 7))])
    JOBS[jid] = {"state": "running", "log": ""}

    def worker():
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            JOBS[jid]["log"] += line
        proc.wait()
        JOBS[jid]["state"] = "done" if proc.returncode == 0 else "error"

    threading.Thread(target=worker, daemon=True).start()
    return jsonify(job=jid)


@app.route("/status/<jid>")
def status(jid):
    j = JOBS.get(jid)
    return (jsonify(state="unknown"), 404) if not j else jsonify(state=j["state"], log=j["log"][-6000:])


@app.route("/out/<path:fname>")
def out_file(fname):
    return send_from_directory(STATE["out"], fname)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>LIFU planner</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#0b0e14;color:#cdd6f4;font:14px system-ui,sans-serif}
 header{padding:16px 24px;background:#121722;border-bottom:1px solid #222a3a}
 header h1{margin:0;font-size:18px} main{max-width:1000px;margin:0 auto;padding:22px}
 .card{background:#121722;border:1px solid #222a3a;border-radius:12px;padding:18px 20px;margin-bottom:18px}
 .card.off{opacity:.45;pointer-events:none}
 h2{font-size:15px;margin:0 0 12px;color:#9db4ff}
 .det{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;font-size:12px;margin-top:10px}
 .det .k{color:#7c88a8} .ok{color:#28ff9b} .bad{color:#ff7a90}
 select,input[type=number],input[type=text]{background:#0e1320;color:#cdd6f4;border:1px solid #2a3550;border-radius:7px;padding:8px 10px;font-size:13px}
 button{background:#2d5bff;color:#fff;border:0;border-radius:8px;padding:9px 16px;font-size:14px;cursor:pointer}
 button.sec{background:#26304a} button:disabled{opacity:.4;cursor:default}
 .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 img.prev{max-width:100%;border-radius:8px;border:1px solid #222a3a;margin-top:8px}
 pre.log{background:#05070c;border:1px solid #222a3a;border-radius:8px;padding:10px;height:210px;overflow:auto;font-size:11px;white-space:pre-wrap}
 iframe{width:100%;height:640px;border:1px solid #222a3a;border-radius:10px;background:#05060a}
 .muted{color:#7c88a8;font-size:12px} a{color:#7aa2ff}
</style></head><body>
<header><h1>🧠🔊 LIFU transcranial focusing — planner</h1></header>
<main>

<div class="card" id="c1">
 <h2>1 · Data folder</h2>
 <p class="muted">Paste the full path to the folder with your scans + device STLs. Files are auto-detected; outputs go to <code>&lt;folder&gt;\pipeline_out</code>.</p>
 <div class="row"><input id="folder" type="text" style="flex:1;min-width:380px" placeholder="C:\Users\...\Bean-LIFU-Modelling-Scans">
  <button id="scanBtn">Scan folder</button></div>
 <div class="det" id="detected"></div>
 <p class="muted" id="scanmsg" style="margin:10px 0 0"></p>
</div>

<div class="card off" id="c2">
 <h2>2 · Choose target</h2>
 <div class="row">
  <label>Atlas <select id="atlas"><option value="subcortical">subcortical</option><option value="cortical">cortical</option></select></label>
  <label style="flex:1;min-width:260px">Target
   <input id="search" type="text" placeholder="type a brain-region name or label number…" list="labellist" style="width:100%">
   <datalist id="labellist"></datalist>
   <span class="muted" id="matchinfo"></span></label>
  <label>Sim voxel (mm) <input id="dx" type="number" value="0.4" step="0.05" min="0.2" style="width:80px"></label>
  <label>GPU VRAM (GB) <input id="lease" type="number" value="7" step="1" min="2" style="width:70px"></label>
 </div>
 <div class="row" style="margin-top:10px">
  <span class="muted">Target nudge (mm, scan X / Y / Z — 0 = atlas centroid; lens &amp; focus follow it):</span>
  <input id="ox" type="number" value="0" step="0.5" style="width:66px" title="X mm">
  <input id="oy" type="number" value="0" step="0.5" style="width:66px" title="Y mm">
  <input id="oz" type="number" value="0" step="0.5" style="width:66px" title="Z mm">
 </div>
 <details open style="margin-top:12px"><summary class="muted" style="cursor:pointer">Advanced device &amp; sim adjustments</summary>
  <div class="row" style="margin-top:10px">
   <label>Frequency (MHz) <input id="freq" type="number" value="1.0" step="0.1" min="0.1" style="width:74px"></label>
   <label>Aperture Ø (mm) <input id="diam" type="number" value="19" step="1" min="3" style="width:74px"></label>
   <label>Source pressure (kPa) <input id="pressure" type="number" value="100" step="10" min="1" style="width:80px"></label>
   <label>Tone-burst cycles <input id="cycles" type="number" value="5" step="1" min="1" style="width:66px"></label>
   <label>Lens material (m/s) <input id="lensc" type="number" value="2500" step="50" min="900" style="width:82px"></label>
  </div></details>
 <div class="row" style="margin-top:12px"><button id="prevBtn" class="sec">Load &amp; confirm</button></div>
</div>

<div class="card off" id="c3">
 <h2>3 · Confirm loading, target &amp; transducer placement</h2>
 <div id="confirm"></div>
 <div class="row" style="margin-top:14px">
  <button id="runBtn" disabled>✓ Looks correct — run simulation</button>
  <span class="muted" id="prevlog"></span>
 </div>
</div>

<div class="card off" id="c4"><h2>4 · Simulation</h2><pre class="log" id="log"></pre></div>
<div class="card off" id="c5"><h2>5 · Results</h2><div id="results"></div></div>

</main>
<script>
const ROLES=[["ct","CT"],["mri","MRI"],["skull","Skull mask"],["brain","Brain mask"],
 ["subcort","Subcortical atlas"],["cort","Cortical atlas"],["adapter","Adapter STL"],["base","Base STL"],
 ["skull_craniotomy","Skull-craniotomy STL (optional)"]];
const $=id=>document.getElementById(id);
function on(id){$(id).classList.remove("off");}

$("scanBtn").onclick=async()=>{
 const folder=$("folder").value.trim();
 $("scanmsg").textContent="scanning…";
 const r=await fetch("/scan",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({folder})});
 const j=await r.json();
 if(!j.ok){$("scanmsg").innerHTML='<span class="bad">'+j.error+'</span>';return;}
 const req=["ct","mri","skull","brain","subcort","cort","adapter","base"];
 $("detected").innerHTML=ROLES.map(([k,lbl])=>{
   const v=j.detected[k];const need=req.includes(k);
   const cls=v?"ok":(need?"bad":"muted");const val=v||(need?"— MISSING —":"—");
   return `<div class="k">${lbl}</div><div class="${cls}">${val}</div>`;}).join("");
 if(j.missing.length){$("scanmsg").innerHTML='<span class="bad">missing required: '+j.missing.join(", ")+'</span>';}
 else{$("scanmsg").innerHTML='<span class="ok">all required files detected ✓</span>';on("c2");loadLabels();}
};

async function loadLabels(){
 const r=await fetch("/labels?atlas="+$("atlas").value);const rows=await r.json();
 const dl=$("labellist");dl.innerHTML="";
 rows.forEach(x=>{const o=document.createElement("option");o.value=x.name+"  [#"+x.num+"]";dl.appendChild(o);});
 $("search").dataset.rows=JSON.stringify(rows);
 $("matchinfo").innerHTML=rows.length?('<span style="color:#28ff9b">'+rows.length+' regions loaded — type a name (autocompletes) or a label number</span>')
   :'<span style="color:#ffca7a">no label CSV found for this atlas — enter a label number</span>';
 showMatch();
}
$("atlas").onchange=loadLabels;
function showMatch(){const c=chosen();const v=$("search").value.trim();
 if(!v)return;
 $("matchinfo").innerHTML=c?('→ <b style="color:#28ff9b">'+(c.name||("label "+c.label))+'</b>  (label '+c.label+')')
   :'<span style="color:#ff7a90">no matching region — keep typing or use a number</span>';}
document.getElementById("search").addEventListener("input",showMatch);

function chosen(){
 const v=$("search").value.trim();const rows=JSON.parse($("search").dataset.rows||"[]");
 let m=v.match(/#(\d+)/);
 if(m)return{label:+m[1],name:(rows.find(r=>r.num==+m[1])||{}).name||""};
 if(/^\d+$/.test(v))return{label:+v,name:(rows.find(r=>r.num==+v)||{}).name||""};
 const hit=rows.find(r=>r.name.toLowerCase()===v.toLowerCase())||rows.find(r=>r.name.toLowerCase().includes(v.toLowerCase()));
 return hit?{label:hit.num,name:hit.name}:null;
}
function params(){return {freq:+$("freq").value,diam:+$("diam").value,pressure:+$("pressure").value,
 cycles:+$("cycles").value,lensc:+$("lensc").value};}

$("prevBtn").onclick=async()=>{
 const c=chosen();if(!c){alert("Pick a target (name or number).");return;}
 on("c3");$("confirm").innerHTML="<span class='muted'>running geometry (no GPU)… ~30 s</span>";$("runBtn").disabled=true;
 const off=[+$("ox").value,+$("oy").value,+$("oz").value].join(",");
 const body=Object.assign({atlas:$("atlas").value,label:c.label,name:c.name,offset:off},params());
 const r=await fetch("/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
 const j=await r.json();const t=Date.now();
 if(!j.ok){$("confirm").innerHTML="<span class='bad'>Preview failed — check the log below.</span>";
   $("prevlog").textContent=j.log.split("\n").filter(x=>x.trim()).slice(-4).join("  |  ");return;}
 $("confirm").innerHTML=`<p class="muted">Target <b>${c.name}</b> (label ${c.label}). Verify the volumes loaded, the red target is in the right structure, and the transducer sits over the craniotomy aimed at it.</p>
   <div class="row"><div style="flex:1;min-width:300px"><b>Target on MRI</b><br><img class="prev" src="/out/step02_target.png?t=${t}"></div>
   <div style="flex:1;min-width:300px"><b>Transducer placement</b><br><img class="prev" src="/out/step07_transducer.png?t=${t}"></div></div>
   <p style="margin-top:16px"><b>Interactive 3-D placement viewer</b> — drag to rotate; green = target, cyan = transducer. Confirm the transducer is aimed at the target through the skull before running.</p>
   <iframe src="/out/placement_viewer.html?t=${t}"></iframe>
   <p><a href="/out/placement_viewer.html?t=${t}" target="_blank">↗ placement viewer full-screen</a></p>
   <details style="margin-top:8px"><summary class="muted">more views</summary>
     <img class="prev" src="/out/step01_inputs.png?t=${t}"><img class="prev" src="/out/step05_hole.png?t=${t}"></details>`;
 $("runBtn").disabled=false;window._sel=body;
};

$("runBtn").onclick=async()=>{
 on("c4");$("log").textContent="starting…";
 const body=Object.assign({},window._sel,params(),{dx:+$("dx").value,lease:+$("lease").value,
   offset:[+$("ox").value,+$("oy").value,+$("oz").value].join(",")});
 const r=await fetch("/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
 poll((await r.json()).job);
};

async function poll(job){
 const j=await (await fetch("/status/"+job)).json();
 $("log").textContent=j.log;$("log").scrollTop=$("log").scrollHeight;
 if(j.state==="running"){setTimeout(()=>poll(job),1500);return;}
 if(j.state==="error"){$("log").textContent+="\n\n[failed — see above]";return;}
 const t=Date.now();on("c5");
 $("results").innerHTML=`<img class="prev" src="/out/step11_pressure.png?t=${t}">
   <p style="margin-top:16px"><b>Interactive 3-D pressure viewer</b></p>
   <iframe src="/out/pressure_viewer.html?t=${t}"></iframe>
   <p class="row" style="margin-top:14px">
     <a href="/out/lens.stl?t=${t}" download>⤓ lens.stl</a>
     <a href="/out/transducer_spec.txt?t=${t}" download>⤓ transducer_spec.txt</a>
     <a href="/out/pressure_viewer.html?t=${t}" target="_blank">↗ viewer full-screen</a></p>`;
}
</script></body></html>"""


if __name__ == "__main__":
    url = "http://localhost:5000"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"LIFU planner running at {url}")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
