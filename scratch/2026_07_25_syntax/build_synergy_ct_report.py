#!/usr/bin/env python
"""Browse-by-cell-type synergy report (self-contained interactive HTML), DeepLIFT-passing pairs only.
Pick a cell type -> its synergistic motif pairs (Δ>0.504 & maxZ>5.38 vs lineage-mismatched null, and
DeepLIFT frac_in >= 0.4). Click a pair -> detail: two motif logos, orientation × spacing ΔJ landscape
(measured in the pair's primary cell type), per-orientation distance curves, 22-cell-type Δ strip, and
a cell-type-specific stats table. Data-forward, no prose/verdicts.

Runs locally (CPU). Reads figures/synergistic_pairs_allCT_long.tsv, RES/{full,subs,candidate}/*.npz,
metadata.tsv, cell_type_metadata.tsv, logos_cwm/.
"""
import os, glob, json, base64, html
import numpy as np, pandas as pd

SYN = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(SYN, "figures")
OUT = os.path.join(SYN, "synergy_by_celltype_report.html")

def b64png(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else ""
def q(x, s=1000):
    return int(round(float(x) * s)) if pd.notna(x) and np.isfinite(x) else 0

# ---- cell-type palette + developmental order ----
_ctm = pd.read_csv(os.path.join(SYN, "cell_type_metadata.tsv"), sep="\t").sort_values("display_order")
CTCOL = dict(zip(_ctm["cell_type"], _ctm["color"]))
CTORDER = list(_ctm["cell_type"])

# ---- long table: 582 synergistic pairs × 22 CT (Δ re-optimized per CT) ----
lng = pd.read_csv(os.path.join(FIG, "synergistic_pairs_allCT_long.tsv"), sep="\t")
tf = {}
for r in lng.itertuples():
    tf[r.idA] = r.tfA; tf[r.idB] = r.tfB

# rows that populate the table: synergistic in that CT AND DeepLIFT-passing
sel = lng[(lng.synergistic_in_ct == True) & (~lng.deeplift_flag.astype(str).eq("artifact?"))].copy()
print(f"table rows (CT,pair) synergistic + DeepLIFT-pass: {len(sel)}")

# ---- per-pair grid (orientation × spacing) from the primary-CT NPZ ----
def grid_for(idA, idB, ct):
    for sub in ("full", "subs", "candidate"):
        p = f"{SYN}/RES/{sub}/{idA}__{idB}__{ct}.npz"
        if os.path.exists(p):
            return np.load(p, allow_pickle=True)
    return None

pairs = lng[["idA", "idB", "primary_ct"]].drop_duplicates()
DATA = {}
logos_used = set()
for r in pairs.itertuples():
    key = f"{r.idA}__{r.idB}"
    z = grid_for(r.idA, r.idB, r.primary_ct)
    if z is None:
        continue
    dJ = np.asarray(z["dJ"], float); gaps = np.asarray(z["gaps"], int)
    orients = [str(x) for x in z["orients"]]
    dS = float(z["dA"]) + float(z["dB"])
    # 22-CT Δ strip for this pair (all CTs from the long table, synergistic or not)
    sub = lng[(lng.idA == r.idA) & (lng.idB == r.idB)]
    cts = {rr.cell_type: round(float(rr.delta), 3) for rr in sub.itertuples() if pd.notna(rr.delta)}
    dlv = sub["deeplift_frac_in"].dropna()
    DATA[key] = {
        "tfA": tf.get(r.idA, r.idA), "tfB": tf.get(r.idB, r.idB), "idA": r.idA, "idB": r.idB,
        "pct": r.primary_ct, "o": orients, "g0": int(gaps[0]),
        "gs": int(gaps[1] - gaps[0]) if len(gaps) > 1 else 1, "gn": len(gaps),
        "dJ": [[q(v) for v in row] for row in dJ], "dS": q(dS), "dA": q(z["dA"]), "dB": q(z["dB"]),
        "oo": str(z["opt_orient"]), "og": int(z["opt_gap"]), "wa": int(z["dJ"].shape[0] and 0) or None,
        "cts": cts, "dl": round(float(dlv.iloc[0]), 3) if len(dlv) else None,
    }
    logos_used.add(r.idA); logos_used.add(r.idB)
# motif widths from the long table (wa/wb)
wmap = {}
for r in lng.itertuples():
    wmap[f"{r.idA}__{r.idB}"] = (int(r.wa), int(r.wb))
for k in DATA:
    DATA[k]["wa"], DATA[k]["wb"] = wmap.get(k, (0, 0))
print(f"pairs embedded: {len(DATA)}")

# ---- logos ----
LOGOS = {}
for st in logos_used:
    u = b64png(f"{SYN}/logos_cwm/{st}/{st}.fwd.png")
    if u: LOGOS[st] = u
print(f"logos embedded: {len(LOGOS)}/{len(logos_used)}")

# ---- per-(CT,pair) CT-specific context for the detail stats ----
def fnum(x, d=3):
    return f"{x:.{d}f}" if pd.notna(x) and np.isfinite(x) else ""
CTX = {}   # "ct|key" -> CT-specific numbers
rows_html = []
counts = {ct: {"n": 0, "hard": 0} for ct in CTORDER}
for r in sel.sort_values(["cell_type", "delta"], ascending=[True, False]).itertuples():
    key = f"{r.idA}__{r.idB}"
    if key not in DATA: continue
    ck = f"{r.cell_type}|{key}"
    CTX[ck] = {"dlt": q(r.delta), "djo": q(r.dJ_opt), "dS": q(r.dS), "oo": str(r.opt_orient),
               "og": int(r.opt_gap), "cc": q(r.opt_center_center), "z": q(r.maxZ),
               "p": float(r.wilcoxon_p), "hard": bool(r.hard_in_ct)}
    counts[r.cell_type]["n"] += 1
    counts[r.cell_type]["hard"] += int(bool(r.hard_in_ct))
    hardcls = " hard" if r.hard_in_ct else ""
    rows_html.append(
        f'<tr data-id="{key}" data-ct="{r.cell_type}" onclick="showPair(this)" style="display:none">'
        f'<td class="id">{r.idA}</td><td class="tf">{html.escape(str(r.tfA))}</td>'
        f'<td class="id">{r.idB}</td><td class="tf">{html.escape(str(r.tfB))}</td>'
        f'<td class="num d">{fnum(r.delta)}</td>'
        f'<td class="num">{fnum(r.dJ_opt)}</td><td class="num">{fnum(r.dS)}</td>'
        f'<td>{r.opt_orient}</td><td class="num">{int(r.opt_gap)}</td>'
        f'<td class="num">{fnum(r.opt_center_center,1)}</td>'
        f'<td class="num z">{fnum(r.maxZ,2)}</td>'
        f'<td class="num p">{r.wilcoxon_p:.1e}</td>'
        f'<td class="flag{hardcls}">{"hard" if r.hard_in_ct else ""}</td>'
        f'<td class="num">{fnum(r.deeplift_frac_in,2)}</td>'
        f'</tr>')
TBODY = "\n".join(rows_html)

# CT selector buttons + counts
CTBTN = "".join(
    f'<button class="ctbtn" data-ct="{ct}" style="--cc:{CTCOL[ct]}" onclick="pickCT(\'{ct}\')">'
    f'{ct} <span class="cn">{counts[ct]["n"]}</span></button>' for ct in CTORDER)

HEADERS = ["motif A", "TF A", "motif B", "TF B", "Δ", "ΔJ", "ΔS", "orient", "gap", "cen-cen",
           "maxZ", "Wilcoxon p", "syntax", "DeepLIFT"]
DEFS = {
 "motif A": "Catalog motif ID (ST##) for the upstream insert; its consensus sequence is what gets inserted.",
 "TF A": "Curator-assigned transcription factor for motif A.",
 "motif B": "Catalog motif ID for the downstream insert.",
 "TF B": "Curator-assigned transcription factor for motif B.",
 "Δ": "ΔJ − ΔS in this cell type = synergy (joint minus additive expectation). Must exceed the cell type's threshold to be called synergistic here.",
 "ΔJ": "Joint effect: mean increase in predicted ln-counts with BOTH motifs inserted, at the optimal arrangement.",
 "ΔS": "Additive expectation = ΔA + ΔB (each motif inserted alone).",
 "orient": "Orientation at the optimal arrangement. FF = A→ B→ · FR = A→ ←B (convergent) · RF = ←A B→ (divergent) · RR = ←A ←B. Motif A is always the upstream insert.",
 "gap": "Inner-edge spacing (bp) between the two motifs at the optimal arrangement (0 = abutting).",
 "cen-cen": "Center-to-center distance (bp) at the optimal arrangement.",
 "maxZ": "Z-score of the best arrangement's ΔJ vs the mean over all arrangements. High = sharp spacing/orientation preference (hard syntax).",
 "Wilcoxon p": "Signed-rank test: joint vs independent (A+B) effect across the background sequences, at the optimal arrangement.",
 "syntax": "hard = maxZ above the cell type's null-calibrated cutoff (sharp preference); otherwise soft.",
 "DeepLIFT": "Fraction of central (±100 bp) attribution landing on the two inserted motifs (≥0.4 required to appear here; near 1 = model reads the inserted motifs).",
}
NUMCOLS = {4, 5, 6, 8, 9, 10, 11, 13}
def _th(i, h):
    label = f'<div class="thl" title="{html.escape(DEFS[h])}" onclick="sortT({i})">{h}</div>'
    ph = "&gt;n" if i in NUMCOLS else "filter"
    inp = f'<input class="colf" data-col="{i}" placeholder="{ph}" oninput="filt()" onclick="event.stopPropagation()">'
    return f"<th>{label}{inp}</th>"
THEAD_CELLS = "".join(_th(i, h) for i, h in enumerate(HEADERS))

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:14px;color:#222}
h1{margin:0 0 2px;color:#3b2d7e;border-bottom:2px solid #3b2d7e;padding-bottom:4px;font-size:22px}
h2{color:#3b2d7e;font-size:15px;margin:18px 0 6px}
.sub{color:#777;font-size:13px;margin:2px 0 10px}
a{color:#1e6091}
.ctsel{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}
.ctbtn{font-size:11.5px;padding:4px 9px;border:1.5px solid var(--cc);border-left:6px solid var(--cc);
  background:#fff;color:#333;border-radius:5px;cursor:pointer;font-weight:600}
.ctbtn .cn{color:#888;font-weight:400;font-size:10px;margin-left:3px}
.ctbtn.sel{background:var(--cc);color:#fff}.ctbtn.sel .cn{color:#eee}
.ctbtn:hover{filter:brightness(.96)}
#cthead{font-size:15px;font-weight:700;margin:12px 0 2px}
#ctmeta{font-size:12px;color:#777;margin-bottom:6px}
#f{padding:5px 9px;font-size:13px;width:260px;margin:6px 0}
.wrap{overflow:auto;max-height:66vh;border:1px solid #e5e7eb;border-radius:4px}
table{border-collapse:collapse;font-size:12px}
th,td{border:1px solid #e5e7eb;padding:3px 6px;white-space:nowrap;vertical-align:middle}
th{background:#eef;position:sticky;top:0;z-index:3;font-size:10.5px;box-shadow:0 1px 0 #cdd3e6;vertical-align:top}
.thl{cursor:pointer}.thl:hover{text-decoration:underline}
.colf{width:100%;box-sizing:border-box;font-size:10px;padding:1px 3px;margin-top:2px;border:1px solid #cdd3e6;border-radius:3px;font-weight:400;font-family:inherit}
.colf:focus{outline:1px solid #3b2d7e}
tbody tr{cursor:pointer}tbody tr:hover{background:#eef4fb}tbody tr.sel{background:#dce8f6}
td.id{font-family:ui-monospace,Menlo,monospace;color:#7a5;font-size:11px}
td.tf{color:#1e6091;font-weight:600}.ct{color:#666;font-size:11px}
.num{text-align:right;font-variant-numeric:tabular-nums}
td.d{font-weight:600}.flag{text-align:center;font-weight:600;font-size:10px;color:#aaa}
.flag.hard{color:#fff;background:#c0392b;border-radius:3px}
#backdrop{position:fixed;inset:0;background:rgba(20,16,40,.4);z-index:49;display:none}
#backdrop.show{display:block}
#detail{position:fixed;inset:26px;z-index:50;overflow:auto;background:#fff;border:2px solid #3b2d7e;
  border-radius:10px;box-shadow:0 12px 44px rgba(0,0,0,.32);padding:14px 18px;display:none}
#detail.show{display:block}
.dhead{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.dhead .pid{font-family:ui-monospace,monospace;font-size:12px;color:#7a5}
.dhead .close{margin-left:auto;cursor:pointer;color:#666;font-size:22px;font-weight:700;line-height:1;padding:0 6px;border:1px solid #ddd;border-radius:5px}
.dhead .close:hover{background:#f3f4f6}
.dgrid{display:grid;grid-template-columns:300px 1fr;gap:22px;margin-top:12px;align-items:start}
@media(max-width:820px){.dgrid{grid-template-columns:1fr}}
.dright{display:flex;flex-direction:column;gap:20px}
.dplot{border:1px solid #eef0f7;border-radius:8px;padding:10px 14px;background:#fcfcff}
.dph{font-size:13px;color:#3b2d7e;font-weight:600;margin-bottom:6px}
.dleft{position:sticky;top:0}
.logos{display:flex;flex-direction:column;gap:10px}
.logos .lb{border:1px solid #e5e7eb;border-radius:5px;padding:6px;background:#fff}
.logos img{width:100%;height:auto}.logos .cap{font-size:11px;color:#1e6091;font-weight:600;margin-bottom:3px}
canvas{max-width:100%;border:1px solid #eee;background:#fff}
.stats-tbl{border-collapse:collapse;font-size:12px;margin-top:6px}
.stats-tbl th{background:#eef;color:#3b2d7e;text-align:left;position:static;padding:2px 8px;border:1px solid #e5e7eb}
.stats-tbl td{padding:2px 8px}
.toggle{font-size:11px;color:#3b2d7e;cursor:pointer;user-select:none;margin:4px 0}
.toggle input{vertical-align:middle}
.note{font-size:11px;color:#888;margin-top:4px}
.ochip{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;margin-right:5px;border-radius:12px;border:1.5px solid var(--oc);color:var(--oc);cursor:pointer;user-select:none}
.ochip.off{opacity:.32;text-decoration:line-through}
#tip{position:fixed;display:none;z-index:60;background:#1c1a17;color:#fff;font-size:11px;line-height:1.5;padding:6px 9px;border-radius:5px;pointer-events:none;box-shadow:0 3px 12px rgba(0,0,0,.35);white-space:nowrap}
#curveleg{font-size:11px;margin:4px 0}
details.key{background:#f7f6fb;border:1px solid #e0dcef;border-radius:6px;padding:6px 12px;margin:8px 0}
details.key summary{cursor:pointer;font-weight:600;color:#3b2d7e;font-size:13px}
.keytbl{border-collapse:collapse;font-size:12px;margin:8px 0}
.keytbl td{border:1px solid #e5e7eb;padding:3px 8px;vertical-align:top}
.keytbl td:first-child{font-weight:600;color:#3b2d7e;white-space:nowrap;font-family:ui-monospace,monospace}
.pctag{display:inline-block;font-size:10px;background:#eef;color:#3b2d7e;border-radius:3px;padding:1px 6px;margin-left:6px}
"""

JS = """
const DATA=__DATA__, LOGOS=__LOGOS__, CTX=__CTX__, CTORDER=__CTORDER__, CTCOL=__CTCOL__;
const OCOL={FF:"#c0392b",FR:"#2980b9",RF:"#27ae60",RR:"#8e44ad"};
let SELCT=CTORDER[0], CURKEY=null, CURCTX=null, XMODE='gap';
function gaps(d){return Array.from({length:d.gn},(_,i)=>d.g0+i*d.gs);}
function pickCT(ct){SELCT=ct;
  document.querySelectorAll('.ctbtn').forEach(b=>b.classList.toggle('sel',b.dataset.ct===ct));
  const n=[...document.querySelectorAll('#t tbody tr')].filter(r=>r.dataset.ct===ct).length;
  document.getElementById('cthead').textContent=ct;
  document.getElementById('ctmeta').textContent=`${n} synergistic, DeepLIFT-passing motif pairs in this cell type (sorted by Δ).`;
  document.getElementById('f').value='';document.querySelectorAll('.colf').forEach(x=>x.value='');
  filt();}
let ftimer;
function filt(){clearTimeout(ftimer);ftimer=setTimeout(()=>{
  const gq=document.getElementById('f').value.toLowerCase();
  const cfs=[...document.querySelectorAll('.colf')].map(x=>({col:+x.dataset.col,v:x.value.trim()})).filter(c=>c.v);
  for(const tr of document.querySelectorAll('#t tbody tr')){
    let vis=(tr.dataset.ct===SELCT);
    if(vis&&gq){const d=DATA[tr.dataset.id];vis=(d.tfA+' '+d.tfB+' '+d.idA+' '+d.idB).toLowerCase().includes(gq);}
    if(vis)for(const c of cfs){if(!matchCol(tr.cells[c.col].textContent,c.v)){vis=false;break;}}
    tr.style.display=vis?'':'none';}},90);}
function matchCol(cell,f){
  const m=f.match(/^(>=|<=|>|<|=)\\s*(-?\\d*\\.?\\d+(?:[eE]-?\\d+)?)$/);
  if(m){const v=parseFloat(cell);if(isNaN(v))return false;const n=parseFloat(m[2]);
    return m[1]==='>'?v>n:m[1]==='<'?v<n:m[1]==='>='?v>=n:m[1]==='<='?v<=n:v===n;}
  return cell.toLowerCase().includes(f.toLowerCase());}
let asc={};function sortT(i){let tb=document.querySelector('#t tbody'),rs=[...tb.rows];asc[i]=!asc[i];
  rs.sort((a,b)=>{let x=a.cells[i].innerText.trim(),y=b.cells[i].innerText.trim();
  let nx=parseFloat(x),ny=parseFloat(y);let c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc[i]?c:-c;});
  rs.forEach(r=>tb.appendChild(r));}
function showPair(tr){
  document.querySelectorAll('#t tbody tr.sel').forEach(e=>e.classList.remove('sel'));tr.classList.add('sel');
  CURKEY=tr.dataset.id;CURCTX=CTX[tr.dataset.ct+'|'+tr.dataset.id];render(tr.dataset.ct);
  document.getElementById('detail').classList.add('show');document.getElementById('backdrop').classList.add('show');
  document.getElementById('detail').scrollTop=0;}
function closeD(){document.getElementById('detail').classList.remove('show');document.getElementById('backdrop').classList.remove('show');}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeD();});
function xvals(d){const g=gaps(d);return XMODE==='gap'?g:g.map(k=>k+(d.wa+d.wb)/2);}
let VIS={},HGEO=null,CGEO=null;
function render(ct){
  const d=DATA[CURKEY];
  document.getElementById('dtitle').textContent=`${d.tfA} × ${d.tfB}`;
  document.getElementById('dpid').innerHTML=`${d.idA} × ${d.idB} · in <b>${ct}</b>`;
  document.getElementById('logoA').src=LOGOS[d.idA]||'';document.getElementById('logoB').src=LOGOS[d.idB]||'';
  document.getElementById('capA').textContent=`${d.tfA}  (${d.idA})`;document.getElementById('capB').textContent=`${d.tfB}  (${d.idB})`;
  VIS={};d.o.forEach(o=>VIS[o]=true);
  document.getElementById('curveleg').innerHTML=d.o.map(o=>`<span class="ochip" data-o="${o}" style="--oc:${OCOL[o]||'#666'}" onclick="togO('${o}')">${o}</span>`).join('')+' <span class=note style="display:inline">(click to toggle)</span>';
  document.getElementById('landct').textContent=d.pct;
  drawHeat(d);drawCurves(d);stats(ct);ctstrip(d,ct);}
function togO(o){VIS[o]=!VIS[o];document.querySelector('.ochip[data-o="'+o+'"]').classList.toggle('off',!VIS[o]);drawCurves(DATA[CURKEY]);}
function div(t){if(t>=0){return `rgb(255,${Math.round(255-t*175)},${Math.round(255-t*195)})`;}
  const tt=-t;return `rgb(${Math.round(255-tt*180)},${Math.round(255-tt*140)},255)`;}
function drawHeat(d){
  const cv=document.getElementById('heat'),ctx=cv.getContext('2d');
  const no=d.o.length,ng=d.gn,pad=54,ch=38,W=Math.max(360,cv.parentElement.clientWidth-4),cw=Math.max(2,(W-pad-10)/ng);
  cv.width=W;cv.height=24+no*ch+24;ctx.clearRect(0,0,cv.width,cv.height);
  const dS=d.dS/1000;let amax=0.001;for(const row of d.dJ)for(const v of row)amax=Math.max(amax,Math.abs(v/1000-dS));
  for(let o=0;o<no;o++){ctx.fillStyle='#333';ctx.font='11px sans-serif';ctx.textAlign='right';ctx.fillText(d.o[o],pad-6,24+o*ch+ch/2+3);
    for(let g=0;g<ng;g++){const v=d.dJ[o][g]/1000-dS,t=Math.max(-1,Math.min(1,v/amax));ctx.fillStyle=div(t);ctx.fillRect(pad+g*cw,24+o*ch,cw,ch-2);}}
  ctx.fillStyle='#333';ctx.textAlign='center';ctx.font='10px sans-serif';
  const g=gaps(d);for(let k=0;k<ng;k+=Math.ceil(ng/8))ctx.fillText(g[k],pad+k*cw+cw/2,cv.height-6);
  ctx.fillText('inner-edge gap (bp) →  color = ΔJ − ΔS (red synergy, blue below additive)',pad+ng*cw/2,14);
  HGEO={pad,cw,ch,ng,no};}
function drawCurves(d){
  const cv=document.getElementById('curve'),ctx=cv.getContext('2d');
  cv.width=Math.max(360,cv.parentElement.clientWidth-4);cv.height=460;ctx.clearRect(0,0,cv.width,cv.height);
  const X=xvals(d),dS=d.dS/1000;let ymin=dS,ymax=dS;for(const row of d.dJ)for(const v of row){const y=v/1000;ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);}
  const pad=46,pb=34,W=cv.width-pad-10,H=cv.height-24-pb,x0=Math.min(...X),x1=Math.max(...X);
  const px=x=>pad+(x-x0)/((x1-x0)||1)*W,py=y=>24+H-(y-ymin)/((ymax-ymin)||1)*H;
  ctx.strokeStyle='#ccc';ctx.beginPath();ctx.moveTo(pad,24);ctx.lineTo(pad,24+H);ctx.lineTo(pad+W,24+H);ctx.stroke();
  ctx.strokeStyle='#333';ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(pad,py(dS));ctx.lineTo(pad+W,py(dS));ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#333';ctx.font='10px sans-serif';ctx.textAlign='left';ctx.fillText('ΔS additive',pad+4,py(dS)-3);
  d.o.forEach((on,o)=>{if(!VIS[on])return;ctx.strokeStyle=OCOL[on]||'#666';ctx.lineWidth=1.7;ctx.beginPath();
    d.dJ[o].forEach((v,g)=>{const xx=px(X[g]),yy=py(v/1000);g?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy);});ctx.stroke();});
  const oi=d.o.indexOf(d.oo),gi=Math.round((d.og-d.g0)/d.gs);
  if(oi>=0&&VIS[d.oo]){ctx.fillStyle='#000';ctx.beginPath();ctx.arc(px(X[gi]),py(d.dJ[oi][gi]/1000),3.8,0,7);ctx.fill();}
  ctx.fillStyle='#333';ctx.textAlign='center';ctx.fillText(XMODE==='gap'?'inner-edge gap (bp)':'center-to-center (bp)',pad+W/2,cv.height-6);
  ctx.save();ctx.translate(12,24+H/2);ctx.rotate(-Math.PI/2);ctx.fillText('ΔJ (log counts)',0,0);ctx.restore();
  CGEO={pad,W,x0,x1,X,px,py};}
function tip(show,x,y,htm){const t=document.getElementById('tip');if(!show){t.style.display='none';return;}
  t.innerHTML=htm;t.style.display='block';t.style.left=(x+14)+'px';t.style.top=(y+12)+'px';}
function cxy(e){const r=e.target.getBoundingClientRect();return [(e.clientX-r.left)*(e.target.width/r.width),(e.clientY-r.top)*(e.target.height/r.height)];}
function curveHover(e){const d=DATA[CURKEY];if(!d||!CGEO)return;const [mx]=cxy(e);const g=CGEO;
  let xi=0,best=1e9;for(let i=0;i<g.X.length;i++){const xx=g.px(g.X[i]);if(Math.abs(xx-mx)<best){best=Math.abs(xx-mx);xi=i;}}
  const gp=gaps(d)[xi],cc=gp+(d.wa+d.wb)/2;
  const lines=d.o.filter(o=>VIS[o]).map(o=>`<span style="color:${OCOL[o]||'#666'}">${o}</span>: ${(d.dJ[d.o.indexOf(o)][xi]/1000).toFixed(3)}`);
  tip(true,e.clientX,e.clientY,`gap ${gp} bp · cc ${cc.toFixed(0)} bp · ΔS ${(d.dS/1000).toFixed(3)}<br>`+lines.join('<br>'));}
function heatHover(e){const d=DATA[CURKEY];if(!d||!HGEO)return;const [mx,my]=cxy(e);const g=HGEO;
  const o=Math.floor((my-24)/g.ch),gi=Math.floor((mx-g.pad)/g.cw);
  if(o<0||o>=g.no||gi<0||gi>=g.ng){tip(false);return;}
  const dj=d.dJ[o][gi]/1000,gp=gaps(d)[gi];
  tip(true,e.clientX,e.clientY,`<span style="color:${OCOL[d.o[o]]||'#666'}">${d.o[o]}</span> · gap ${gp} bp<br>ΔJ ${dj.toFixed(3)} · ΔJ−ΔS ${(dj-d.dS/1000).toFixed(3)}`);}
function stats(ct){
  const d=DATA[CURKEY],x=CURCTX;
  const rows=[['Δ = ΔJ − ΔS',(x.dlt/1000).toFixed(3)],['ΔJ (optimal)',(x.djo/1000).toFixed(3)],['ΔS = ΔA+ΔB',(x.dS/1000).toFixed(3)],
    ['optimal orientation',x.oo],['optimal gap (bp)',x.og],['center-to-center (bp)',(x.cc/1000).toFixed(1)],
    ['max Z (arrangements)',(x.z/1000).toFixed(2)],['Wilcoxon p',x.p.toExponential(2)],
    ['syntax',x.hard?'hard':'soft'],['motif widths (A/B)',d.wa+' / '+d.wb],
    ['DeepLIFT frac on motifs',d.dl!=null?d.dl.toFixed(2):'—']];
  document.getElementById('stbl').innerHTML=`<tr><th>quantity — in ${ct}</th><th>value</th></tr>`+rows.map(r=>`<tr><td>${r[0]}</td><td class=num>${r[1]}</td></tr>`).join('');}
let CSGEO=null;
function ctstrip(d,selct){
  const cv=document.getElementById('ctstrip'),note=document.getElementById('ctstripnote');
  cv.style.display='block';cv.width=Math.max(340,cv.parentElement.clientWidth-4);cv.height=252;
  const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  const order=CTORDER.filter(c=>c in d.cts),vals=order.map(c=>d.cts[c]),mx=Math.max(0.01,...vals.map(Math.abs));
  const padL=6,padT=10,padB=112,W=cv.width-padL-6,H=cv.height-padT-padB,zero=padT+H/2,bw=W/order.length;
  order.forEach((c,i)=>{const x=padL+i*bw;if(c===selct){ctx.fillStyle='rgba(17,17,17,0.08)';ctx.fillRect(x,padT,Math.max(1,bw),H);}});
  ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(padL,zero);ctx.lineTo(padL+W,zero);ctx.stroke();
  order.forEach((c,i)=>{const v=vals[i],bh=Math.abs(v)/mx*(H/2),x=padL+i*bw;
    ctx.fillStyle=CTCOL[c]||'#888';ctx.fillRect(x+0.5,v>=0?zero-bh:zero,Math.max(1,bw-1),bh);
    ctx.save();ctx.translate(x+bw/2+3,padT+H+5);ctx.rotate(Math.PI/2);ctx.textAlign='left';ctx.font=(c===selct?'bold ':'')+'8.5px sans-serif';ctx.fillStyle=c===selct?'#111':'#555';ctx.fillText(c,0,0);ctx.restore();});
  CSGEO={x:padL,bw,ct:order,v:vals};
  note.innerHTML=`Δ (ΔJ−ΔS) at each cell type's own optimal arrangement · bars colored by cell type · up = Δ&gt;0 · grey band = the selected cell type (${selct}) · hover for values.`;}
function ctstripHover(e){if(!CSGEO)return;const [mx]=cxy(e);const i=Math.floor((mx-CSGEO.x)/CSGEO.bw);
  if(i<0||i>=CSGEO.ct.length){tip(false);return;}
  tip(true,e.clientX,e.clientY,`<span style="color:${CTCOL[CSGEO.ct[i]]||'#888'}">■</span> ${CSGEO.ct[i]}<br>Δ = ${CSGEO.v[i].toFixed(3)}`);}
function setX(v){XMODE=v;if(CURKEY)drawCurves(DATA[CURKEY]);}
function setupInteract(){
  const cv=document.getElementById('curve');cv.addEventListener('mousemove',curveHover);cv.addEventListener('mouseleave',()=>tip(false));
  const hm=document.getElementById('heat');hm.addEventListener('mousemove',heatHover);hm.addEventListener('mouseleave',()=>tip(false));
  const cs=document.getElementById('ctstrip');cs.addEventListener('mousemove',ctstripHover);cs.addEventListener('mouseleave',()=>tip(false));
  let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(()=>{if(CURKEY){drawHeat(DATA[CURKEY]);drawCurves(DATA[CURKEY]);ctstrip(DATA[CURKEY],document.querySelector('#t tbody tr.sel').dataset.ct);}},150);});
}
window.addEventListener('DOMContentLoaded',()=>{setupInteract();pickCT(CTORDER[0]);});
"""

DATA_JSON = json.dumps(DATA, separators=(",", ":"))
js = (JS.replace("__DATA__", DATA_JSON).replace("__LOGOS__", json.dumps(LOGOS, separators=(",", ":")))
        .replace("__CTX__", json.dumps(CTX, separators=(",", ":")))
        .replace("__CTORDER__", json.dumps(CTORDER, separators=(",", ":")))
        .replace("__CTCOL__", json.dumps(CTCOL, separators=(",", ":"))))

nsyn = len(DATA); nrows = len(sel)
HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Synergy by cell type — islet differentiation</title><style>{CSS}</style></head><body>
<h1>TF-motif-pair synergy by cell type — islet ChromBPNet catalog v1.1</h1>
<div class="sub">In-silico pairwise marginalization (Liu et al., Nature 2026 method). {nsyn} synergistic
motif pairs (Δ&gt;0.504 &amp; maxZ&gt;5.38 vs a lineage-mismatched null), DeepLIFT-passing only, each
re-evaluated in all 22 cell types. Pick a cell type below; click a pair for the spacing × orientation detail.</div>

<details class="key">
  <summary>Key — columns &amp; method</summary>
  <div style="font-size:12px;color:#555;margin:6px 0 8px">Two motif consensus sequences are inserted into 100 GC-matched inaccessible backgrounds across every
  orientation × spacing (0–200 bp); accessibility is predicted with each cell type's ChromBPNet model; the <b>joint</b> effect ΔJ at the optimal arrangement is
  compared to the <b>additive</b> expectation ΔS = ΔA+ΔB. Δ = ΔJ − ΔS. Single fold. A pair appears under a cell type only if it is synergistic <i>there</i>.</div>
  <table class="keytbl">
    <tr><td>motif A / B</td><td>catalog motif ID (ST##); its consensus is the inserted sequence. Motif A is the upstream insert.</td></tr>
    <tr><td>TF A / B</td><td>curator's transcription-factor annotation for that motif.</td></tr>
    <tr><td>Δ</td><td>ΔJ − ΔS in the selected cell type — the amount the joint effect exceeds the additive expectation. Must clear this cell type's synergy threshold to appear.</td></tr>
    <tr><td>ΔJ / ΔS</td><td>joint effect at the optimal arrangement / additive expectation (ΔA+ΔB).</td></tr>
    <tr><td>orient</td><td>FF = A→ B→ · FR = A→ ←B (convergent) · RF = ←A B→ (divergent) · RR = ←A ←B. Motif A is always the upstream insert.</td></tr>
    <tr><td>gap / cen-cen</td><td>inner-edge spacing / center-to-center distance (bp) at the optimal arrangement in this cell type.</td></tr>
    <tr><td>maxZ</td><td>z-score of the best arrangement vs all arrangements (high = hard syntax; sharp spacing/orientation preference).</td></tr>
    <tr><td>Wilcoxon p</td><td>signed-rank test, joint vs independent (A+B) across backgrounds at the optimal arrangement.</td></tr>
    <tr><td>syntax</td><td>hard = maxZ above the cell type's null-calibrated cutoff; otherwise soft.</td></tr>
    <tr><td>DeepLIFT</td><td>fraction of central attribution on the two inserted motifs (≥0.4 required to appear here; near 1 = model reads the motifs).</td></tr>
    <tr><td>heatmap / curves (detail)</td><td>orientation × spacing landscape, measured in the cell type where the pair's signal is strongest (per-cell-type landscapes were not stored). Hover for values.</td></tr>
    <tr><td>cell-type strip (detail)</td><td>Δ at each cell type's own optimal arrangement; the selected cell type is highlighted.</td></tr>
  </table>
</details>

<h2>Cell type <span class="ref" style="font-weight:400;font-size:12px;color:#888">— pick one (developmental order; number = synergistic pairs)</span></h2>
<div class="ctsel">{CTBTN}</div>

<div id="cthead"></div><div id="ctmeta"></div>
<input id="f" placeholder="search this cell type (e.g. ISL1, NKX6)…" oninput="filt()">
<span class="ref" style="margin-left:8px;font-size:12px;color:#888">or filter any column in its header — numeric columns accept operators, e.g. Δ <code>&gt;1</code> · hover a column name for its definition</span>
<div class="wrap"><table id="t"><thead><tr>
{THEAD_CELLS}
</tr></thead><tbody>
{TBODY}
</tbody></table></div>

<div id="backdrop" onclick="closeD()"></div>
<div id="detail">
  <div class="dhead"><b id="dtitle" style="font-size:16px;color:#3b2d7e"></b>
    <span class="pid" id="dpid"></span><span class="close" onclick="closeD()">✕</span></div>
  <div class="dgrid">
    <div class="dleft">
      <div class="logos">
        <div class="lb"><div class="cap" id="capA"></div><img id="logoA"></div>
        <div class="lb"><div class="cap" id="capB"></div><img id="logoB"></div>
      </div>
      <table class="stats-tbl" id="stbl"></table>
    </div>
    <div class="dright">
      <div class="dplot">
        <div class="dph">Distance-dependent joint effect <span class="pctag">measured in <span id="landct"></span></span></div>
        <label class="toggle"><input type="radio" name="xm" checked onclick="setX('gap')"> inner-edge gap</label>
        <label class="toggle"><input type="radio" name="xm" onclick="setX('center')"> center-to-center</label>
        <div id="curveleg"></div>
        <canvas id="curve"></canvas>
      </div>
      <div class="dplot">
        <div class="dph">Orientation × spacing (ΔJ − ΔS)</div>
        <canvas id="heat"></canvas>
      </div>
      <div class="dplot">
        <div class="dph">Cell-type specificity</div>
        <canvas id="ctstrip"></canvas><div id="ctstripnote" class="note"></div>
      </div>
    </div>
  </div>
</div>
<div id="tip"></div>
<script>{js}</script>
</body></html>"""

open(OUT, "w").write(HTML)
print(f"wrote {OUT}  ({len(HTML)/1e6:.1f} MB)")
