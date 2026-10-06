#!/usr/bin/env python
"""Build a single self-contained interactive HTML synergy report for an experimental collaborator.
Data-forward (no prose/verdicts). Adopts the v1 collaborator-bundle visual identity.

Sections: header + LIVE Δ/Z threshold controls (flags+counts recompute client-side; null percentiles
shown as reference) + overview figures + filterable/sortable table of all 2211 pairs + per-pair detail
panel (2 motif logos, orientation×spacing ΔJ heatmap, per-orientation distance curves with
gap↔center-to-center toggle, cell-type specificity strip for hero pairs, stats table).

Runs locally (CPU). Reads figures/, RES/{full,candidate,null}/*.npz, metadata.tsv, logos_cwm/.
"""
import os, glob, json, base64, html
import numpy as np, pandas as pd

SYN = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(SYN, "figures")
OUT = os.path.join(SYN, "synergy_report.html")

def b64png(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else ""

def q(x, s=1000):
    return int(round(float(x) * s)) if np.isfinite(x) else 0

# ---- motif -> TF label ----
meta = pd.read_csv(os.path.join(SYN, "metadata.tsv"), sep="\t")
tf = {r.short_id: (r.curator_tf if isinstance(r.curator_tf, str) and r.curator_tf.strip() else r.short_id)
      for r in meta.itertuples()}

# ---- calls: concat raw chunk summaries from full + subs runs (uniform schema), dedupe by pair ----
_sum = glob.glob(f"{SYN}/RES/full/summary__*.tsv") + glob.glob(f"{SYN}/RES/subs/summary__*.tsv")
full = pd.concat([pd.read_csv(f, sep="\t") for f in _sum if os.path.getsize(f) > 0], ignore_index=True)
full = full.drop_duplicates(subset=["idA", "idB"], keep="last").reset_index(drop=True)
print(f"calls loaded: {len(full)} pairs (from {len(_sum)} summary files)")
# CT-specificity: pair -> {ct: delta}. Prefer the all-pairs ctspec sweep (optimal arrangement across all
# CTs); fall back to the 22 hero candidate pairs if ctspec not yet computed.
cts = {}
_cs = glob.glob(f"{SYN}/RES/ctspec/ctspec__*.tsv")
if _cs:
    csdf = pd.concat([pd.read_csv(f, sep="\t") for f in _cs if os.path.getsize(f) > 0], ignore_index=True)
    for (a, b), g in csdf.groupby(["idA", "idB"]):
        cts[f"{a}__{b}"] = {r.ct: round(float(r.delta), 3) for r in g.itertuples() if pd.notna(r.delta)}
    print(f"CT-specificity: all-pairs ctspec sweep ({len(cts)} pairs)")
else:
    cand = pd.read_csv(os.path.join(FIG, "candidate_calls.tsv"), sep="\t")
    for (a, b), g in cand.groupby(["idA", "idB"]):
        cts[f"{a}__{b}"] = {r.ct: round(float(r.delta), 3) for r in g.itertuples()}
    print(f"CT-specificity: hero candidate pairs only ({len(cts)} pairs)")
# overlay RE-OPTIMIZED per-CT Δ for synergistic pairs (syn_allct: full sweep re-optimized in each CT)
_sa = glob.glob(f"{SYN}/RES/syn_allct/summary__*.tsv")
if _sa:
    sadf = pd.concat([pd.read_csv(f, sep="\t") for f in _sa if os.path.getsize(f) > 0], ignore_index=True)
    n_over = 0
    for (a, b), g in sadf.groupby(["idA", "idB"]):
        cts[f"{a}__{b}"] = {r.ct: round(float(r.delta), 3) for r in g.itertuples() if pd.notna(r.delta)}
        n_over += 1
    print(f"CT-specificity: overlaid {n_over} synergistic pairs with re-optimized (syn_allct) Δ")

# ---- DeepLIFT confirmation: pair -> frac_in (attribution fraction on the 2 inserted motifs) ----
dl = {}
for f in glob.glob(f"{SYN}/RES/deeplift/deeplift__*.tsv"):
    if os.path.getsize(f) == 0: continue
    for r in pd.read_csv(f, sep="\t").itertuples():
        v = getattr(r, "frac_in", None)
        if pd.notna(v): dl[f"{r.idA}__{r.idB}"] = round(float(v), 3)
print(f"DeepLIFT: {len(dl)} confirmed pairs")

# ---- null reference percentiles ----
nul = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(f"{SYN}/RES/null/summary__*.tsv")], ignore_index=True)
NULL = {"d95": float(np.nanpercentile(nul.delta, 95)), "d99": float(np.nanpercentile(nul.delta, 99)),
        "z95": float(np.nanpercentile(nul.maxZ, 95)), "z99": float(np.nanpercentile(nul.maxZ, 99))}

# ---- per-pair grids from NPZ ----
def grid_for(idA, idB, ct):
    for sub in ("full", "subs", "candidate"):
        p = f"{SYN}/RES/{sub}/{idA}__{idB}__{ct}.npz"
        if os.path.exists(p):
            return np.load(p, allow_pickle=True)
    return None

DATA = {}
logos_used = set()
missing = 0
for r in full.itertuples():
    key = f"{r.idA}__{r.idB}"
    z = grid_for(r.idA, r.idB, r.ct)
    if z is None:
        missing += 1; continue
    dJ = np.asarray(z["dJ"], float); gaps = np.asarray(z["gaps"], int)
    orients = [str(x) for x in z["orients"]]
    dS = float(z["dA"]) + float(z["dB"])
    DATA[key] = {
        "tfA": tf.get(r.idA, r.idA), "tfB": tf.get(r.idB, r.idB), "idA": r.idA, "idB": r.idB, "ct": r.ct,
        "o": orients, "g0": int(gaps[0]), "gs": int(gaps[1] - gaps[0]) if len(gaps) > 1 else 1, "gn": len(gaps),
        "dJ": [[q(v) for v in row] for row in dJ],  # orient x gap, x1000 ints
        "dS": q(dS), "dA": q(z["dA"]), "dB": q(z["dB"]),
        "oo": str(z["opt_orient"]), "og": int(z["opt_gap"]), "cc": q(r.opt_center_center),
        "z": q(r.maxZ), "p": float(r.wilcoxon_p), "wa": int(r.wa), "wb": int(r.wb),
        "djo": q(r.dJ_opt), "dlt": q(r.delta), "soft": bool(r.soft), "hero": int(key in cts),
        "dl": dl.get(key),
    }
    if key in cts:
        DATA[key]["cts"] = cts[key]
    logos_used.add(r.idA); logos_used.add(r.idB)
print(f"pairs embedded: {len(DATA)}  (missing npz: {missing})")

# ---- logos (consensus fwd) ----
LOGOS = {}
for st in logos_used:
    p = f"{SYN}/logos_cwm/{st}/{st}.fwd.png"
    u = b64png(p)
    if u: LOGOS[st] = u
print(f"logos embedded: {len(LOGOS)}/{len(logos_used)}")

# ---- canonical cell-type palette + order (config/cell_type_metadata.tsv) ----
_ctm = pd.read_csv(os.path.join(SYN, "cell_type_metadata.tsv"), sep="\t").sort_values("display_order")
CTCOL = dict(zip(_ctm["cell_type"], _ctm["color"]))
CTORDER = list(_ctm["cell_type"])

# ---- hero CT-specificity z-matrix (for the interactive overview heatmap) ----
CTZ = {"p": [], "c": [], "z": [], "k": []}
_zp = f"{FIG}/ct_specificity_zmatrix.tsv"
if os.path.exists(_zp):
    _zdf = pd.read_csv(_zp, sep="\t", index_col=0)
    CTZ = {"p": list(_zdf.index), "c": list(_zdf.columns),
           "z": [[None if pd.isna(v) else round(float(v), 3) for v in row] for row in _zdf.values], "k": []}
    # map each hero-pair label -> pair key (idA__idB) so heatmap rows can filter the table
    _cc = pd.read_csv(os.path.join(FIG, "candidate_calls.tsv"), sep="\t").drop_duplicates(["idA", "idB"])
    _lab2key = {r["pair"]: f"{r['idA']}__{r['idB']}" for _, r in _cc.iterrows()}
    CTZ["k"] = [_lab2key.get(p, "") for p in CTZ["p"]]

# ---- table rows (numeric cells; syn/hard/soft filled client-side) ----
def fnum(x, d=3):
    return f"{x:.{d}f}" if pd.notna(x) and np.isfinite(x) else ""
rows = []
for r in full.itertuples():
    key = f"{r.idA}__{r.idB}"
    if key not in DATA: continue
    tfa, tfb = tf.get(r.idA, r.idA), tf.get(r.idB, r.idB)
    rows.append(
        f'<tr data-id="{key}" onclick="showPair(this)">'
        f'<td class="id">{r.idA}</td><td class="tf">{html.escape(tfa)}</td>'
        f'<td class="id">{r.idB}</td><td class="tf">{html.escape(tfb)}</td>'
        f'<td class="ct">{r.ct}</td>'
        f'<td class="num">{fnum(r.dJ_opt)}</td><td class="num">{fnum(r.dS)}</td>'
        f'<td class="num d">{fnum(r.delta)}</td>'
        f'<td>{r.opt_orient}</td><td class="num">{int(r.opt_gap)}</td>'
        f'<td class="num">{fnum(r.opt_center_center,1)}</td>'
        f'<td class="num z">{fnum(r.maxZ,2)}</td>'
        f'<td class="num p">{r.wilcoxon_p:.1e}</td>'
        f'<td class="flag syn"></td><td class="flag hard"></td><td class="flag soft"></td>'
        f'</tr>')
TBODY = "\n".join(rows)

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:14px;color:#222}
h1{margin:0 0 2px;color:#3b2d7e;border-bottom:2px solid #3b2d7e;padding-bottom:4px;font-size:22px}
h2{color:#3b2d7e;font-size:15px;margin:20px 0 6px}
.sub{color:#777;font-size:13px;margin:2px 0 10px}
a{color:#1e6091}
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;background:#f7f6fb;border:1px solid #e0dcef;
  border-radius:6px;padding:9px 12px;margin:8px 0;font-size:13px;position:sticky;top:0;z-index:5}
.controls label{font-weight:600;color:#3b2d7e}
.controls input[type=number]{width:70px;padding:3px 5px;font-size:13px}
.controls .ref{color:#888;font-size:11px}
.counts b{font-variant-numeric:tabular-nums}
#f{padding:5px 9px;font-size:13px;width:260px;margin:6px 0}
.leg{font-size:11px;margin:4px 0 8px;color:#555}
.lg-sw{display:inline-block;width:11px;height:11px;border:1px solid #999;border-radius:2px;vertical-align:middle;margin:0 2px 0 8px}
.wrap{overflow:auto;max-height:72vh;border:1px solid #e5e7eb;border-radius:4px}
table{border-collapse:collapse;font-size:12px}
th,td{border:1px solid #e5e7eb;padding:3px 6px;white-space:nowrap;vertical-align:middle}
th{background:#eef;position:sticky;top:0;z-index:3;cursor:default;font-size:10.5px;box-shadow:0 1px 0 #cdd3e6;vertical-align:top}
.thl{cursor:pointer}.thl:hover{text-decoration:underline}
.colf{width:100%;box-sizing:border-box;font-size:10px;padding:1px 3px;margin-top:2px;border:1px solid #cdd3e6;border-radius:3px;font-weight:400;font-family:inherit}
.colf:focus{outline:1px solid #3b2d7e}
tbody tr{cursor:pointer}tbody tr:hover{background:#eef4fb}tbody tr.sel{background:#dce8f6}
td.id{font-family:ui-monospace,Menlo,monospace;color:#7a5;font-size:11px}
td.tf{color:#1e6091;font-weight:600}.ct{color:#666;font-size:11px}
.num{text-align:right;font-variant-numeric:tabular-nums}
td.d{font-weight:600}.flag{text-align:center;font-weight:600;font-size:11px}
.flag.on{color:#fff;border-radius:3px}
.syn.on{background:#e67e22}.hard.on{background:#c0392b}.soft.on{background:#d9a441}
.ovgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;margin:6px 0}
.ovcard{border:1px solid #e0dcef;border-radius:8px;padding:10px 12px;background:#fff;box-shadow:var(--shadow,0 1px 2px rgba(40,30,20,.05))}
.ovh{font-size:13px;font-weight:600;color:#3b2d7e;margin-bottom:6px}
.ovcard canvas{width:100%;display:block;cursor:default}
.ovn{font-size:10.5px;color:#888;margin-top:5px;line-height:1.4}
#backdrop{position:fixed;inset:0;background:rgba(20,16,40,.4);z-index:49;display:none}
#backdrop.show{display:block}
#detail{position:fixed;inset:26px;z-index:50;overflow:auto;background:#fff;border:2px solid #3b2d7e;
  border-radius:10px;box-shadow:0 12px 44px rgba(0,0,0,.32);padding:14px 18px;display:none}
#detail.show{display:block}
.dhead .close{margin-left:auto;cursor:pointer;color:#666;font-size:22px;font-weight:700;line-height:1;
  padding:0 6px;border:1px solid #ddd;border-radius:5px}
.dhead .close:hover{background:#f3f4f6}
.dhead{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.dhead .pid{font-family:ui-monospace,monospace;font-size:12px;color:#7a5}
.dhead .close{margin-left:auto;cursor:pointer;color:#999;font-size:18px;font-weight:700}
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
.stats-tbl th{background:#eef;color:#3b2d7e;text-align:left;position:static;padding:2px 8px;border:1px solid #e5e7eb;cursor:default}
.stats-tbl td{padding:2px 8px}
.toggle{font-size:11px;color:#3b2d7e;cursor:pointer;user-select:none;margin:4px 0}
.toggle input{vertical-align:middle}
.ctstrip{display:flex;gap:1px;align-items:flex-end;height:70px;margin-top:4px}
.ctstrip .bar{width:14px;background:#c0392b;position:relative}
.ctstrip .bar span{position:absolute;bottom:-14px;left:50%;transform:translateX(-50%) rotate(0);font-size:7px;color:#666;white-space:nowrap}
.note{font-size:11px;color:#888;margin-top:4px}
.ochip{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;margin-right:5px;border-radius:12px;
  border:1.5px solid var(--oc);color:var(--oc);cursor:pointer;user-select:none}
.ochip.off{opacity:.32;text-decoration:line-through}
#tip{position:fixed;display:none;z-index:60;background:#1c1a17;color:#fff;font-size:11px;line-height:1.5;
  padding:6px 9px;border-radius:5px;pointer-events:none;box-shadow:0 3px 12px rgba(0,0,0,.35);white-space:nowrap}
#curveleg{font-size:11px;margin:4px 0}
details.key{background:#f7f6fb;border:1px solid #e0dcef;border-radius:6px;padding:6px 12px;margin:8px 0}
details.key summary{cursor:pointer;font-weight:600;color:#3b2d7e;font-size:13px}
.keytbl{border-collapse:collapse;font-size:12px;margin:8px 0}
.keytbl td{border:1px solid #e5e7eb;padding:3px 8px;vertical-align:top}
.keytbl td:first-child{font-weight:600;color:#3b2d7e;white-space:nowrap;font-family:ui-monospace,monospace}
.keycols{columns:2;column-gap:24px}@media(max-width:800px){.keycols{columns:1}}
"""

JS = """
const DATA=__DATA__, LOGOS=__LOGOS__, NULL=__NULL__, CTZ=__CTZ__;
const CTORDER=__CTORDER__, CTCOL=__CTCOL__;
const OCOL={FF:"#c0392b",FR:"#2980b9",RF:"#27ae60",RR:"#8e44ad"};
function gaps(d){return Array.from({length:d.gn},(_,i)=>d.g0+i*d.gs);}
function thr(){return {D:parseFloat(document.getElementById('td').value), Z:parseFloat(document.getElementById('tz').value)};}
let ROWS=[];
function initRows(){
  ROWS=[...document.querySelectorAll('#t tbody tr')].map(tr=>{
    const d=DATA[tr.dataset.id];
    return {tr, txt:(d.tfA+' '+d.tfB+' '+d.idA+' '+d.idB+' '+d.ct+' '+d.tfA+'×'+d.tfB).toLowerCase(),
      delta:parseFloat(tr.querySelector('td.d').textContent),
      z:parseFloat(tr.querySelector('td.z').textContent),
      p:parseFloat(tr.querySelector('td.p').textContent), soft:d.soft,
      cS:tr.querySelector('td.flag.syn'), cH:tr.querySelector('td.flag.hard'), cF:tr.querySelector('td.flag.soft')};
  });
}
function setF(c,on){c.textContent=on?'\\u2714':'';c.classList.toggle('on',on);}
function apply(){
  const {D,Z}=thr(); let ns=0,nh=0,nf=0;
  for(const r of ROWS){
    const syn=(r.delta>D)&&(r.p<0.001), hard=syn&&(r.z>Z), soft=syn&&r.soft&&!hard;
    setF(r.cS,syn);setF(r.cH,hard);setF(r.cF,soft); if(syn)ns++; if(hard)nh++; if(soft)nf++;
  }
  document.getElementById('cnt').innerHTML=`<b>${ns}</b> synergistic · <b>${nh}</b> hard · <b>${nf}</b> soft &nbsp;<span class=ref>(of ${ROWS.length}; null 95%: Δ=${NULL.d95.toFixed(2)}, Z=${NULL.z95.toFixed(1)} · 99%: Δ=${NULL.d99.toFixed(2)}, Z=${NULL.z99.toFixed(1)})</span>`;
  if(ROWS.length)drawOverview();
}
let ftimer;
function filt(){clearTimeout(ftimer);ftimer=setTimeout(()=>{
  const gq=document.getElementById('f').value.toLowerCase();
  const cfs=[...document.querySelectorAll('.colf')].map(x=>({col:+x.dataset.col,v:x.value.trim()})).filter(c=>c.v);
  for(const r of ROWS){let vis=r.txt.includes(gq);
    if(vis)for(const c of cfs){if(!matchCol(r.tr.cells[c.col].textContent,c.v)){vis=false;break;}}
    r.tr.style.display=vis?'':'none';}},110);}
function matchCol(cell,f){
  const m=f.match(/^(>=|<=|>|<|=)\\s*(-?\\d*\\.?\\d+(?:[eE]-?\\d+)?)$/);
  if(m){const v=parseFloat(cell);if(isNaN(v))return false;const n=parseFloat(m[2]);
    return m[1]==='>'?v>n:m[1]==='<'?v<n:m[1]==='>='?v>=n:m[1]==='<='?v<=n:v===n;}
  return cell.toLowerCase().includes(f.toLowerCase());}
let asc={};function sortT(i){let tb=document.querySelector('#t tbody'),rs=[...tb.rows];asc[i]=!asc[i];
  rs.sort((a,b)=>{let x=a.cells[i].innerText.trim(),y=b.cells[i].innerText.trim();
  let nx=parseFloat(x),ny=parseFloat(y);let c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc[i]?c:-c;});
  rs.forEach(r=>tb.appendChild(r));}
let CUR=null, XMODE='gap';
function showPair(tr){
  document.querySelectorAll('#t tbody tr.sel').forEach(e=>e.classList.remove('sel'));tr.classList.add('sel');
  CUR=DATA[tr.dataset.id]; render();
  document.getElementById('detail').classList.add('show');
  document.getElementById('backdrop').classList.add('show');
  document.getElementById('detail').scrollTop=0;
}
function closeD(){document.getElementById('detail').classList.remove('show');document.getElementById('backdrop').classList.remove('show');}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeD();});
function xvals(d){const g=gaps(d);return XMODE==='gap'?g:g.map(k=>k+(d.wa+d.wb)/2);}
let VIS={}, HGEO=null, CGEO=null;
function render(){
  const d=CUR;
  document.getElementById('dtitle').textContent=`${d.tfA} × ${d.tfB}`;
  document.getElementById('dpid').textContent=`${d.idA} × ${d.idB} · primary CT: ${d.ct}`;
  document.getElementById('logoA').src=LOGOS[d.idA]||'';document.getElementById('logoB').src=LOGOS[d.idB]||'';
  document.getElementById('capA').textContent=`${d.tfA}  (${d.idA})`;document.getElementById('capB').textContent=`${d.tfB}  (${d.idB})`;
  VIS={}; d.o.forEach(o=>VIS[o]=true);
  document.getElementById('curveleg').innerHTML=d.o.map(o=>`<span class="ochip" data-o="${o}" style="--oc:${OCOL[o]||'#666'}" onclick="togO('${o}')">${o}</span>`).join('')+' <span class=ref>(click to toggle)</span>';
  drawHeat(d);drawCurves(d);stats(d);ctstrip(d);
}
function togO(o){VIS[o]=!VIS[o];document.querySelector('.ochip[data-o="'+o+'"]').classList.toggle('off',!VIS[o]);drawCurves(CUR);}
function div(t){if(t>=0){return `rgb(255,${Math.round(255-t*175)},${Math.round(255-t*195)})`;}
  const tt=-t;return `rgb(${Math.round(255-tt*180)},${Math.round(255-tt*140)},255)`;}
function drawHeat(d){
  const cv=document.getElementById('heat'),ctx=cv.getContext('2d');
  const no=d.o.length,ng=d.gn,pad=54,ch=38,W=Math.max(360,cv.parentElement.clientWidth-4);
  const cw=Math.max(2,(W-pad-10)/ng);
  cv.width=W;cv.height=24+no*ch+24;ctx.clearRect(0,0,cv.width,cv.height);
  const dS=d.dS/1000; let amax=0.001;
  for(const row of d.dJ)for(const v of row)amax=Math.max(amax,Math.abs(v/1000-dS));
  for(let o=0;o<no;o++){
    ctx.fillStyle='#333';ctx.font='11px sans-serif';ctx.textAlign='right';ctx.fillText(d.o[o],pad-6,24+o*ch+ch/2+3);
    for(let g=0;g<ng;g++){const v=d.dJ[o][g]/1000-dS,t=Math.max(-1,Math.min(1,v/amax));
      ctx.fillStyle=div(t);ctx.fillRect(pad+g*cw,24+o*ch,cw,ch-2);}
  }
  ctx.fillStyle='#333';ctx.textAlign='center';ctx.font='10px sans-serif';
  const g=gaps(d);for(let k=0;k<ng;k+=Math.ceil(ng/8))ctx.fillText(g[k],pad+k*cw+cw/2,cv.height-6);
  ctx.fillText('inner-edge gap (bp) →  color = ΔJ − ΔS (red synergy, blue below additive)',pad+ng*cw/2,14);
  HGEO={pad,cw,ch,ng,no,amax,dS};
}
function drawCurves(d){
  const cv=document.getElementById('curve'),ctx=cv.getContext('2d');
  cv.width=Math.max(360,cv.parentElement.clientWidth-4);cv.height=480;ctx.clearRect(0,0,cv.width,cv.height);
  const X=xvals(d),dS=d.dS/1000;
  let ymin=dS,ymax=dS;for(const row of d.dJ)for(const v of row){const y=v/1000;ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);}
  const pad=46,pb=34,W=cv.width-pad-10,H=cv.height-24-pb,x0=Math.min(...X),x1=Math.max(...X);
  const px=x=>pad+(x-x0)/((x1-x0)||1)*W, py=y=>24+H-(y-ymin)/((ymax-ymin)||1)*H;
  ctx.strokeStyle='#ccc';ctx.beginPath();ctx.moveTo(pad,24);ctx.lineTo(pad,24+H);ctx.lineTo(pad+W,24+H);ctx.stroke();
  ctx.strokeStyle='#333';ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(pad,py(dS));ctx.lineTo(pad+W,py(dS));ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#333';ctx.font='10px sans-serif';ctx.textAlign='left';ctx.fillText('ΔS additive',pad+4,py(dS)-3);
  d.o.forEach((on,o)=>{if(!VIS[on])return;ctx.strokeStyle=OCOL[on]||'#666';ctx.lineWidth=1.7;ctx.beginPath();
    d.dJ[o].forEach((v,g)=>{const xx=px(X[g]),yy=py(v/1000);g?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy);});ctx.stroke();});
  const oi=d.o.indexOf(d.oo),gi=Math.round((d.og-d.g0)/d.gs);
  if(oi>=0&&VIS[d.oo]){ctx.fillStyle='#000';ctx.beginPath();ctx.arc(px(X[gi]),py(d.dJ[oi][gi]/1000),3.8,0,7);ctx.fill();}
  ctx.fillStyle='#333';ctx.textAlign='center';ctx.fillText(XMODE==='gap'?'inner-edge gap (bp)':'center-to-center (bp)',pad+W/2,cv.height-6);
  ctx.save();ctx.translate(12,24+H/2);ctx.rotate(-Math.PI/2);ctx.fillText('ΔJ (log counts)',0,0);ctx.restore();
  CGEO={pad,W,x0,x1,X,px,py};
}
function tip(show,x,y,htm){const t=document.getElementById('tip');if(!show){t.style.display='none';return;}
  t.innerHTML=htm;t.style.display='block';t.style.left=(x+14)+'px';t.style.top=(y+12)+'px';}
function cxy(e){const r=e.target.getBoundingClientRect();return [(e.clientX-r.left)*(e.target.width/r.width),(e.clientY-r.top)*(e.target.height/r.height)];}
function curveHover(e){if(!CUR||!CGEO)return;const [mx]=cxy(e);const g=CGEO;
  let xi=0,best=1e9;for(let i=0;i<g.X.length;i++){const xx=g.px(g.X[i]);if(Math.abs(xx-mx)<best){best=Math.abs(xx-mx);xi=i;}}
  const gp=gaps(CUR)[xi],cc=gp+(CUR.wa+CUR.wb)/2;
  const lines=CUR.o.filter(o=>VIS[o]).map(o=>`<span style="color:${OCOL[o]||'#666'}">${o}</span>: ${(CUR.dJ[CUR.o.indexOf(o)][xi]/1000).toFixed(3)}`);
  tip(true,e.clientX,e.clientY,`gap ${gp} bp · cc ${cc.toFixed(0)} bp · ΔS ${(CUR.dS/1000).toFixed(3)}<br>`+lines.join('<br>'));}
function heatHover(e){if(!CUR||!HGEO)return;const [mx,my]=cxy(e);const g=HGEO;
  const o=Math.floor((my-24)/g.ch), gi=Math.floor((mx-g.pad)/g.cw);
  if(o<0||o>=g.no||gi<0||gi>=g.ng){tip(false);return;}
  const dj=CUR.dJ[o][gi]/1000, gp=gaps(CUR)[gi];
  tip(true,e.clientX,e.clientY,`<span style="color:${OCOL[CUR.o[o]]||'#666'}">${CUR.o[o]}</span> · gap ${gp} bp<br>ΔJ ${dj.toFixed(3)} · ΔJ−ΔS ${(dj-CUR.dS/1000).toFixed(3)}`);}
function setupInteract(){
  const cv=document.getElementById('curve');cv.addEventListener('mousemove',curveHover);cv.addEventListener('mouseleave',()=>tip(false));
  const hm=document.getElementById('heat');hm.addEventListener('mousemove',heatHover);hm.addEventListener('mouseleave',()=>tip(false));
  const sc=document.getElementById('ovscatter');sc.addEventListener('mousemove',scHover);sc.addEventListener('mouseleave',()=>tip(false));sc.addEventListener('click',()=>{if(SCHIT)openPair(SCHIT);});
  const ct=document.getElementById('ovctheat');ct.addEventListener('mousemove',ctheatHover);ct.addEventListener('mouseleave',()=>tip(false));ct.addEventListener('click',ctheatClick);ct.style.cursor='pointer';
  const cs=document.getElementById('ctstrip');cs.addEventListener('mousemove',ctstripHover);cs.addEventListener('mouseleave',()=>tip(false));
  let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(()=>{drawScatter();drawCensus();drawSpacing();drawCTheat();},150);});
}
// ---------- interactive overview ----------
function ovsz(id,h){const cv=document.getElementById(id);cv.width=Math.max(340,(cv.parentElement.clientWidth-4));cv.height=h;return cv;}
let SPTS=[], SCHIT=null, CTHGEO=null;
function openPair(k){const tr=document.querySelector('#t tbody tr[data-id="'+CSS.escape(k)+'"]');if(tr){tr.scrollIntoView({block:'center'});showPair(tr);}}
function drawScatter(){
  const {D,Z}=thr(),cv=ovsz('ovscatter',430),ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  let lo=1e9,hi=-1e9;for(const r of ROWS){const d=DATA[r.tr.dataset.id];lo=Math.min(lo,d.dS/1000,d.djo/1000);hi=Math.max(hi,d.dS/1000,d.djo/1000);}
  lo-=0.15;hi+=0.15;const pad=46,W=cv.width-pad-14,H=cv.height-34-10;const px=x=>pad+(x-lo)/(hi-lo)*W,py=y=>10+H-(y-lo)/(hi-lo)*H;
  ctx.strokeStyle='#bbb';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(px(lo),py(lo));ctx.lineTo(px(hi),py(hi));ctx.stroke();ctx.setLineDash([]);
  ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(pad,10);ctx.lineTo(pad,10+H);ctx.lineTo(pad+W,10+H);ctx.stroke();
  SPTS=[];
  for(const r of ROWS){const d=DATA[r.tr.dataset.id];const syn=(r.delta>D)&&(r.p<0.001),hard=syn&&(r.z>Z);
    const x=px(d.dS/1000),y=py(d.djo/1000);ctx.fillStyle=hard?'#c0392b':(syn?'#e67e22':'rgba(150,160,170,.45)');
    ctx.beginPath();ctx.arc(x,y,hard?3.2:2.2,0,7);ctx.fill();if(syn)SPTS.push({x,y,key:r.tr.dataset.id});else SPTS.push({x,y,key:r.tr.dataset.id});}
  ctx.fillStyle='#333';ctx.font='11px sans-serif';ctx.textAlign='center';ctx.fillText('ΔS = ΔA+ΔB',pad+W/2,cv.height-4);
  ctx.save();ctx.translate(12,10+H/2);ctx.rotate(-Math.PI/2);ctx.fillText('ΔJ (optimal)',0,0);ctx.restore();
}
function scHover(e){const [mx,my]=cxy(e);let best=1e9,hit=null;for(const s of SPTS){const dd=(s.x-mx)**2+(s.y-my)**2;if(dd<best){best=dd;hit=s;}}
  SCHIT=(hit&&best<90)?hit.key:null;e.target.style.cursor=SCHIT?'pointer':'default';
  if(SCHIT){const d=DATA[SCHIT];tip(true,e.clientX,e.clientY,`${d.tfA} × ${d.tfB}<br>ΔJ ${(d.djo/1000).toFixed(3)} · ΔS ${(d.dS/1000).toFixed(3)} · Δ ${(d.dlt/1000).toFixed(3)}`);}else tip(false);}
function drawCensus(){
  const {D,Z}=thr();let h=0,s=0,y=0,n=0;
  for(const r of ROWS){const sy=(r.delta>D)&&(r.p<0.001),hd=sy&&(r.z>Z),sf=sy&&r.soft&&!hd;if(hd)h++;else if(sf)s++;else if(sy)y++;else n++;}
  const cv=ovsz('ovcensus',430),ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  const cats=[['hard',h,'#c0392b'],['soft',s,'#e67e22'],['syn only',y,'#95a5a6'],['none',n,'#dfe6e9']];
  const mx=Math.max(...cats.map(c=>c[1]),1),pad=40,W=cv.width-pad-16,H=cv.height-54,gap=W/cats.length,bw=gap*0.58;
  cats.forEach((c,i)=>{const bh=c[1]/mx*H,x=pad+i*gap+gap*0.21,yy=12+H-bh;ctx.fillStyle=c[2];ctx.fillRect(x,yy,bw,bh);
    ctx.fillStyle='#333';ctx.font='12px sans-serif';ctx.textAlign='center';ctx.fillText(c[1],x+bw/2,yy-4);
    ctx.font='11px sans-serif';ctx.fillText(c[0],x+bw/2,cv.height-8);});
}
function drawSpacing(){
  const {D,Z}=thr(),vals=[];
  for(const r of ROWS){const sy=(r.delta>D)&&(r.p<0.001),hd=sy&&(r.z>Z);if(hd)vals.push(DATA[r.tr.dataset.id].cc/1000);}
  const cv=ovsz('ovspacing',430),ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  const nb=27,bins=new Array(nb).fill(0);for(const v of vals){const b=Math.floor(v/3);if(b>=0&&b<nb)bins[b]++;}
  const mx=Math.max(...bins,1),pad=36,W=cv.width-pad-14,H=cv.height-52,bw=W/nb;
  ctx.fillStyle='#c0392b';bins.forEach((c,i)=>{const bh=c/mx*H;ctx.fillRect(pad+i*bw,12+H-bh,bw-1,bh);});
  ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(pad,12+H);ctx.lineTo(pad+W,12+H);ctx.stroke();
  ctx.fillStyle='#333';ctx.font='10px sans-serif';ctx.textAlign='center';for(let k=0;k<=nb;k+=6)ctx.fillText(k*3,pad+k*bw,cv.height-16);
  ctx.fillText('center-to-center (bp) — '+vals.length+' hard pairs',pad+W/2,cv.height-3);
}
function drawCTheat(){
  const np=CTZ.p.length,nc=CTZ.c.length;if(!np)return;
  const cv=document.getElementById('ovctheat');cv.width=Math.max(340,cv.parentElement.clientWidth-4);
  const padL=138,padT=6,cw=Math.max(6,Math.floor((cv.width-padL-6)/nc)),ch=15;
  cv.height=padT+np*ch+78;const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  for(let i=0;i<np;i++){ctx.fillStyle='#333';ctx.font='9px sans-serif';ctx.textAlign='right';ctx.fillText(CTZ.p[i].slice(0,20),padL-4,padT+i*ch+ch/2+3);
    for(let j=0;j<nc;j++){const v=CTZ.z[i][j];ctx.fillStyle=(v==null)?'#eee':div(Math.max(-1,Math.min(1,v/3)));ctx.fillRect(padL+j*cw,padT+i*ch,cw-1,ch-1);}}
  ctx.fillStyle='#666';ctx.font='8px sans-serif';
  for(let j=0;j<nc;j++){ctx.save();ctx.translate(padL+j*cw+cw/2+3,padT+np*ch+4);ctx.rotate(Math.PI/2);ctx.textAlign='left';ctx.fillText(CTZ.c[j],0,0);ctx.restore();}
  CTHGEO={padL,padT,cw,ch,np,nc};
}
function ctheatHover(e){if(!CTHGEO)return;const [mx,my]=cxy(e),g=CTHGEO,i=Math.floor((my-g.padT)/g.ch),j=Math.floor((mx-g.padL)/g.cw);
  if(mx<g.padL||i<0||i>=g.np||j<0||j>=g.nc){tip(false);return;}const v=CTZ.z[i][j];
  tip(true,e.clientX,e.clientY,`${CTZ.p[i]}<br>${CTZ.c[j]}: z = ${v==null?'n/a':v}<br><span class=ref>click row → filter table</span>`);}
function ctheatClick(e){if(!CTHGEO||!CTZ.k)return;const [mx,my]=cxy(e),i=Math.floor((my-CTHGEO.padT)/CTHGEO.ch);
  if(i<0||i>=CTHGEO.np)return;const key=CTZ.k[i];if(!key||!DATA[key])return;
  document.querySelectorAll('.colf').forEach(x=>x.value='');
  document.getElementById('f').value=DATA[key].tfA+'×'+DATA[key].tfB;filt();
  document.querySelector('.wrap').scrollIntoView({behavior:'smooth',block:'start'});}
function drawOverview(){drawScatter();drawCensus();drawSpacing();}
function stats(d){
  const dS=(d.dS/1000), dJ=Math.max(...d.dJ[d.o.indexOf(d.oo)])/1000, delta=dJ-dS;
  const rows=[['ΔJ (optimal)',dJ.toFixed(3)],['ΔS = ΔA+ΔB',dS.toFixed(3)],['ΔA / ΔB',(d.dA/1000).toFixed(3)+' / '+(d.dB/1000).toFixed(3)],
    ['ΔJ − ΔS',delta.toFixed(3)],['optimal orientation',d.oo],['optimal gap (bp)',d.og],['center-to-center (bp)',(d.cc/1000).toFixed(1)],
    ['max Z (arrangements)',(d.z/1000).toFixed(2)],['Wilcoxon p',d.p.toExponential(2)],['motif widths (A/B)',d.wa+' / '+d.wb],['primary cell type',d.ct],
    ['DeepLIFT frac on motifs', d.dl!=null ? d.dl.toFixed(2)+(d.dl<0.4?' ⚠ artifact?':'') : '—']];
  document.getElementById('stbl').innerHTML='<tr><th>quantity</th><th>value</th></tr>'+rows.map(r=>`<tr><td>${r[0]}</td><td class=num>${r[1]}</td></tr>`).join('');
}
let CSGEO=null;
function ctstrip(d){
  const cv=document.getElementById('ctstrip'),note=document.getElementById('ctstripnote');
  if(!d.cts){cv.style.display='none';CSGEO=null;
    const dj=Math.max(...d.dJ[d.o.indexOf(d.oo)])/1000-d.dS/1000;
    note.textContent=`Cell-type profile not computed — primary CT ${d.ct} only (Δ=${dj.toFixed(3)}).`;return;}
  cv.style.display='block';cv.width=Math.max(340,cv.parentElement.clientWidth-4);cv.height=200;
  const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  const order=CTORDER.filter(c=>c in d.cts),vals=order.map(c=>d.cts[c]),mx=Math.max(0.01,...vals.map(Math.abs));
  const padL=6,padT=8,padB=62,W=cv.width-padL-6,H=cv.height-padT-padB,zero=padT+H/2,bw=W/order.length;
  ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(padL,zero);ctx.lineTo(padL+W,zero);ctx.stroke();
  order.forEach((c,i)=>{const v=vals[i],bh=Math.abs(v)/mx*(H/2),x=padL+i*bw;
    ctx.fillStyle=CTCOL[c]||'#888';ctx.fillRect(x+0.5,v>=0?zero-bh:zero,Math.max(1,bw-1),bh);
    ctx.save();ctx.translate(x+bw/2+3,padT+H+4);ctx.rotate(Math.PI/2);ctx.textAlign='left';ctx.font='8px sans-serif';ctx.fillStyle='#555';ctx.fillText(c,0,0);ctx.restore();});
  CSGEO={x:padL,bw,ct:order,v:vals};
  note.innerHTML="Δ (ΔJ−ΔS) at this pair's optimal arrangement, per cell type · bars colored by cell type · up = Δ>0, down = Δ<0 · hover for values.";
}
function ctstripHover(e){if(!CSGEO)return;const [mx]=cxy(e);const i=Math.floor((mx-CSGEO.x)/CSGEO.bw);
  if(i<0||i>=CSGEO.ct.length){tip(false);return;}
  tip(true,e.clientX,e.clientY,`<span style="color:${CTCOL[CSGEO.ct[i]]||'#888'}">■</span> ${CSGEO.ct[i]}<br>Δ = ${CSGEO.v[i].toFixed(3)}`);}
function setX(v){XMODE=v;if(CUR)drawCurves(CUR);}
window.addEventListener('DOMContentLoaded',()=>{initRows();setupInteract();apply();drawCTheat();});
"""

# normalize hero-pair keys to the table's actual combination-order keys (homodimers absent -> "")
def _norm_key(k):
    if k in DATA: return k
    r = "__".join(k.split("__")[::-1])
    return r if r in DATA else ""
CTZ["k"] = [_norm_key(k) for k in CTZ.get("k", [])]

# ---- table header cells: sortable label + per-column filter input ----
HEADERS = ["motif A", "curated TF A", "motif B", "curated TF B", "primary CT", "ΔJ", "ΔS", "Δ",
           "opt orient", "opt gap", "center-cen", "maxZ", "Wilcoxon p", "syn", "hard", "soft"]
NUMCOLS = {5, 6, 7, 9, 10, 11, 12}
def _th(i, h):
    label = f'<div class="thl" onclick="sortT({i})">{h}</div>'
    if i < 13:  # skip the 3 dynamic flag columns
        ph = "&gt;n" if i in NUMCOLS else "filter"
        inp = f'<input class="colf" data-col="{i}" placeholder="{ph}" oninput="filt()" onclick="event.stopPropagation()">'
    else:
        inp = ""
    return f"<th>{label}{inp}</th>"
THEAD_CELLS = "".join(_th(i, h) for i, h in enumerate(HEADERS))

DATA_JSON = json.dumps(DATA, separators=(",", ":"))
LOGOS_JSON = json.dumps(LOGOS, separators=(",", ":"))
NULL_JSON = json.dumps(NULL)
js = (JS.replace("__DATA__", DATA_JSON).replace("__LOGOS__", LOGOS_JSON).replace("__NULL__", NULL_JSON)
        .replace("__CTZ__", json.dumps(CTZ, separators=(",", ":")))
        .replace("__CTORDER__", json.dumps(CTORDER, separators=(",", ":")))
        .replace("__CTCOL__", json.dumps(CTCOL, separators=(",", ":"))))

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>TF-motif-pair synergy — islet differentiation</title><style>{CSS}</style></head><body>
<h1>TF-motif-pair synergy — islet ChromBPNet catalog v1.1</h1>
<div class="sub">In-silico pairwise marginalization (Liu et al., Nature 2026 method). ΔJ = joint predicted
log-count effect of two motifs; ΔS = ΔA+ΔB additive expectation; Δ = ΔJ−ΔS. All {len(DATA)} keep-set
pairs, each in its primary cell type. Click a row for the spacing × orientation detail.</div>

<div class="controls">
  <span><label>synergy Δ ></label> <input id="td" type="number" step="0.01" value="0.50" oninput="apply()"></span>
  <span><label>hard Z ></label> <input id="tz" type="number" step="0.1" value="5.4" oninput="apply()"></span>
  <span class="counts" id="cnt"></span>
</div>
<div class="leg">Flags (syn / hard / soft) and the overview charts recompute live from the Δ / Z thresholds above.</div>

<details class="key" open>
  <summary>Key — columns, labels & method</summary>
  <div style="font-size:12px;color:#555;margin:6px 0 8px">Each pair: the two motif consensus sequences are inserted into 100 GC-matched inaccessible
  background sequences across every orientation × spacing (0–200 bp); accessibility is predicted with that cell type's ChromBPNet model; the
  <b>joint</b> effect ΔJ is compared to the <b>additive</b> expectation ΔS = ΔA+ΔB. Method: Liu et al., <i>Nature</i> 2026. Single model fold.</div>
  <table class="keytbl">
    <tr><th>name</th><th>description</th></tr>
    <tr><td>motif A / B</td><td>catalog motif ID (ST##); its consensus is the inserted sequence.</td></tr>
    <tr><td>curated TF A / B</td><td>collaborator's TF annotation for that motif (may be uncertain, e.g. "PAX6 (or ISL1)").</td></tr>
    <tr><td>primary CT</td><td>cell type where the two motifs most co-occur (Fi-NeMo hits); its ChromBPNet model was used for this pair's sweep.</td></tr>
    <tr><td>ΔJ</td><td>joint effect: mean increase in predicted log-counts with <b>both</b> motifs inserted, at the optimal arrangement.</td></tr>
    <tr><td>ΔS</td><td>additive expectation = ΔA + ΔB (each motif inserted alone).</td></tr>
    <tr><td>Δ</td><td>ΔJ − ΔS. &gt; 0 means more than additive.</td></tr>
    <tr><td>ΔA / ΔB</td><td>marginal effect of motif A / B inserted alone (log-counts vs background).</td></tr>
    <tr><td>opt orient</td><td>orientation at the optimal arrangement. F=forward, R=reverse-complement of (motif A, motif B): <b>FF FR RF RR</b>. FF = A→ B→ · FR = A→ ←B · RF = ←A B→ · RR = ←A ←B. Same-motif/palindromic pairs have fewer unique orientations.</td></tr>
    <tr><td>opt gap</td><td>inner-edge spacing (bp) at the optimal arrangement (0 = abutting).</td></tr>
    <tr><td>center-cen</td><td>center-to-center distance (bp) at the optimal arrangement.</td></tr>
    <tr><td>maxZ</td><td>z-score of the best arrangement's ΔJ vs the mean over all arrangements (higher = sharper spacing/orientation preference).</td></tr>
    <tr><td>Wilcoxon p</td><td>signed-rank test, joint vs independent (A+B) across the background sequences at the optimal arrangement.</td></tr>
    <tr><td>syn / hard / soft</td><td>flags recomputed from the Δ/Z inputs above. syn = Δ&gt;cutoff & p&lt;0.001; hard = syn & maxZ&gt;cutoff; soft = syn & Δ&gt;0.15 at some 20–150 bp arrangement (not hard).</td></tr>
    <tr><td>heatmap (detail)</td><td>orientation (rows) × spacing (cols, 0–200 bp); color = ΔJ − ΔS (red = above additive/synergy, blue = below). Hover for values.</td></tr>
    <tr><td>curves (detail)</td><td>ΔJ vs spacing, one line per orientation; dashed = additive ΔS; black dot = optimal. Click legend to toggle orientations; switch x-axis gap↔center; hover for values.</td></tr>
    <tr><td>cell-type specificity (detail)</td><td>Δ at the pair's optimal arrangement (fixed orientation + gap), re-evaluated in each of the 22 cell-type models. Bars: red = Δ&gt;0, blue = Δ&lt;0; heights scaled per pair.</td></tr>
  </table>
</details>

<h2>Overview <span class="ref" style="font-weight:400;font-size:12px;color:#888">— recomputes live with the thresholds above</span></h2>
<div class="ovgrid">
  <div class="ovcard"><div class="ovh">Joint vs. additive — all {len(DATA)} pairs</div><canvas id="ovscatter"></canvas>
    <div class="ovn">x = ΔS, y = ΔJ (optimal). grey = not synergistic · orange = synergistic · red = hard. Hover to identify · click to open the pair.</div></div>
  <div class="ovcard"><div class="ovh">Syntax census</div><canvas id="ovcensus"></canvas>
    <div class="ovn">counts of pairs by flag at the current thresholds.</div></div>
  <div class="ovcard"><div class="ovh">Hard-syntax spacing</div><canvas id="ovspacing"></canvas>
    <div class="ovn">center-to-center distance (bp) of pairs currently flagged hard.</div></div>
  <div class="ovcard"><div class="ovh">Cell-type specificity — hero pairs</div><canvas id="ovctheat"></canvas>
    <div class="ovn">z of Δ (=ΔJ−ΔS) across cell types, excl. each pair's own max CT. Hover for values. (22 hero pairs; expands as the all-CT sweep completes.)</div></div>
</div>

<h2>All pairs ({len(DATA)})</h2>
<input id="f" placeholder="global search (e.g. ISL1, NKX6, early_SC_EC)…" oninput="filt()">
<span class="ref" style="margin-left:8px">or filter any column in its header — numeric columns accept operators, e.g. Δ <code>&gt;0.5</code>, Wilcoxon p <code>&lt;0.001</code></span>
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
        <div class="dph">Distance-dependent joint effect</div>
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
