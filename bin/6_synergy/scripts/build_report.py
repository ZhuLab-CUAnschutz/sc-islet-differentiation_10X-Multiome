#!/usr/bin/env python
"""Build the collaborator bundle for the first-pass synergy run (single-task, all 22 CT).

Single-task only (no MT); empirical-null +
BH calling (live control = FDR q / min Δ / min frac_pos), 2701 pairs incl 73 homodimers, canonical
cell-type metadata. Data-forward (no interpretive takeaways); per-pair detail drawn client-side on
<canvas> from embedded numeric grids; only motif logos are raster.

Reads $SYNERGY_DIR/{tables,inputs/logos_b64.json,sweep/chunk_*/*.npz}. Emits into $SYNERGY_DIR/report/:  index.html, 1_methodological_overview.html, 2_all_pairs_report.html,
3_synergy_by_cell_type.html, 3_synergy_by_cell_type.xlsx, data/*.tsv, figures/*.png, REPORT.md
"""
import os, glob, json, base64, html, shutil
import numpy as np, pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SF = os.environ.get("SYNERGY_DIR", os.path.join(REPO, "results", "6_synergy"))  # synergy work dir
CONFIG = os.path.join(REPO, "config")
OUT = os.path.join(SF, "report"); os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
ALPHA = 0.05

def q(x, s=1000):
    return int(round(float(x) * s)) if (x is not None and np.isfinite(x)) else 0

# ---------- load ----------
calls = pd.read_csv(os.path.join(SF, "tables/calls_long.tsv"), sep="\t")
annot = pd.read_csv(os.path.join(SF, "tables/pair_annot.tsv"), sep="\t")
# representative CT per pair = argmin q_delta (tie -> larger delta); its grid is drawn in the pair modal
top = calls.sort_values(["q_delta", "delta"], ascending=[True, False]).groupby("pair").first()
GRIDS = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(SF, "sweep", "chunk_*", "*__*__*.npz"))}
LOGOS = json.load(open(os.path.join(SF, "inputs/logos_b64.json")))
ctm = pd.read_csv(os.path.join(CONFIG, "cell_type_metadata.tsv"), sep="\t").sort_values("display_order")
CTCOL = dict(zip(ctm.cell_type, ctm.color)); CTORDER = list(ctm.cell_type)
tf = dict(zip(annot.pair.map(lambda p: p.split("__")[0]), annot.tf_A))  # not used; per-row below

# per-pair aggregates
nsyn_ct = calls[calls.synergistic].groupby("pair").size()          # #CT synergistic
syn_any = set(calls[calls.synergistic].pair.unique())
cts_delta = {p: dict(zip(g.ct, g.delta.round(3))) for p, g in calls.groupby("pair")}
cts_syn   = {p: set(g.loc[g.synergistic, "ct"]) for p, g in calls.groupby("pair")}
annot_i = annot.set_index("pair")

def grid_for(pair):
    ct = top.loc[pair, "ct"]
    p = GRIDS.get(f"{pair}__{ct}")
    return (np.load(p, allow_pickle=True), ct) if p else (None, ct)

def build_data(pairs):
    """DATA dict for a set of pairs, each at its representative (min q_delta) CT."""
    D = {}; missing = 0
    rep = calls.sort_values(["q_delta", "delta"], ascending=[True, False]).groupby("pair").first()
    for pair in pairs:
        z, ct = grid_for(pair)
        if z is None: missing += 1; continue
        r = rep.loc[pair]; a = annot_i.loc[pair]
        dJ = np.asarray(z["dJ"], float); gaps = np.asarray(z["gaps"], int); orients = [str(x) for x in z["orients"]]
        dS = float(z["dA"]) + float(z["dB"])
        D[pair] = {
            "tfA": _tf(a.tf_A, pair.split("__")[0]), "tfB": _tf(a.tf_B, pair.split("__")[1]),
            "idA": pair.split("__")[0], "idB": pair.split("__")[1], "ct": ct,
            "o": orients, "g0": int(gaps[0]), "gs": int(gaps[1]-gaps[0]) if len(gaps) > 1 else 1, "gn": len(gaps),
            "dJ": [[q(v) for v in row] for row in dJ], "dS": q(dS), "dA": q(z["dA"]), "dB": q(z["dB"]),
            "oo": str(z["opt_orient"]), "og": int(z["opt_gap"]), "cc": q(r.opt_center_center),
            "z": q(r.maxZ), "fp": round(float(r.frac_pos), 3) if pd.notna(r.frac_pos) else None,
            "wa": int(r.wa), "wb": int(r.wb), "djo": q(r.dJ_opt), "dlt": q(r.delta),
            "qd": float(r.q_delta), "qz": float(r.q_maxz), "wp": float(r.wilcoxon_p),
            "nsc": int(nsyn_ct.get(pair, 0)),
            "homo": bool(a.is_homodimer), "comp": bool(a.involves_composite),
            "catA": _s(a.category_A), "catB": _s(a.category_B),
            "cts": cts_delta.get(pair, {}), "sct": sorted(cts_syn.get(pair, [])),
        }
    return D, missing

def _tf(v, fb): return v if isinstance(v, str) and v.strip() and v != "nan" else fb
def _s(v): return "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)

# ---------- CT-specificity z-matrix for the overview heatmap (top synergistic pairs) ----------
def ctz_matrix(pairs, nmax=45):
    sub = [p for p in pairs if p in syn_any]
    rep = calls.sort_values("q_delta").groupby("pair").first()
    sub = sorted(sub, key=lambda p: rep.loc[p, "q_delta"])[:nmax]
    mat = pd.DataFrame({p: cts_delta.get(p, {}) for p in sub}).T.reindex(columns=CTORDER)
    z = mat.sub(mat.mean(1), axis=0).div(mat.std(1).replace(0, 1), axis=0)
    labels = [f'{_tf(annot_i.loc[p].tf_A, p.split("__")[0])}×{_tf(annot_i.loc[p].tf_B, p.split("__")[1])}' for p in sub]
    return {"p": labels, "c": CTORDER,
            "z": [[None if pd.isna(v) else round(float(v), 3) for v in row] for row in z.values],
            "k": sub}

# ======================================================================================
# shared CSS / JS  (adapted from build_synergy_report.py; live control = q / Δ / frac_pos)
# ======================================================================================
CSS = r"""
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:14px;color:#222}
h1{margin:0 0 2px;color:#3b2d7e;border-bottom:2px solid #3b2d7e;padding-bottom:4px;font-size:22px}
h2{color:#3b2d7e;font-size:15px;margin:20px 0 6px}
.sub{color:#777;font-size:13px;margin:2px 0 10px}
a{color:#1e6091}
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;background:#f7f6fb;border:1px solid #e0dcef;
  border-radius:6px;padding:9px 12px;margin:8px 0;font-size:13px;position:sticky;top:0;z-index:5}
.controls label{font-weight:600;color:#3b2d7e}
.controls input[type=number]{width:72px;padding:3px 5px;font-size:13px}
.controls .ref{color:#888;font-size:11px}
.counts b{font-variant-numeric:tabular-nums}
#f{padding:5px 9px;font-size:13px;width:300px;margin:6px 0}
.leg{font-size:11px;margin:4px 0 8px;color:#555}
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
.cat{font-size:10px;text-align:center;color:#666}
.ovgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;margin:6px 0}
.ovcard{border:1px solid #e0dcef;border-radius:8px;padding:10px 12px;background:#fff;box-shadow:0 1px 2px rgba(40,30,20,.05)}
.ovh{font-size:13px;font-weight:600;color:#3b2d7e;margin-bottom:6px}
.ovcard canvas{width:100%;display:block;cursor:default}
.ovn{font-size:10.5px;color:#888;margin-top:5px;line-height:1.4}
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
.stats-tbl th{background:#eef;color:#3b2d7e;text-align:left;position:static;padding:2px 8px;border:1px solid #e5e7eb;cursor:default}
.stats-tbl td{padding:2px 8px}
.toggle{font-size:11px;color:#3b2d7e;cursor:pointer;user-select:none;margin:4px 0}
.note{font-size:11px;color:#888;margin-top:4px}
.ochip{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;margin-right:5px;border-radius:12px;
  border:1.5px solid var(--oc);color:var(--oc);cursor:pointer;user-select:none}
.ochip.off{opacity:.32;text-decoration:line-through}
#tip{position:fixed;display:none;z-index:60;background:#1c1a17;color:#fff;font-size:11px;line-height:1.5;
  padding:6px 9px;border-radius:5px;pointer-events:none;box-shadow:0 3px 12px rgba(0,0,0,.35);white-space:nowrap}
details.key{background:#f7f6fb;border:1px solid #e0dcef;border-radius:6px;padding:6px 12px;margin:8px 0}
details.key summary{cursor:pointer;font-weight:600;color:#3b2d7e;font-size:13px}
.keytbl{border-collapse:collapse;font-size:12px;margin:8px 0}
.keytbl td{border:1px solid #e5e7eb;padding:3px 8px;vertical-align:top}
.keytbl td:first-child{font-weight:600;color:#3b2d7e;white-space:nowrap;font-family:ui-monospace,monospace}
.ctsel{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}
.ctbtn{font-size:11px;padding:3px 8px;border-radius:5px;border:1px solid #ccc;cursor:pointer;border-left:5px solid var(--cc)}
.ctbtn.sel{background:#3b2d7e;color:#fff;border-color:#3b2d7e}
"""

JS = r"""
const DATA=__DATA__, LOGOS=__LOGOS__, CTZ=__CTZ__, CTORDER=__CTORDER__, CTCOL=__CTCOL__;
const HASOV=__HASOV__;
const OCOL={FF:"#c0392b",FR:"#2980b9",RF:"#27ae60",RR:"#8e44ad"};
function gaps(d){return Array.from({length:d.gn},(_,i)=>d.g0+i*d.gs);}
function thr(){return {Q:parseFloat(document.getElementById('tq').value),
  D:parseFloat(document.getElementById('td').value), F:parseFloat(document.getElementById('tf2').value)};}
let ROWS=[];
function initRows(){
  ROWS=[...document.querySelectorAll('#t tbody tr')].map(tr=>{
    const d=DATA[tr.dataset.id];
    return {tr, id:tr.dataset.id,
      txt:(d.tfA+' '+d.tfB+' '+d.idA+' '+d.idB+' '+tr.dataset.ct+' '+d.tfA+'×'+d.tfB).toLowerCase(),
      qd:+tr.dataset.qd, qz:+tr.dataset.qz, dlt:+tr.dataset.dlt, fp:(tr.dataset.fp===''?1:+tr.dataset.fp),
      cS:tr.querySelector('td.flag.syn'), cH:tr.querySelector('td.flag.hard'), cF:tr.querySelector('td.flag.soft')};
  });
}
function setF(c,on){if(!c)return;c.textContent=on?'✔':'';c.classList.toggle('on',on);}
function apply(){
  const {Q,D,F}=thr(); let ns=0,nh=0,nf=0;
  for(const r of ROWS){
    const syn=(r.qd<Q)&&(r.dlt>D)&&(r.fp>=F), hard=syn&&(r.qz<Q), soft=syn&&!hard;
    setF(r.cS,syn);setF(r.cH,hard);setF(r.cF,soft); if(syn)ns++; if(hard)nh++; if(soft)nf++;
  }
  const el=document.getElementById('cnt');
  if(el) el.innerHTML=`<b>${ns}</b> synergistic · <b>${nh}</b> hard · <b>${nf}</b> soft <span class=ref>(of ${ROWS.length} shown; FDR q&lt;${Q}, Δ&gt;${D}, frac_pos≥${F})</span>`;
  if(HASOV&&ROWS.length)drawOverview();
}
let ftimer;
function filt(){clearTimeout(ftimer);ftimer=setTimeout(()=>{
  const gq=document.getElementById('f').value.toLowerCase();
  const cfs=[...document.querySelectorAll('.colf')].map(x=>({col:+x.dataset.col,v:x.value.trim()})).filter(c=>c.v);
  const ctf=window.SELCT||null;
  for(const r of ROWS){let vis=r.txt.includes(gq);
    if(vis&&ctf)vis=(r.tr.dataset.ct===ctf);
    if(vis)for(const c of cfs){if(!matchCol(r.tr.cells[c.col].textContent,c.v)){vis=false;break;}}
    r.tr.style.display=vis?'':'none';}},110);}
function matchCol(cell,f){
  const m=f.match(/^(>=|<=|>|<|=)\s*(-?\d*\.?\d+(?:[eE]-?\d+)?)$/);
  if(m){const v=parseFloat(cell);if(isNaN(v))return false;const n=parseFloat(m[2]);
    return m[1]==='>'?v>n:m[1]==='<'?v<n:m[1]==='>='?v>=n:m[1]==='<='?v<=n:v===n;}
  return cell.toLowerCase().includes(f.toLowerCase());}
let asc={};function sortT(i){let tb=document.querySelector('#t tbody'),rs=[...tb.rows];asc[i]=!asc[i];
  rs.sort((a,b)=>{let x=a.cells[i].innerText.trim(),y=b.cells[i].innerText.trim();
  let nx=parseFloat(x),ny=parseFloat(y);let c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc[i]?c:-c;});
  rs.forEach(r=>tb.appendChild(r));}
let CUR=null, XMODE='gap', VIS={}, HGEO=null, CGEO=null, CSGEO=null;
function showPair(tr){
  document.querySelectorAll('#t tbody tr.sel').forEach(e=>e.classList.remove('sel'));tr.classList.add('sel');
  CUR=DATA[tr.dataset.id]; render();
  document.getElementById('detail').classList.add('show');document.getElementById('backdrop').classList.add('show');
  document.getElementById('detail').scrollTop=0;
}
function closeD(){document.getElementById('detail').classList.remove('show');document.getElementById('backdrop').classList.remove('show');}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeD();});
function xvals(d){const g=gaps(d);return XMODE==='gap'?g:g.map(k=>k+(d.wa+d.wb)/2);}
function render(){
  const d=CUR;
  document.getElementById('dtitle').textContent=`${d.tfA} × ${d.tfB}`;
  document.getElementById('dpid').textContent=`${d.idA} × ${d.idB} · shown at most-significant CT: ${d.ct}`;
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
  const cw=Math.max(2,(W-pad-10)/ng);cv.width=W;cv.height=24+no*ch+24;ctx.clearRect(0,0,cv.width,cv.height);
  const dS=d.dS/1000;let amax=0.001;for(const row of d.dJ)for(const v of row)amax=Math.max(amax,Math.abs(v/1000-dS));
  for(let o=0;o<no;o++){ctx.fillStyle='#333';ctx.font='11px sans-serif';ctx.textAlign='right';ctx.fillText(d.o[o],pad-6,24+o*ch+ch/2+3);
    for(let g=0;g<ng;g++){const v=d.dJ[o][g]/1000-dS,t=Math.max(-1,Math.min(1,v/amax));ctx.fillStyle=div(t);ctx.fillRect(pad+g*cw,24+o*ch,cw,ch-2);}}
  ctx.fillStyle='#333';ctx.textAlign='center';ctx.font='10px sans-serif';
  const g=gaps(d);for(let k=0;k<ng;k+=Math.ceil(ng/8))ctx.fillText(g[k],pad+k*cw+cw/2,cv.height-6);
  ctx.fillText('inner-edge gap (bp) →  color = ΔJ − ΔS (red synergy, blue below additive)',pad+ng*cw/2,14);
  HGEO={pad,cw,ch,ng,no,amax,dS};
}
function drawCurves(d){
  const cv=document.getElementById('curve'),ctx=cv.getContext('2d');
  cv.width=Math.max(360,cv.parentElement.clientWidth-4);cv.height=420;ctx.clearRect(0,0,cv.width,cv.height);
  const X=xvals(d),dS=d.dS/1000;let ymin=dS,ymax=dS;for(const row of d.dJ)for(const v of row){const y=v/1000;ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);}
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
function stats(d){
  const dS=(d.dS/1000), dJ=(d.djo/1000), delta=(d.dlt/1000);
  const rows=[['ΔJ (optimal)',dJ.toFixed(3)],['ΔS = ΔA+ΔB',dS.toFixed(3)],['ΔA / ΔB',(d.dA/1000).toFixed(3)+' / '+(d.dB/1000).toFixed(3)],
    ['Δ = ΔJ − ΔS',delta.toFixed(3)],['optimal orientation',d.oo],['optimal gap (bp)',d.og],['center-to-center (bp)',(d.cc/1000).toFixed(1)],
    ['max Z (arrangements)',(d.z/1000).toFixed(2)],['frac_pos (bg consistency)',d.fp==null?'—':d.fp.toFixed(2)],
    ['q (empirical-null Δ, BH)',d.qd.toExponential(2)],['q (maxZ, BH)',d.qz.toExponential(2)],
    ['Wilcoxon p (record)',d.wp.toExponential(2)],['motif widths (A/B)',d.wa+' / '+d.wb],
    ['# cell types synergistic',d.nsc+' / 22'],['category A / B',(d.catA||'—')+' / '+(d.catB||'—')]];
  document.getElementById('stbl').innerHTML='<tr><th>quantity</th><th>value</th></tr>'+rows.map(r=>`<tr><td>${r[0]}</td><td class=num>${r[1]}</td></tr>`).join('');
}
function ctstrip(d){
  const cv=document.getElementById('ctstrip'),note=document.getElementById('ctstripnote');
  cv.style.display='block';cv.width=Math.max(340,cv.parentElement.clientWidth-4);cv.height=210;
  const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  const order=CTORDER.filter(c=>c in d.cts),vals=order.map(c=>d.cts[c]),mx=Math.max(0.01,...vals.map(Math.abs));
  const synset=new Set(d.sct||[]);
  const padL=6,padT=8,padB=70,W=cv.width-padL-6,H=cv.height-padT-padB,zero=padT+H/2,bw=W/order.length;
  ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(padL,zero);ctx.lineTo(padL+W,zero);ctx.stroke();
  order.forEach((c,i)=>{const v=vals[i],bh=Math.abs(v)/mx*(H/2),x=padL+i*bw;
    ctx.fillStyle=CTCOL[c]||'#888';ctx.fillRect(x+0.5,v>=0?zero-bh:zero,Math.max(1,bw-1),bh);
    if(synset.has(c)){ctx.fillStyle='#111';ctx.beginPath();ctx.arc(x+bw/2,padT-3,2.4,0,7);ctx.fill();}
    ctx.save();ctx.translate(x+bw/2+3,padT+H+6);ctx.rotate(Math.PI/2);ctx.textAlign='left';ctx.font='8px sans-serif';ctx.fillStyle='#555';ctx.fillText(c,0,0);ctx.restore();});
  CSGEO={x:padL,bw,ct:order,v:vals};
  note.innerHTML="Δ (ΔJ−ΔS) at each cell type's own optimal arrangement · bars colored by cell type · up=Δ&gt;0, down=Δ&lt;0 · ● marks cell types where the pair is called synergistic (q&lt;0.05) · hover for values.";
}
function ctstripHover(e){if(!CSGEO)return;const [mx]=cxy(e);const i=Math.floor((mx-CSGEO.x)/CSGEO.bw);
  if(i<0||i>=CSGEO.ct.length){tip(false);return;}
  tip(true,e.clientX,e.clientY,`<span style="color:${CTCOL[CSGEO.ct[i]]||'#888'}">■</span> ${CSGEO.ct[i]}<br>Δ = ${CSGEO.v[i].toFixed(3)}`);}
function setX(v){XMODE=v;if(CUR)drawCurves(CUR);}
// ---------- overview (page 2 only) ----------
let SPTS=[],SCHIT=null,CTHGEO=null;
function ovsz(id,h){const cv=document.getElementById(id);cv.width=Math.max(340,(cv.parentElement.clientWidth-4));cv.height=h;return cv;}
function openPair(k){const tr=document.querySelector('#t tbody tr[data-id="'+CSS.escape(k)+'"]');if(tr){tr.scrollIntoView({block:'center'});showPair(tr);}}
function drawOverview(){drawScatter();drawCensus();drawSpacing();drawCTheat();}
function drawScatter(){
  const {Q,D,F}=thr(),cv=ovsz('ovscatter',420),ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  let lo=1e9,hi=-1e9;for(const r of ROWS){const d=DATA[r.id];lo=Math.min(lo,d.dS/1000,d.djo/1000);hi=Math.max(hi,d.dS/1000,d.djo/1000);}
  lo-=0.15;hi+=0.15;const pad=46,W=cv.width-pad-14,H=cv.height-34-10;const px=x=>pad+(x-lo)/(hi-lo)*W,py=y=>10+H-(y-lo)/(hi-lo)*H;
  ctx.strokeStyle='#bbb';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(px(lo),py(lo));ctx.lineTo(px(hi),py(hi));ctx.stroke();ctx.setLineDash([]);
  ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(pad,10);ctx.lineTo(pad,10+H);ctx.lineTo(pad+W,10+H);ctx.stroke();
  SPTS=[];
  for(const r of ROWS){const d=DATA[r.id];const syn=(r.qd<Q)&&(r.dlt>D)&&(r.fp>=F),hard=syn&&(r.qz<Q);
    const x=px(d.dS/1000),y=py(d.djo/1000);ctx.fillStyle=hard?'#c0392b':(syn?'#e67e22':'rgba(150,160,170,.4)');
    ctx.beginPath();ctx.arc(x,y,hard?3.2:2.1,0,7);ctx.fill();SPTS.push({x,y,key:r.id});}
  ctx.fillStyle='#333';ctx.font='11px sans-serif';ctx.textAlign='center';ctx.fillText('ΔS = ΔA+ΔB',pad+W/2,cv.height-4);
  ctx.save();ctx.translate(12,10+H/2);ctx.rotate(-Math.PI/2);ctx.fillText('ΔJ (optimal)',0,0);ctx.restore();
}
function scHover(e){const [mx,my]=cxy(e);let best=1e9,hit=null;for(const s of SPTS){const dd=(s.x-mx)**2+(s.y-my)**2;if(dd<best){best=dd;hit=s;}}
  SCHIT=(hit&&best<90)?hit.key:null;e.target.style.cursor=SCHIT?'pointer':'default';
  if(SCHIT){const d=DATA[SCHIT];tip(true,e.clientX,e.clientY,`${d.tfA} × ${d.tfB}<br>ΔJ ${(d.djo/1000).toFixed(3)} · ΔS ${(d.dS/1000).toFixed(3)} · Δ ${(d.dlt/1000).toFixed(3)}`);}else tip(false);}
function drawCensus(){
  const {Q,D,F}=thr();let h=0,s=0,y=0,n=0;
  for(const r of ROWS){const sy=(r.qd<Q)&&(r.dlt>D)&&(r.fp>=F),hd=sy&&(r.qz<Q),sf=sy&&!hd;if(hd)h++;else if(sf)s++;else n++;}
  const cv=ovsz('ovcensus',420),ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  const cats=[['hard',h,'#c0392b'],['soft',s,'#e67e22'],['not syn',n,'#dfe6e9']];
  const mx=Math.max(...cats.map(c=>c[1]),1),pad=40,W=cv.width-pad-16,H=cv.height-54,gap=W/cats.length;
  cats.forEach((c,i)=>{const bh=c[1]/mx*H,x=pad+i*gap+gap*0.21,yy=12+H-bh,bw=gap*0.58;ctx.fillStyle=c[2];ctx.fillRect(x,yy,bw,bh);
    ctx.fillStyle='#333';ctx.font='12px sans-serif';ctx.textAlign='center';ctx.fillText(c[1],x+bw/2,yy-4);
    ctx.font='11px sans-serif';ctx.fillText(c[0],x+bw/2,cv.height-8);});
}
function drawSpacing(){
  const {Q,D,F}=thr(),vals=[];
  for(const r of ROWS){const sy=(r.qd<Q)&&(r.dlt>D)&&(r.fp>=F),hd=sy&&(r.qz<Q);if(hd)vals.push(DATA[r.id].cc/1000);}
  const cv=ovsz('ovspacing',420),ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  const nb=30,bins=new Array(nb).fill(0);for(const v of vals){const b=Math.floor(v/3);if(b>=0&&b<nb)bins[b]++;}
  const mx=Math.max(...bins,1),pad=36,W=cv.width-pad-14,H=cv.height-52,bw=W/nb;
  ctx.fillStyle='#c0392b';bins.forEach((c,i)=>{const bh=c/mx*H;ctx.fillRect(pad+i*bw,12+H-bh,bw-1,bh);});
  ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(pad,12+H);ctx.lineTo(pad+W,12+H);ctx.stroke();
  ctx.fillStyle='#333';ctx.font='10px sans-serif';ctx.textAlign='center';for(let k=0;k<=nb;k+=6)ctx.fillText(k*3,pad+k*bw,cv.height-16);
  ctx.fillText('center-to-center (bp) — '+vals.length+' hard pairs',pad+W/2,cv.height-3);
}
function drawCTheat(){
  const np=CTZ.p.length,nc=CTZ.c.length;if(!np)return;
  const cv=document.getElementById('ovctheat');cv.width=Math.max(340,cv.parentElement.clientWidth-4);
  const padL=150,padT=6,cw=Math.max(6,Math.floor((cv.width-padL-6)/nc)),ch=14;
  cv.height=padT+np*ch+82;const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  for(let i=0;i<np;i++){ctx.fillStyle='#333';ctx.font='9px sans-serif';ctx.textAlign='right';ctx.fillText(CTZ.p[i].slice(0,24),padL-4,padT+i*ch+ch/2+3);
    for(let j=0;j<nc;j++){const v=CTZ.z[i][j];ctx.fillStyle=(v==null)?'#eee':div(Math.max(-1,Math.min(1,v/2)));ctx.fillRect(padL+j*cw,padT+i*ch,cw-1,ch-1);}}
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
function setupInteract(){
  const cv=document.getElementById('curve');cv.addEventListener('mousemove',curveHover);cv.addEventListener('mouseleave',()=>tip(false));
  const hm=document.getElementById('heat');hm.addEventListener('mousemove',heatHover);hm.addEventListener('mouseleave',()=>tip(false));
  const cs=document.getElementById('ctstrip');cs.addEventListener('mousemove',ctstripHover);cs.addEventListener('mouseleave',()=>tip(false));
  if(HASOV){const sc=document.getElementById('ovscatter');sc.addEventListener('mousemove',scHover);sc.addEventListener('mouseleave',()=>tip(false));sc.addEventListener('click',()=>{if(SCHIT)openPair(SCHIT);});
    const ct=document.getElementById('ovctheat');ct.addEventListener('mousemove',ctheatHover);ct.addEventListener('mouseleave',()=>tip(false));ct.addEventListener('click',ctheatClick);ct.style.cursor='pointer';
    let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(drawOverview,150);});}
}
function pickCT(ct){window.SELCT=ct;document.querySelectorAll('.ctbtn').forEach(b=>b.classList.toggle('sel',b.dataset.ct===ct));
  const h=document.getElementById('cthead');if(h)h.textContent=ct;filt();}
window.addEventListener('DOMContentLoaded',()=>{initRows();setupInteract();apply();if(window.INITCT)pickCT(window.INITCT);});
"""

def controls_html(ndesc):
    return f"""<div class="controls">
  <span><label>FDR q &lt;</label> <input id="tq" type="number" step="0.01" value="0.05" oninput="apply()"></span>
  <span><label>min Δ &gt;</label> <input id="td" type="number" step="0.05" value="0" oninput="apply()"></span>
  <span><label>min frac_pos ≥</label> <input id="tf2" type="number" step="0.05" value="0" oninput="apply()"></span>
  <span class="counts" id="cnt"></span>
</div>
<div class="leg">Flags (syn / hard / soft){ndesc} recompute live from the controls. <b>synergistic</b> = empirical-null Δ passes BH at the chosen FDR (and any Δ / frac_pos floors); <b>hard</b> = also maxZ passes BH; <b>soft</b> = synergistic but not arrangement-locked.</div>"""

MODAL = """
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
      <div class="dplot"><div class="dph">Distance-dependent joint effect</div>
        <label class="toggle"><input type="radio" name="xm" checked onclick="setX('gap')"> inner-edge gap</label>
        <label class="toggle"><input type="radio" name="xm" onclick="setX('center')"> center-to-center</label>
        <div id="curveleg"></div><canvas id="curve"></canvas></div>
      <div class="dplot"><div class="dph">Orientation × spacing (ΔJ − ΔS)</div><canvas id="heat"></canvas></div>
      <div class="dplot"><div class="dph">Across cell types</div><canvas id="ctstrip"></canvas><div id="ctstripnote" class="note"></div></div>
    </div>
  </div>
</div>
<div id="tip"></div>"""

def fnum(x, d=3):
    return f"{x:.{d}f}" if (pd.notna(x) and np.isfinite(x)) else ""

def inject(js, DATA, CTZ, hasov):
    return (js.replace("__DATA__", json.dumps(DATA, separators=(",", ":")))
              .replace("__LOGOS__", json.dumps(LOGOS, separators=(",", ":")))
              .replace("__CTZ__", json.dumps(CTZ, separators=(",", ":")))
              .replace("__CTORDER__", json.dumps(CTORDER, separators=(",", ":")))
              .replace("__CTCOL__", json.dumps(CTCOL, separators=(",", ":")))
              .replace("__HASOV__", "true" if hasov else "false"))

# columns: 0 mA 1 tfA 2 mB 3 tfB 4 CT 5 ΔJ 6 ΔS 7 Δ 8 orient 9 gap 10 cc 11 maxZ 12 frac_pos 13 q_delta 14 #CTsyn 15 cat | 16 syn 17 hard 18 soft
HEADERS = ["motif A","TF A","motif B","TF B","cell type","ΔJ","ΔS","Δ","orient","gap","cen-cen","maxZ",
           "frac_pos","q (Δ)","#CT syn","cat","syn","hard","soft"]
NUMCOLS = {5,6,7,9,10,11,12,13,14}
DEFS = {0:"catalog motif ID (ST##); its consensus is the inserted sequence",1:"curated TF for motif A",
  2:"catalog motif ID for motif B",3:"curated TF for motif B",
  4:"the cell type shown for this pair = its most significant one (min q); the modal profiles all 22",
  5:"joint effect: mean Δlog-counts with BOTH motifs at the optimal arrangement",6:"additive expectation ΔA+ΔB",
  7:"ΔJ − ΔS (>0 = more than additive)",8:"orientation at the optimal arrangement (FF/FR/RF/RR)",
  9:"inner-edge gap (bp) at the optimal arrangement",10:"center-to-center distance (bp)",
  11:"z of the best arrangement's ΔJ vs all arrangements (higher = sharper geometry preference)",
  12:"fraction of the 32 backgrounds where joint>additive (within-pair consistency)",
  13:"BH q-value of Δ vs the in-matrix empirical null (pooled across all pair×CT)",
  14:"number of the 22 cell types where the pair is called synergistic (q<0.05)",
  15:"motif category: comp=involves a composite/dimer motif, homo=homodimer (A×A), base=both base"}
def _th(i,h):
    lab=f'<div class="thl" title="{html.escape(DEFS.get(i,""))}" onclick="sortT({i})">{h}</div>'
    if i<16:
        ph="&gt;n" if i in NUMCOLS else "filter"
        inp=f'<input class="colf" data-col="{i}" placeholder="{ph}" oninput="filt()" onclick="event.stopPropagation()">'
    else: inp=""
    return f"<th>{lab}{inp}</th>"
THEAD="".join(_th(i,h) for i,h in enumerate(HEADERS))

def cat_of(d): return "comp" if d["comp"] else ("homo" if d["homo"] else "base")

def row_html(pair, d, ctval, rowvals):
    """rowvals = dict with dJ_opt,dS,delta,opt_orient,opt_gap,cc,maxZ,frac_pos,q_delta,q_maxz for the shown CT."""
    fp = rowvals["frac_pos"]; fpa = f"{fp:.3f}" if (pd.notna(fp) and np.isfinite(fp)) else ""
    return (f'<tr data-id="{pair}" data-ct="{ctval}" data-qd="{rowvals["q_delta"]:.3e}" '
      f'data-qz="{rowvals["q_maxz"]:.3e}" data-dlt="{rowvals["delta"]:.4f}" data-fp="{fpa}" onclick="showPair(this)">'
      f'<td class="id">{d["idA"]}</td><td class="tf">{html.escape(d["tfA"])}</td>'
      f'<td class="id">{d["idB"]}</td><td class="tf">{html.escape(d["tfB"])}</td>'
      f'<td class="ct">{ctval}</td>'
      f'<td class="num">{fnum(rowvals["dJ_opt"])}</td><td class="num">{fnum(rowvals["dS"])}</td>'
      f'<td class="num d">{fnum(rowvals["delta"])}</td>'
      f'<td>{rowvals["opt_orient"]}</td><td class="num">{int(rowvals["opt_gap"])}</td>'
      f'<td class="num">{fnum(rowvals["cc"],1)}</td><td class="num z">{fnum(rowvals["maxZ"],2)}</td>'
      f'<td class="num">{fnum(rowvals["frac_pos"],2)}</td>'
      f'<td class="num">{rowvals["q_delta"]:.1e}</td><td class="num">{d["nsc"]}</td>'
      f'<td class="cat">{cat_of(d)}</td>'
      f'<td class="flag syn"></td><td class="flag hard"></td><td class="flag soft"></td></tr>')

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head><body>
{header}
{controls}
{keyblock}
{overview}
<h2>{tabletitle}</h2>
{ctsel}
<input id="f" placeholder="global search (e.g. ISL1, NKX6, early_SC_EC)…" oninput="filt()">
<span class="ref" style="margin-left:8px">or filter any column header — numeric columns accept operators, e.g. Δ <code>&gt;0.5</code>, q <code>&lt;0.01</code></span>
<div class="wrap"><table id="t"><thead><tr>{thead}</tr></thead><tbody>
{tbody}
</tbody></table></div>
{modal}
<script>{js}</script>
</body></html>"""
# ======================================================================================
# page builders
# ======================================================================================
ALLPAIRS = sorted(cts_delta.keys())
SYNPAIRS = sorted(syn_any)
DATA_ALL, miss_all = build_data(ALLPAIRS)
DATA_SYN = {p: DATA_ALL[p] for p in SYNPAIRS if p in DATA_ALL}
CTZ = ctz_matrix(ALLPAIRS)
n_pairsyn = len(SYNPAIRS)
n_pctsyn = int(calls.synergistic.sum())
n_hard = int(calls.hard.sum())
print(f"DATA: {len(DATA_ALL)} pairs (missing npz {miss_all}); synergistic pairs {n_pairsyn}; "
      f"pair-CT syn {n_pctsyn} (hard {n_hard})")

rep = calls.sort_values(["q_delta", "delta"], ascending=[True, False]).groupby("pair").first()

# ---- page 2: all pairs (one row per pair at its representative CT) ----
rows2 = []
for pair in ALLPAIRS:
    if pair not in DATA_ALL: continue
    d = DATA_ALL[pair]; r = rep.loc[pair]
    rv = {"dJ_opt": r.dJ_opt, "dS": r.dS, "delta": r.delta, "opt_orient": r.opt_orient,
          "opt_gap": r.opt_gap, "cc": r.opt_center_center, "maxZ": r.maxZ, "frac_pos": r.frac_pos,
          "q_delta": r.q_delta, "q_maxz": r.q_maxz}
    rows2.append(row_html(pair, d, d["ct"], rv))
OVERVIEW = f"""<h2>Overview <span class="ref" style="font-weight:400;font-size:12px;color:#888">— recomputes live with the controls above</span></h2>
<div class="ovgrid">
  <div class="ovcard"><div class="ovh">Joint vs additive — all {len(DATA_ALL)} pairs</div><canvas id="ovscatter"></canvas>
    <div class="ovn">x = ΔS, y = ΔJ (optimal). grey = not synergistic · orange = synergistic · red = hard. Hover to identify · click to open.</div></div>
  <div class="ovcard"><div class="ovh">Syntax census</div><canvas id="ovcensus"></canvas>
    <div class="ovn">pairs by flag at the current thresholds (representative CT).</div></div>
  <div class="ovcard"><div class="ovh">Hard-syntax spacing</div><canvas id="ovspacing"></canvas>
    <div class="ovn">center-to-center (bp) of pairs currently flagged hard.</div></div>
  <div class="ovcard"><div class="ovh">Cell-type specificity — top synergistic pairs</div><canvas id="ovctheat"></canvas>
    <div class="ovn">row-z of Δ across the 22 cell types (top {len(CTZ["p"])} pairs by q). Hover for values · click a row → filter the table.</div></div>
</div>"""
KEY = """<details class="key">
  <summary>Key — columns & method</summary>
  <table class="keytbl">
  """ + "".join(f"<tr><td>{html.escape(h)}</td><td>{html.escape(DEFS.get(i,''))}</td></tr>" for i,h in enumerate(HEADERS) if i in DEFS) + """
  <tr><td>syn / hard / soft</td><td>live flags: syn = Δ passes BH at the chosen FDR q (+ optional Δ/frac_pos floors); hard = also maxZ passes BH (arrangement-locked); soft = syn & not hard.</td></tr>
  </table></details>"""

hdr2 = ('<h1>TF-motif-pair synergy — all pairs</h1>'
  f'<div class="sub">In-silico pairwise marginalization (single-task ChromBPNet, 22 cell types; method after Liu et al. Nature 2026). '
  f'ΔJ = joint predicted log-count effect; ΔS = ΔA+ΔB; Δ = ΔJ−ΔS. All <b>{len(DATA_ALL)}</b> motif pairs '
  f'(2,628 heterotypic + 73 homodimers), each shown at its most significant cell type. Click a row for the '
  f'spacing×orientation detail and the profile across all 22 cell types.</div>')
page2 = PAGE.format(title="Synergy — all pairs", css=CSS, header=hdr2, controls=controls_html(""),
  keyblock=KEY, overview=OVERVIEW, tabletitle=f"All pairs ({len(DATA_ALL)})", ctsel="", thead=THEAD,
  tbody="\n".join(rows2), modal=MODAL, js=inject(JS, DATA_ALL, CTZ, True))
open(os.path.join(OUT, "2_all_pairs_report.html"), "w").write(page2)
print(f"wrote 2_all_pairs_report.html ({len(page2)/1e6:.1f} MB)")

# ---- page 3: by cell type (one row per (pair,CT) that is synergistic in that CT) ----
csyn = calls[calls.synergistic].copy()
rows3 = []
for r in csyn.itertuples():
    if r.pair not in DATA_SYN: continue
    d = DATA_SYN[r.pair]
    rv = {"dJ_opt": r.dJ_opt, "dS": r.dS, "delta": r.delta, "opt_orient": r.opt_orient,
          "opt_gap": r.opt_gap, "cc": r.opt_center_center, "maxZ": r.maxZ, "frac_pos": r.frac_pos,
          "q_delta": r.q_delta, "q_maxz": r.q_maxz}
    rows3.append((r.ct, row_html(r.pair, d, r.ct, rv)))
# CT buttons with per-CT synergistic counts, colored by palette, in developmental order
ctcount = csyn.groupby("ct").size().to_dict()
btns = "".join(
    f'<div class="ctbtn" data-ct="{c}" style="--cc:{CTCOL.get(c,"#999")}" onclick="pickCT(\'{c}\')">{c} ({ctcount.get(c,0)})</div>'
    for c in CTORDER if c in ctcount)
CTSEL = (f'<div class="sub">Synergistic pairs browsable one cell type at a time. Showing <b id="cthead">{CTORDER[0]}</b>. '
         f'Each cell type\'s count is its number of synergistic pairs (q&lt;0.05).</div><div class="ctsel">{btns}</div>')
tbody3 = "\n".join(h for _, h in rows3)
hdr3 = ('<h1>TF-motif-pair synergy — by cell type</h1>'
  f'<div class="sub">The <b>{n_pairsyn}</b> pairs synergistic in ≥1 cell type ({n_pctsyn} pair×cell-type calls, {n_hard} hard), '
  f'browsable one cell type at a time. Click a pair for its detail and 22-cell-type profile.</div>')
# page 3 uses a per-CT initial selection
js3 = inject(JS, DATA_SYN, {"p": [], "c": CTORDER, "z": [], "k": []}, False)
js3 = f"window.INITCT={json.dumps(CTORDER[0])};\n" + js3
page3 = PAGE.format(title="Synergy — by cell type", css=CSS, header=hdr3, controls=controls_html(" (within the selected cell type)"),
  keyblock=KEY, overview="", tabletitle="Synergistic pairs by cell type", ctsel=CTSEL, thead=THEAD,
  tbody=tbody3, modal=MODAL, js=js3)
open(os.path.join(OUT, "3_synergy_by_cell_type.html"), "w").write(page3)
print(f"wrote 3_synergy_by_cell_type.html ({len(page3)/1e6:.1f} MB)")

# ---- data downloads ----
calls.to_csv(os.path.join(OUT, "data", "all_calls_long.tsv"), sep="\t", index=False)
csyn.to_csv(os.path.join(OUT, "data", "synergistic_calls.tsv"), sep="\t", index=False)
# per-pair summary (representative CT + #CT syn)
persum = rep.reset_index()[["pair", "idA", "idB", "ct", "dJ_opt", "dS", "delta", "opt_orient", "opt_gap",
    "opt_center_center", "maxZ", "frac_pos", "q_delta", "q_maxz", "call"]].copy()
persum["n_ct_synergistic"] = persum.pair.map(lambda p: int(nsyn_ct.get(p, 0)))
persum = persum.merge(annot[["pair", "tf_A", "tf_B", "category_A", "category_B", "is_homodimer", "involves_composite"]], on="pair")
persum.to_csv(os.path.join(OUT, "data", "per_pair_summary.tsv"), sep="\t", index=False)

# ---- xlsx ----
xlsx = os.path.join(OUT, "3_synergy_by_cell_type.xlsx")
with pd.ExcelWriter(xlsx) as xw:   # xlsxwriter if installed, else openpyxl
    persum.sort_values("q_delta").to_excel(xw, sheet_name="per_pair_summary", index=False)
    csyn.sort_values(["ct", "q_delta"]).to_excel(xw, sheet_name="synergistic_pair_by_CT", index=False)
    ctm.to_excel(xw, sheet_name="cell_type_metadata", index=False)
print(f"wrote xlsx + 3 data TSVs")

# ---- figures (copy the QC figures the methods page references) ----
for f in ["synergy_clustermap_rowz.png", "orientation_gap_qc.png", "block_membership.png",
          "synergy_clustermap_endocrine.png", "synergy_clustermap_non_endocrine.png"]:
    src = os.path.join(SF, "figures", f)
    if os.path.exists(src): shutil.copy(src, os.path.join(OUT, "figures", f))
for f in ["cluster_labels.tsv", "block_cluster_labels.tsv"]:
    src = os.path.join(SF, "tables", f)
    if os.path.exists(src): shutil.copy(src, os.path.join(OUT, "data", f))

# ---- page 1: methodological overview ----
methods = f"""<!doctype html><html><head><meta charset="utf-8"><title>Methodological overview</title>
<style>body{{font-family:Georgia,'Times New Roman',serif;max-width:820px;margin:34px auto;padding:0 22px;line-height:1.6;color:#1a1a1a}}
h1,h2{{font-family:-apple-system,'Segoe UI',sans-serif;color:#3b2d7e}} h1{{font-size:24px;border-bottom:2px solid #3b2d7e;padding-bottom:4px}}
h2{{font-size:16px;margin-top:26px;border-bottom:1px solid #e0dcef;padding-bottom:3px}} a{{color:#1e6091}}
figure{{margin:14px 0}} img{{max-width:100%;border:1px solid #e0dcef;border-radius:8px}} figcaption{{font-size:13px;color:#666;margin-top:5px}}
code{{background:#f3f1fb;padding:1px 5px;border-radius:3px;font-size:90%}} .small{{color:#777;font-size:13px}}</style></head><body>
<h1>Methodological overview</h1>
<div class="small">First-pass in-silico motif-pair synergy screen · single-task ChromBPNet · 22 cell states of stem-cell-derived islet differentiation · {len(DATA_ALL)} pairs. Model predictions, not experimental measurements.</div>

<h2>Models &amp; motifs</h2>
<p>Each of the 22 cell states has its own single-task ChromBPNet accessibility model (fold&nbsp;0). Motifs are the {73}
curated catalog v1.1 contribution-weight-matrix (CWM) consensus sequences (the 74-motif catalog minus ST53, "no expression").
Every heterotypic pair (C(73,2)=2,628) plus all 73 homodimers (A×A) is tested = <b>{len(DATA_ALL)}</b> pairs.</p>

<h2>In-silico marginalization</h2>
<p>For a pair, the two CWM consensus sequences are inserted into <b>32</b> GC-matched inaccessible background sequences across
every orientation (FF/FR/RF/RR; fewer for palindromic/homodimeric pairs) and inner-edge gap 0–50&nbsp;bp (1-bp steps). The model
predicts log-counts; the <b>joint</b> effect ΔJ (both motifs) is compared to the <b>additive</b> expectation ΔS&nbsp;=&nbsp;ΔA+ΔB
(each alone). <b>Δ&nbsp;=&nbsp;ΔJ&nbsp;−&nbsp;ΔS</b> at the single optimal arrangement is the synergy statistic. Gaps cap at 50&nbsp;bp
because &gt;99.9% of optima fall below it (median optimal gap here = 2&nbsp;bp).</p>

<h2>Calling synergy — in-matrix empirical null + BH</h2>
<p>Across 2,701 pairs most motif pairs do not truly cooperate, so the bulk of all Δ values is the no-synergy background. Per cell
type we fit that null on the central bulk (median + robust spread) and compute a right-tail p for each pair; the p-values are pooled
across all pair×cell-type tests and Benjamini–Hochberg corrected. <b>Synergistic = q&nbsp;&lt;&nbsp;0.05.</b> This retires the small
hand-built lineage-mismatched null. In the reports the FDR level is a live control.</p>

<h2>Hard vs soft syntax</h2>
<p><b>maxZ</b> measures how far the best arrangement sits above all other orientation×gap arrangements. <b>Hard</b> syntax
(arrangement-locked, composite-like) = synergistic and maxZ also passes BH; <b>soft</b> syntax (spacing-tolerant) = synergistic but not
peaky. Orientation-specificity feeds hardness (a pair that only works in one orientation has high maxZ).</p>

<h2>Within-pair consistency (frac_pos)</h2>
<p><code>frac_pos</code> = the fraction of the 32 backgrounds where joint&gt;additive at the optimal arrangement — an independent
robustness measure. Synergistic calls sit at median 0.94 (vs 0.75 for non-synergistic), i.e. calls are not driven by a few outlier
backgrounds. It is reported per pair and available as a live floor, but does not gate the default call.</p>

<h2>Across cell types</h2>
<p>Every pair is scored in all 22 models, so each has a full cell-type Δ profile (the strip in the per-pair detail). Filtering to pairs
synergistic in ≥1 cell type and clustering the row-z of Δ resolves the major axis below.</p>
<figure><img src="figures/synergy_clustermap_rowz.png"><figcaption>Row-z of Δ across cell types for the {n_pairsyn} pairs
synergistic in ≥1 cell type (Ward; k=2). Column bars: endocrine status and lineage (canonical metadata). The dominant split is
endocrine vs foregut/non-pancreatic.</figcaption></figure>
<p>A pair enters the <b>endocrine block</b> if it is called synergistic (q&lt;0.05) in ≥1 of the 11 endocrine cell types, and the
<b>non-endocrine block</b> if synergistic in ≥1 of the 11 non-endocrine cell types — so membership is by the synergy call, not by
clustering. This gives endocrine-only (192), shared (200), and non-endocrine-only (268) pairs.</p>
<figure><img src="figures/block_membership.png"><figcaption>Every synergistic pair (row), grouped by its assigned block, showing
which cell types it is synergistic in (q&lt;0.05). Purple = an endocrine cell type, orange = a non-endocrine cell type, grey = not
synergistic. Endocrine-only pairs fill only the middle (endocrine) columns; non-endocrine-only pairs only the flanking columns; shared
pairs both.</figcaption></figure>
<p>Reclustering within each block — on the row-z of Δ across only that block's cell types — sharpens the substructure (within-block
silhouette 0.20 / 0.19 vs 0.16 for the pooled clustering).</p>
<figure><img src="figures/synergy_clustermap_endocrine.png"><figcaption>Endocrine block: 392 pairs (row-z of Δ across the 11
endocrine cell types), reclustered. Lineage bar: ENP / β / α / EC / δ.</figcaption></figure>
<figure><img src="figures/synergy_clustermap_non_endocrine.png"><figcaption>Non-endocrine block: 468 pairs across the 11
non-endocrine cell types (DE → gut tube → foregut → pancreatic progenitor → non-pancreatic), reclustered.</figcaption></figure>

<h2>Orientation &amp; spacing QC</h2>
<figure><img src="figures/orientation_gap_qc.png"><figcaption>Among synergistic calls: winning orientation (hard/soft), optimal gap
(median 2&nbsp;bp — validates the 0–50&nbsp;bp window), and frac_pos (all ≥0.75).</figcaption></figure>

<h2>Software &amp; data</h2>
<p>ChromBPNet via bpnetlite; marginalization via tangermeme (ersatz/predict). Motif CWMs = catalog v1.1. Cell-type ordering/colors
from the shared canonical <code>cell_type_metadata.tsv</code>. First pass: n=32 backgrounds, gaps 0–50; a later tier re-runs called
pairs at n=100, gaps 0–200.</p>
</body></html>"""
open(os.path.join(OUT, "1_methodological_overview.html"), "w").write(methods)

# ---- index ----
index = f"""<!doctype html><html><head><meta charset="utf-8"><title>TF-motif-pair synergy in islet differentiation</title>
<style>body{{font-family:-apple-system,'Segoe UI',sans-serif;max-width:860px;margin:36px auto;padding:0 22px;line-height:1.55;color:#1a1a1a}}
h1{{color:#3b2d7e;font-size:25px;margin-bottom:4px}} .sub{{color:#666;font-size:14px;margin-bottom:22px}}
h2{{color:#3b2d7e;font-size:16px;margin-top:28px;border-bottom:1px solid #e0dcef;padding-bottom:3px}} a{{color:#1e6091;text-decoration:none}}
.card{{display:block;border:1px solid #e0dcef;border-radius:8px;padding:14px 16px;margin:10px 0;background:#faf9fd}}
.card:hover{{background:#f3f1fb}} .card .t{{font-weight:700;font-size:15px;color:#3b2d7e}} .card .d{{font-size:13px;color:#555;margin-top:3px}}
.tag{{display:inline-block;font-size:11px;background:#eceaf6;color:#3b2d7e;border-radius:4px;padding:1px 7px;margin-left:6px}}
img{{max-width:100%;border:1px solid #e0dcef;border-radius:8px;margin:8px 0}} .small{{font-size:13px;color:#777}} ul{{font-size:14px}}</style></head><body>
<h1>TF-motif-pair synergy in islet differentiation</h1>
<div class="sub">In-silico pairwise motif marginalization across 22 cell states of stem-cell-derived islet differentiation, single-task ChromBPNet. Method after Liu et al., Nature 2026.</div>
<p>Every pair of catalog motifs is inserted into inaccessible background sequences to ask whether the two together change predicted
accessibility more than the sum of each alone. Of <b>{len(DATA_ALL)}</b> pairs (2,628 heterotypic + 73 homodimers),
<b>{n_pairsyn}</b> are synergistic in ≥1 cell type ({n_pctsyn} pair×cell-type calls, {n_hard} hard) at FDR&nbsp;q&lt;0.05.</p>
<img src="figures/synergy_clustermap_rowz.png">
<h2>Reports and tables</h2>
<a class="card" href="1_methodological_overview.html"><div class="t">1. Methodological overview <span class="tag">read first</span></div>
  <div class="d">Models, marginalization, the empirical-null + BH synergy call, hard vs soft syntax, and the frac_pos consistency check, with a figure per step.</div></a>
<a class="card" href="2_all_pairs_report.html"><div class="t">2. All pairs report <span class="tag">interactive</span></div>
  <div class="d">All {len(DATA_ALL)} pairs, filterable and sortable, with live FDR / Δ / frac_pos controls. Click any pair for its spacing×orientation detail and 22-cell-type profile.</div></a>
<a class="card" href="3_synergy_by_cell_type.html"><div class="t">3. Synergy by cell type <span class="tag">interactive</span></div>
  <div class="d">The {n_pairsyn} synergistic pairs, browsable one cell state at a time, each with its detail and full cell-type profile.</div></a>
<h2>Data files</h2>
<ul>
  <li><a href="3_synergy_by_cell_type.xlsx">3_synergy_by_cell_type.xlsx</a>: per-pair summary + synergistic pair×cell-type calls as a workbook.</li>
  <li><a href="data/per_pair_summary.tsv">per_pair_summary.tsv</a>: one row per pair (representative cell type + #CT synergistic).</li>
  <li><a href="data/synergistic_calls.tsv">synergistic_calls.tsv</a> · <a href="data/all_calls_long.tsv">all_calls_long.tsv</a>: every pair×cell-type call.</li>
</ul>
<p class="small" style="margin-top:26px;border-top:1px solid #e0dcef;padding-top:10px">First-pass screen (n=32 backgrounds, gaps 0–50). Contents are model predictions, not experimental measurements.</p>
</body></html>"""
open(os.path.join(OUT, "index.html"), "w").write(index)

# ---- REPORT.md (provenance-style, data-forward) ----
rm = f"""# Synergy collaborator bundle — first pass

Generated by `bin/6_synergy/scripts/build_report.py` (single-task ChromBPNet, 22 cell types).

- Pairs: {len(DATA_ALL)} (2,628 heterotypic + 73 homodimers); motifs: 73 (catalog v1.1 minus ST53).
- Sweep: 1-bp, gaps 0–50, n=32 backgrounds. Calling: in-matrix empirical null + BH, q<0.05.
- Synergistic pairs (≥1 CT): {n_pairsyn}. Pair×CT calls: {n_pctsyn} ({n_hard} hard).

## Files
- `index.html` — landing page.
- `1_methodological_overview.html` — methods, with figures.
- `2_all_pairs_report.html` — interactive all-pairs table + per-pair detail (live FDR/Δ/frac_pos).
- `3_synergy_by_cell_type.html` — synergistic pairs browsable per cell type.
- `3_synergy_by_cell_type.xlsx`, `data/*.tsv` — tables.
- `figures/` — clustermap + orientation/gap QC.

Model predictions, not experimental measurements. First pass; a later tier re-runs called pairs at n=100, gaps 0–200.
"""
open(os.path.join(OUT, "REPORT.md"), "w").write(rm)
print("wrote 1_methodological_overview.html, index.html, REPORT.md")
print("DONE ->", OUT)

