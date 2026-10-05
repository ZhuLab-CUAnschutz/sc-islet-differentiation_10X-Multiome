#!/usr/bin/env python
"""Group synergistic pairs by their BINARY lineage signature (which lineages they fire in) rather than
by clustering the continuous Δ matrix. Answers: (1) did within-block continuous clustering gain much;
(2) characterize the shared pairs as cross-lineage bridges; (3) discrete lineage-signature groups.

Lineage = the canonical `lineage` column (7): progenitor, endocrine_progenitor, beta, alpha, EC, delta,
non_pancreatic. A pair "fires" in a lineage if synergistic (q<0.05) in >=1 of that lineage's cell types.
"""
import os
from collections import Counter
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SF = os.environ.get("SYNERGY_DIR", os.path.join(REPO, "results", "6_synergy"))  # synergy work dir
CONFIG = os.path.join(REPO, "config")
calls = pd.read_csv(os.path.join(SF, "tables/calls_long.tsv"), sep="\t")
ctm = pd.read_csv(os.path.join(CONFIG, "cell_type_metadata.tsv"), sep="\t").sort_values("display_order")
lin = pd.read_csv(os.path.join(CONFIG, "lineage_metadata.tsv"), sep="\t").sort_values("display_order")
LINORDER = list(lin.lineage); LINCOL = dict(zip(lin.lineage, lin.color))
lineage_of = dict(zip(ctm.cell_type, ctm.lineage))
ENDO_LIN = {"endocrine_progenitor", "beta", "alpha", "EC", "delta"}   # endocrine block lineages

# binary pair x cell-type synergy, restricted to the 660 pairs syn in >=1 CT
syn = calls.pivot(index="pair", columns="ct", values="synergistic").fillna(False).astype(bool)
syn = syn[syn.any(axis=1)]
# collapse to pair x lineage (fires in >=1 CT of the lineage)
linmat = pd.DataFrame({L: syn[[c for c in syn.columns if lineage_of.get(c) == L]].any(axis=1)
                       for L in LINORDER}, index=syn.index)[LINORDER]
print(f"{len(linmat)} synergistic pairs x {len(LINORDER)} lineages\n")

# ---- (3) discrete lineage-signature groups ----
sig = linmat.apply(lambda r: tuple(int(x) for x in r), axis=1)
freq = Counter(sig)
def sig_name(s):
    return "+".join(L for L, b in zip(LINORDER, s) if b) or "none"
print("=== top lineage signatures (discrete groups) ===")
top = freq.most_common(14)
for s, n in top:
    print(f"  {n:4d}  {sig_name(s)}")
print(f"  ... {len(freq)} distinct signatures; top 14 cover "
      f"{sum(n for _, n in top)}/{len(linmat)} = {sum(n for _,n in top)/len(linmat)*100:.0f}%")
nfire = linmat.sum(axis=1)
print(f"\nbreadth: pairs firing in 1 lineage={int((nfire==1).sum())}, 2={int((nfire==2).sum())}, "
      f"3={int((nfire==3).sum())}, >=4={int((nfire>=4).sum())}")

# ---- (2) shared pairs as cross-lineage bridges ----
in_endo = linmat[list(ENDO_LIN & set(LINORDER))].any(axis=1)
in_non = linmat[[L for L in LINORDER if L not in ENDO_LIN]].any(axis=1)
shared = linmat[in_endo & in_non]
print(f"\n=== shared pairs ({len(shared)}): endocrine-lineage x non-endocrine-lineage co-firing ===")
endo_ls = [L for L in LINORDER if L in ENDO_LIN]
non_ls = [L for L in LINORDER if L not in ENDO_LIN]
co = pd.DataFrame({nl: [int((shared[el] & shared[nl]).sum()) for el in endo_ls] for nl in non_ls}, index=endo_ls)
print(co)
# ENP_phase1 specific bridge the collaborator noticed
enp1 = syn["ENP_phase1"]
lateNP = syn[[c for c in ["exocrine", "liver", "FB_FLT1"] if c in syn.columns]].any(axis=1)
print(f"\nENP_phase1 & non-pancreatic(exocrine/liver/FB) co-fire: {int((enp1 & lateNP).sum())} pairs "
      f"(of {int(enp1.sum())} that fire in ENP_phase1)")

# ---- reusable signature figure ----
def h2(h): return tuple(int(h[i:i+2], 16) / 255 for i in (1, 3, 5))
EMP = h2("#eeeeee")

def sig_figure(binmat, cols, fill_hex, is_endo, fname, title, ylabel, figsize, sep="signature"):
    """binmat: pairs x cols bool. Rows sorted by (breadth, developmental barycenter, signature) so
    specific groups sit on top and the fills cascade down the developmental axis. sep: 'signature'
    draws a line between identical signatures, 'breadth' between different #-active bands, None=off."""
    B = binmat[cols]
    sg = B.apply(lambda r: tuple(int(x) for x in r), axis=1)
    nf = B.sum(axis=1)
    def bary(s):
        idx = [i for i, b in enumerate(s) if b]
        return sum(idx) / len(idx) if idx else -1
    order = sorted(B.index, key=lambda p: (nf[p], bary(sg[p]), sg[p]))
    M = B.loc[order]
    rgb = np.ones((len(M), len(cols), 3))
    for j, c in enumerate(cols):
        col = h2(fill_hex[c]); v = M[c].values
        rgb[~v, j] = EMP; rgb[v, j] = col
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    prev = None
    for y, p in enumerate(order):
        key = sg[p] if sep == "signature" else (nf[p] if sep == "breadth" else None)
        if sep and prev is not None and key != prev:
            ax.axhline(y - 0.5, color="#111", lw=0.6 if sep == "signature" else 1.3)
        prev = key
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=8.5)
    for j, c in enumerate(cols):
        ax.get_xticklabels()[j].set_color("#5e4fa2" if is_endo(c) else "#f46d43")
    ax.set_yticks([]); ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(handles=[Patch(facecolor=LINCOL[L], label=L) for L in LINORDER]
              + [Patch(facecolor=EMP, label="not synergistic")],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(SF, "figures", f"{fname}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# (A) lineage signatures — 7 lineages, grouped by identical signature
lin_fill = {L: LINCOL[L] for L in LINORDER}
sig_figure(linmat, LINORDER, lin_fill, lambda L: L in ENDO_LIN,
           "lineage_signatures",
           "Lineage signatures — which lineages each pair is synergistic in (binary)\n"
           "sorted: specific (top) → promiscuous (bottom), developmental order within; purple x-labels = endocrine",
           f"{len(linmat)} synergistic pairs", (6.4, 11), sep="signature")
# (B) cell-type signatures — all 22 CT, colored by lineage, breadth-banded
CTORDER = list(ctm.cell_type)
ct_fill = {c: LINCOL[lineage_of[c]] for c in CTORDER}
ctsyn = syn[CTORDER]
sig_figure(ctsyn, CTORDER, ct_fill, lambda c: lineage_of[c] in ENDO_LIN,
           "celltype_signatures",
           "Cell-type signatures — which cell types each pair is synergistic in (binary)\n"
           "sorted: specific (top) → broad (bottom), developmental order within; fill = lineage color",
           f"{len(ctsyn)} synergistic pairs", (8.0, 11), sep="breadth")
print("\n-> figures/lineage_signatures.png, figures/celltype_signatures.png")

# ---- (C) developmental categories: ordered FIRST-MATCH predicates (exact + rules); rest = "mixed" ----
P, EP, B, A, EC, D, NP = "progenitor", "endocrine_progenitor", "beta", "alpha", "EC", "delta", "non_pancreatic"
MAT = {B, A, EC, D}                                   # mature endocrine lineages
def cat_of(s):
    hp, hn, he, m = P in s, NP in s, EP in s, (s & MAT)
    # non-endocrine axis
    if s == {P}: return "progenitor only"
    if s == {NP}: return "non-pancreatic only"
    if s == {P, NP}: return "progenitor + non-pancreatic"
    # endocrine progenitor
    if s == {EP}: return "endocrine-prog only"
    if s == {P, EP}: return "progenitor + endocrine-prog"
    if s == {P, EP, NP}: return "prog + endocrine-prog + non-panc"
    # broad (exact, before the general rules below)
    if s == {P, EP, B, A, EC, D}: return "all but non-pancreatic"
    if len(s) == 7: return "all 7 lineages"
    # single / paired mature endocrine (no ENP, no P, no NP)
    if s == {B}: return "β only"
    if s == {A}: return "α only"
    if s == {EC}: return "EC only"
    if s == {D}: return "δ only"
    if s <= MAT and len(s) == 2: return "mature endocrine pairs"   # any two mature (β+α, β+EC, ...)
    # RULES (request #3 + drain mixed)
    if he and m and not hp and not hn: return "endocrine-prog + ≥1 mature endo"
    if hp and he and m and not hn:     return "progenitor + endocrine-prog + mature"
    if hp and m and not he and not hn: return "progenitor + mature endo"
    if m and not he and not hp and not hn: return "other mature-endo mix"
    if hn and (he or m): return "non-pancreatic + endocrine mix"   # NP bridged with endocrine
    return "mixed"
CATORDER = ["progenitor only", "non-pancreatic only", "progenitor + non-pancreatic",
    "endocrine-prog only", "progenitor + endocrine-prog", "prog + endocrine-prog + non-panc",
    "endocrine-prog + ≥1 mature endo", "progenitor + endocrine-prog + mature",
    "β only", "α only", "EC only", "δ only", "mature endocrine pairs",
    "other mature-endo mix", "progenitor + mature endo", "non-pancreatic + endocrine mix",
    "all but non-pancreatic", "all 7 lineages", "mixed"]

sigset = linmat.apply(lambda r: frozenset(L for L, b in zip(LINORDER, r) if b), axis=1)
cat = sigset.map(cat_of)
def bary_p(p):
    idx = [LINORDER.index(L) for L in sigset[p]]; return sum(idx)/len(idx) if idx else -1
order, blocks = [], []
for c in CATORDER:
    pairs = sorted([p for p in linmat.index if cat[p] == c], key=lambda p: (len(sigset[p]), bary_p(p)))
    if not pairs: continue
    y0 = len(order); order += pairs
    blocks.append((f"{c} ({len(pairs)})", len(pairs), y0, len(order)))
n_mixed = int((cat == "mixed").sum())
print(f"\n=== developmental categories: {len(order)-n_mixed}/{len(linmat)} covered, mixed={n_mixed} ===")
for lab, n, *_ in blocks: print(f"  {n:4d}  {lab}")
if n_mixed:
    mix_freq = Counter(sig_name(tuple(int(L in sigset[p]) for L in LINORDER)) for p in linmat.index if cat[p] == "mixed")
    print("  biggest signatures still in MIXED:")
    for s, n in mix_freq.most_common(8): print(f"      {n:4d}  {s}")

M = linmat.loc[order][LINORDER]
rgb = np.ones((len(M), len(LINORDER), 3))
for j, L in enumerate(LINORDER):
    v = M[L].values; rgb[~v, j] = EMP; rgb[v, j] = h2(LINCOL[L])
fig, ax = plt.subplots(figsize=(8.6, 13.5))
ax.imshow(rgb, aspect="auto", interpolation="nearest")
for lab, n, yy0, yy1 in blocks:
    if 0 < yy1 < len(order): ax.axhline(yy1 - 0.5, color="#111", lw=1.0)
real = [b for b in blocks if b[1] > 0]
tot = len(order)
centers = [(yy0 + yy1) / 2 - 0.5 for _, _, yy0, yy1 in real]
minsep = tot * 0.030                         # push labels apart if they'd collide, keep order
label_y, last = [], -1e9
for c in centers:
    y = max(c, last + minsep); label_y.append(y); last = y
for (lab, n, yy0, yy1), cy, ly in zip(real, centers, label_y):
    ax.annotate(lab, xy=(-0.5, cy), xytext=(-len(LINORDER) * 0.60, ly), textcoords="data",
                ha="right", va="center", fontsize=8.5, annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#888", shrinkA=2, shrinkB=0))
ax.set_xticks(range(len(LINORDER))); ax.set_xticklabels(LINORDER, rotation=90, fontsize=9)
for j, L in enumerate(LINORDER):
    ax.get_xticklabels()[j].set_color("#5e4fa2" if L in ENDO_LIN else "#f46d43")
ax.set_yticks([])
ax.set_title(f"Synergy pairs grouped across differentiation ({len(blocks)} developmental categories)\n"
             "ordered early → late; first-match rules, every pair categorized", fontsize=11)
ax.legend(handles=[Patch(facecolor=LINCOL[L], label=L) for L in LINORDER] + [Patch(facecolor=EMP, label="not synergistic")],
          loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(SF, "figures", f"lineage_groups_ordered.{ext}"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("-> figures/lineage_groups_ordered.png")

# write signature tables
sigtab = linmat.copy(); sigtab["signature"] = sig.map(sig_name); sigtab["n_lineages"] = nfire
sigtab["dev_category"] = cat
sigtab.to_csv(os.path.join(SF, "tables/lineage_signatures.tsv"), sep="\t")

# ---- (D) 3-panel version. Non-endocrine panel splits progenitor into sub-stages; endo/mixed on 7 lineages ----
SHORT = {"endocrine-prog only": "ENP", "endocrine-prog + ≥1 mature endo": "ENP + mature",
    "β only": "β", "α only": "α", "EC only": "EC", "δ only": "δ",
    "mature endocrine pairs": "mature pairs", "other mature-endo mix": "mature mix",
    "progenitor + endocrine-prog": "prog + ENP", "prog + endocrine-prog + non-panc": "prog + ENP + non-panc",
    "progenitor + endocrine-prog + mature": "prog + ENP + mature", "progenitor + mature endo": "prog + mature",
    "non-pancreatic + endocrine mix": "non-panc + endocrine", "all but non-pancreatic": "all but non-panc",
    "all 7 lineages": "all 7"}
ENDO_PANEL = ["endocrine-prog only", "endocrine-prog + ≥1 mature endo", "β only", "α only", "EC only",
    "δ only", "mature endocrine pairs", "other mature-endo mix"]
MIXED_PANEL = ["progenitor + endocrine-prog", "progenitor + endocrine-prog + mature",
    "prog + endocrine-prog + non-panc", "progenitor + mature endo", "non-pancreatic + endocrine mix",
    "all but non-pancreatic", "all 7 lineages"]

# progenitor -> sub-stage (from canonical `grouping`); non-endocrine panel columns
grp_of = dict(zip(ctm.cell_type, ctm.grouping))
FINE_MAP = {"definitive_endoderm": "DE", "posterior_gut_tube": "gut tube", "posterior_foregut": "foregut",
            "pancreatic_progenitor": "panc-prog", "non_pancreatic": "non-pancreatic"}
NE_COLS = ["DE", "gut tube", "foregut", "panc-prog", "non-pancreatic"]
NE_COL = {"DE": "#cfcfcf", "gut tube": "#9e9e9e", "foregut": "#6f6f6f", "panc-prog": "#3f3f3f", "non-pancreatic": "#8c564b"}
ct_fine = {c: FINE_MAP.get(grp_of[c]) for c in CTORDER}
ne_pairs = [p for p in linmat.index if not (sigset[p] & set(ENDO_LIN))]
finemat = pd.DataFrame({F: syn[[c for c in CTORDER if ct_fine.get(c) == F]].any(axis=1) for F in NE_COLS},
                       index=syn.index).loc[ne_pairs]
PROG_FINE = {"DE", "gut tube", "foregut", "panc-prog"}
def cat_ne(s):
    prog = s & PROG_FINE; hn = "non-pancreatic" in s
    if s == {"DE"}: return "DE"
    if s == {"gut tube"}: return "gut tube"
    if s == {"foregut"}: return "foregut"
    if s == {"panc-prog"}: return "panc-prog"
    if prog and not hn and len(prog) >= 2: return "multi-progenitor"
    if prog and hn: return "progenitor + non-panc"
    if s == {"non-pancreatic"}: return "non-pancreatic"
    return "non-pancreatic"
NE_ORDER = ["DE", "gut tube", "foregut", "panc-prog", "multi-progenitor", "progenitor + non-panc", "non-pancreatic"]
finesig = finemat.apply(lambda r: frozenset(F for F, b in zip(NE_COLS, r) if b), axis=1)
necat = finesig.map(cat_ne)
def fbary(fs): idx = [NE_COLS.index(x) for x in fs]; return sum(idx)/len(idx) if idx else -1

def render(ax, mat, cols, colcol, blocks, title, labelcolor):
    order, bl = [], []
    for lab, pairs in blocks:
        if not pairs: continue
        y0 = len(order); order += pairs; bl.append((lab, y0, len(order)))
    M = mat.loc[order][cols]
    rgb = np.ones((len(M), len(cols), 3))
    for j, c in enumerate(cols):
        v = M[c].values; rgb[~v, j] = EMP; rgb[v, j] = h2(colcol[c])
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    for lab, y0, y1 in bl:
        if 0 < y1 < len(order): ax.axhline(y1 - 0.5, color="#111", lw=1.0)
    for sp in ax.spines.values(): sp.set_edgecolor("#bbb")
    tot = len(order); centers = [(y0 + y1) / 2 - 0.5 for _, y0, y1 in bl]
    minsep = tot * 0.05; ly, last = [], -1e9
    for cc in centers:
        y = max(cc, last + minsep); ly.append(y); last = y
    for (lab, y0, y1), cy, lyi in zip(bl, centers, ly):
        ax.annotate(lab, xy=(-0.5, cy), xytext=(-0.7, lyi), textcoords="data", ha="right", va="center",
                    fontsize=8.5, annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=0.6, color="#999"))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=7.5)
    for j, c in enumerate(cols): ax.get_xticklabels()[j].set_color(labelcolor(c))
    ax.set_yticks([]); ax.set_title(f"{title} ({tot})", fontsize=12, fontweight="bold", pad=8)

def blocks_7lin(cats):
    return [(f"{SHORT.get(c, c)} ({sum(cat==c)})",
             sorted([p for p in linmat.index if cat[p] == c], key=lambda p: (len(sigset[p]), bary_p(p)))) for c in cats]
ne_blocks = [(f"{c} ({int((necat==c).sum())})",
              sorted([p for p in ne_pairs if necat[p] == c], key=lambda p: (len(finesig[p]), fbary(finesig[p])))) for c in NE_ORDER]
lincol = {L: LINCOL[L] for L in LINORDER}
lab7 = lambda L: "#5e4fa2" if L in ENDO_LIN else "#f46d43"
labNE = lambda c: "#8c564b" if c == "non-pancreatic" else "#555555"

ENDO_ONLY_COLS = ["endocrine_progenitor", "beta", "alpha", "EC", "delta"]   # drop always-empty prog/non-panc
def build_3panel(endo_cols, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.6), gridspec_kw={"wspace": 0.95})
    render(axes[0], finemat, NE_COLS, NE_COL, ne_blocks, "Non-endocrine only", labNE)
    render(axes[1], linmat, endo_cols, lincol, blocks_7lin(ENDO_PANEL), "Endocrine only", lab7)
    render(axes[2], linmat, LINORDER, lincol, blocks_7lin(MIXED_PANEL), "Mixed endocrine + non-endocrine", lab7)
    fig.subplots_adjust(top=0.80, bottom=0.14)
    fig.legend(handles=[Patch(facecolor=LINCOL[L], label=L) for L in LINORDER]
               + [Patch(facecolor="#7f7f7f", label="progenitor sub-stage (light→dark = DE→panc-prog)"), Patch(facecolor=EMP, label="not synergistic")],
               loc="upper center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.92))
    fig.suptitle("Synergy pairs across differentiation, split by endocrine involvement", fontsize=14, y=0.985)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(SF, "figures", f"{fname}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
build_3panel(ENDO_ONLY_COLS, "lineage_groups_3panel")            # trimmed endocrine panel (5 cols)
build_3panel(LINORDER, "lineage_groups_3panel_allcols")          # endocrine panel keeps all 7 cols

# UNIFIED: all three panels share the same 10 fine columns (progenitor split + endocrine + non-panc)
FINE_ALL = ["DE", "gut tube", "foregut", "panc-prog", "endocrine_progenitor", "beta", "alpha", "EC", "delta", "non-pancreatic"]
FINE_MAP2 = {"definitive_endoderm": "DE", "posterior_gut_tube": "gut tube", "posterior_foregut": "foregut",
             "pancreatic_progenitor": "panc-prog", "non_pancreatic": "non-pancreatic"}
def fineall_of(ct): return FINE_MAP2.get(grp_of.get(ct), lineage_of.get(ct))
FINEALL_COL = {"DE": "#cfcfcf", "gut tube": "#9e9e9e", "foregut": "#6f6f6f", "panc-prog": "#3f3f3f",
    "endocrine_progenitor": LINCOL["endocrine_progenitor"], "beta": LINCOL["beta"], "alpha": LINCOL["alpha"],
    "EC": LINCOL["EC"], "delta": LINCOL["delta"], "non-pancreatic": LINCOL["non_pancreatic"]}
fineall_mat = pd.DataFrame({F: syn[[c for c in CTORDER if fineall_of(c) == F]].any(axis=1) for F in FINE_ALL}, index=syn.index)
ENDO_FINE = {"endocrine_progenitor", "beta", "alpha", "EC", "delta"}
def labFA(c): return "#5e4fa2" if c in ENDO_FINE else ("#8c564b" if c == "non-pancreatic" else "#555555")
figU, axU = plt.subplots(1, 3, figsize=(17.5, 7.6), gridspec_kw={"wspace": 0.7})
render(axU[0], fineall_mat, FINE_ALL, FINEALL_COL, ne_blocks, "Non-endocrine only", labFA)
render(axU[1], fineall_mat, FINE_ALL, FINEALL_COL, blocks_7lin(ENDO_PANEL), "Endocrine only", labFA)
render(axU[2], fineall_mat, FINE_ALL, FINEALL_COL, blocks_7lin(MIXED_PANEL), "Mixed endocrine + non-endocrine", labFA)
figU.subplots_adjust(top=0.82, bottom=0.16)
figU.legend(handles=[Patch(facecolor=FINEALL_COL[c], label=c) for c in FINE_ALL] + [Patch(facecolor=EMP, label="not synergistic")],
            loc="upper center", ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.94))
figU.suptitle("Synergy pairs across differentiation — same 10 lineage columns in every panel", fontsize=14, y=0.99)
for ext in ("png", "pdf"):
    figU.savefig(os.path.join(SF, "figures", f"lineage_groups_3panel_unified.{ext}"), dpi=150, bbox_inches="tight")
plt.close(figU)
print("-> figures/lineage_groups_3panel.png (+_allcols, +_unified)")

# ================= tables for all 660 pairs =================
ann = calls.drop_duplicates("pair").set_index("pair")[["tf_A", "tf_B"]]
ne_set = set(ne_pairs)
def major_of(p):
    e = bool(sigset[p] & set(ENDO_LIN)); n = bool(sigset[p] & {P, NP})
    return "mixed" if (e and n) else ("endocrine_only" if e else "non_endocrine_only")
def asciify(x): return x.replace("β", "beta").replace("α", "alpha").replace("δ", "delta").replace("≥", ">=")
sub_of = {p: asciify(necat[p] if p in ne_set else cat[p]) for p in linmat.index}
sub_rank = {asciify(name): i for i, name in enumerate(NE_ORDER)}
for i, name in enumerate(CATORDER): sub_rank.setdefault(asciify(name), 100 + i)
maj_rank = {"non_endocrine_only": 0, "endocrine_only": 1, "mixed": 2}
nct = syn[CTORDER].sum(axis=1)
syn_cts = {p: ";".join([c for c in CTORDER if bool(syn.loc[p, c])]) for p in linmat.index}
# Table 1: major group + subgroup breakdown
t1 = pd.DataFrame({
    "idA": [p.split("__")[0] for p in linmat.index], "idB": [p.split("__")[1] for p in linmat.index],
    "tf_A": [ann.loc[p, "tf_A"] for p in linmat.index], "tf_B": [ann.loc[p, "tf_B"] for p in linmat.index],
    "major_group": [major_of(p) for p in linmat.index], "subgroup": [sub_of[p] for p in linmat.index],
    "n_ct_synergistic": [int(nct[p]) for p in linmat.index],
    "synergistic_cell_types": [syn_cts[p] for p in linmat.index],
}, index=linmat.index)
t1.index.name = "pair"
t1 = t1.sort_values(by=["major_group", "subgroup", "n_ct_synergistic"],
                    key=lambda col: col.map(maj_rank) if col.name == "major_group"
                    else (col.map(sub_rank) if col.name == "subgroup" else -col))
t1.to_csv(os.path.join(SF, "tables/pair_groups.tsv"), sep="\t")
# Table 2: per-cell-type True/False synergy, same row order
t2 = syn.reindex(index=t1.index, columns=CTORDER).astype(bool)
t2.insert(0, "tf_B", [ann.loc[p, "tf_B"] for p in t2.index])
t2.insert(0, "tf_A", [ann.loc[p, "tf_A"] for p in t2.index])
t2.insert(0, "major_group", t1["major_group"]); t2.insert(1, "subgroup", t1["subgroup"])
t2.index.name = "pair"
t2.to_csv(os.path.join(SF, "tables/pair_ct_synergy_binary.tsv"), sep="\t")
print(f"-> tables/pair_groups.tsv ({len(t1)} pairs), tables/pair_ct_synergy_binary.tsv ({t2.shape[1]-4} CT cols)")
print(t1.major_group.value_counts().to_string())
