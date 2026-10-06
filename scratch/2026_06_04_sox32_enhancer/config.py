"""
config.py — single source of truth for the sox32 enhancer analysis.

All paths, constants, construct-feature coordinates, motif databases, colors, and
matplotlib setup live here so the rest of the codebase has zero duplicated literals.

TF set switch: env var SOX32_TFSET (default "nodal" = EOMES/T/FOXH1) selects which
named-TF set the FIMO scan + motif-summary figures use, writing to separate paths
(SET_SUFFIX) so multiple sets coexist. e.g. SOX32_TFSET=mixl1_smad2.
"""
import os
import matplotlib

# ---------------------------------------------------------------- core inputs
FASTA = "/cellar/users/aklie/projects/islet_organoid_differentiation/sox32_enh.fa"
MODELS_ROOT = (
    "/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/"
    "results/3_single_task_models/_flat"
)
CONFIG_TSV = (
    "/cellar/users/aklie/projects/islet_organoid_differentiation/"
    "sc-islet-differentiation_10X-Multiome/config/cell_type_metadata.tsv"
)
MODEL_FILE = "chrombpnet_nobias.h5"            # bias-corrected single-task model

INPUT_LEN = 2114                               # ChromBPNet receptive field
OUTPUT_LEN = 1000                              # profile output window (centered)


def output_window(input_len=INPUT_LEN, output_len=OUTPUT_LEN):
    s = input_len // 2 - output_len // 2
    return s, s + output_len


# -------------------------------------------------- construct feature map (bp)
# Reconstructed from sequence + collaborator: the sox32 enhancer is the central
# ~1.1 kb (AT-rich); flanks carry vector + a minimal promoter (TATA@1614) + eGFP.
ENHANCER = (507, 1607)
FEATURES = [          # (label, start, end)  — non-overlapping, 0-based half-open
    ("5'_flank",       0,    507),
    ("enhancer",       507,  1607),
    ("min_promoter",   1607, 1647),
    ("eGFP_CDS",       1647, INPUT_LEN),
]
POINT_FEATURES = [("TATA", 1614), ("Kozak", 1641), ("eGFP_ATG", 1647)]


def feature_of(pos):
    """Which construct feature a base position falls in."""
    for lab, a, b in FEATURES:
        if a <= pos < b:
            return lab
    return "?"


def feature_of_span(start, end):
    """Feature containing a span's midpoint (for motif hits / seqlets)."""
    return feature_of((start + end) // 2)


# --------------------------------------------------------- motif databases
HOCOMOCO_MEME = "/cellar/users/aklie/data/ref/motifs/hocomoco_meme.meme"
_VIERSTRA_DIR = "/cellar/users/aklie/data/ref/motifs/jvierstra/motif-clustering-v2.0beta"
VIERSTRA_CONSENSUS_MEME = f"{_VIERSTRA_DIR}/consensus_pwms.meme"   # 693 named archetypes
VIERSTRA_MOTIFS_MEME = f"{_VIERSTRA_DIR}/motifs.meme"              # 2178 member motifs
VIERSTRA_XLSX = (
    "/carter/users/aklie/projects/islet_organoid_differentiation/"
    "sc-islet-differentiation_10X-Multiome/ref/motif_annotations.xlsx"
)
CHROMBPNET_CATALOG_MEME = (
    "/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/"
    "results/3_single_task_models/motifs/gimme_cluster/meme/combined_mapped.meme"
)

# named-TF sets (label -> HOCOMOCO H12 id). Switch with env SOX32_TFSET.
TF_SETS = {
    "nodal": {                                   # collaborator's first ask
        "EOMES": "EOMES.H12CORE.0.PSM.A",
        "T/TBXT": "TBXT.H12CORE.0.PS.A",
        "FOXH1": "FOXH1.H12CORE.0.P.B",
    },
    "mixl1_smad2": {                             # collaborator's follow-up
        "MIXL1": "MIXL1.H12CORE.0.SM.B",
        "SMAD2": "SMAD2.H12CORE.0.P.B",
    },
}
_TF_SET_COLORS = {
    "nodal": {"EOMES": "#1b9e77", "T/TBXT": "#7570b3", "FOXH1": "#d95f02"},
    "mixl1_smad2": {"MIXL1": "#e7298a", "SMAD2": "#66a61e"},
}
_TF_SET_VQUERY = {
    "nodal": {"EOMES": "EOMES", "T/TBXT": "TBXT|T_MA0009|T_TBX|BRAC", "FOXH1": "FOXH1"},
    "mixl1_smad2": {"MIXL1": "MIXL1|MIX", "SMAD2": "SMAD2|SMAD3|SMAD"},
}

TFSET = os.environ.get("SOX32_TFSET", "nodal")
NAMED_TF = TF_SETS[TFSET]
TF_COLORS = _TF_SET_COLORS[TFSET]
NAMED_TF_VIERSTRA_QUERY = _TF_SET_VQUERY[TFSET]
# FIMO motif ids can't contain "/", so use safe names
FIMO_TF_NAMES = {label: label.replace("/", "_") for label in NAMED_TF}
SET_SUFFIX = "" if TFSET == "nodal" else f"_{TFSET}"         # keeps default outputs unchanged
FIMO_DIR = f"outputs/fimo{SET_SUFFIX}"
FIMO_HITS_CSV = f"{FIMO_DIR}/sox32_fimo_hits.csv"

# annotation passes for seqlet labeling (name -> MEME path or {label:id} dict)
ANNOTATION_DBS = {
    "catalog": CHROMBPNET_CATALOG_MEME,
    "named3": NAMED_TF,
    "vierstra": VIERSTRA_CONSENSUS_MEME,
}

MEME_FIMO_BIN = "/cellar/users/aklie/opt/meme/bin/fimo"

# ------------------------------------------------------------------ plotting
def setup_mpl():
    """Headless backend + editable-text PDFs + white style. Call at script top."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("white")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    return plt
