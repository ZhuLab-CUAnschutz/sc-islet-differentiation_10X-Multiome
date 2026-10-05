"""Step 7.1 — classify motifs in motif_compendium_cluster/ per Liu et al. 2026 Fig 3d.

Categories (per paper Fig 3d):
  base               — single canonical TF binding site
  base_with_flank    — canonical core + meaningful (non-TF-matching) flank
  homocomposite      — two+ adjacent sites of the SAME TF family
  heterocomposite    — two+ adjacent sites of DIFFERENT TF families
  unresolved         — no Vierstra hit at the loose threshold
  repeat             — low-complexity / repetitive (very high copair_composition)
  partial            — short truncation of a canonical site (W < 12 effective)

Algorithm:
  1. Run TomTom against Vierstra v2.0beta archetypes at LOOSE E (E<=100)
     so that secondary hits within a single catalog motif are captured.
  2. For each catalog motif, group its hits by aligned offset into "slots"
     (cluster offsets within +/-2bp). Each slot = one putative sub-motif.
  3. Pull the dominant TF family for each slot via _tf_family() (strips
     Vierstra suffixes and digit-tails from the TF token).
  4. Classify on slot count + family overlap + effective width.

Output:
  results/.../motif_compendium_cluster/qc/motif_classification.tsv
    columns: name, posneg, category, n_slots, slot_families, effective_width,
             primary_family, primary_subname, primary_evalue
"""
import argparse, re, gzip, shutil, subprocess, sys, tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


def _tf_family(vierstra_target: str) -> str:
    """Strip Vierstra suffixes/digits to TF family stub.
    AC0001.1 -> AC, NRF1_NRF_1 -> NRF, ETS+RUNX -> ETS,RUNX.
    """
    if not isinstance(vierstra_target, str) or not vierstra_target:
        return ""
    head = vierstra_target.split(".")[0].split("_")[0]
    parts = [p for p in head.split("+") if p]
    fams = []
    for p in parts:
        i = len(p)
        while i > 0 and p[i - 1].isdigit():
            i -= 1
        stub = p[:i] if i > 0 else p
        fams.append(stub.upper())
    return ",".join(sorted(set(fams)))


def _run_tomtom(query_meme: Path, db_meme: Path, out_dir: Path, evalue: float) -> Path:
    """Run TomTom at loose E threshold. Returns path to tomtom.tsv. Caches by file presence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tt = out_dir / "tomtom.tsv"
    if tt.exists() and tt.stat().st_size > 0:
        print(f"[cache] reusing {tt}", flush=True)
        return tt
    cmd = [
        "tomtom", "-no-ssc", "-oc", str(out_dir), "-verbosity", "1",
        "-min-overlap", "5", "-dist", "pearson",
        "-evalue", "-thresh", str(evalue),
        str(query_meme), str(db_meme),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return tt


def _slotify(group: pd.DataFrame, slot_tol: int = 2) -> list:
    """Group hits by aligned query offset into slots. Returns list of slot dicts."""
    if group.empty:
        return []
    g = group.sort_values("Optimal_offset").reset_index(drop=True)
    slots = []  # each: {offsets, hits}
    cur_off = None
    cur = None
    for _, row in g.iterrows():
        off = int(row["Optimal_offset"])
        if cur is None or abs(off - cur_off) > slot_tol:
            cur = {"offsets": [off], "hits": [row]}
            slots.append(cur)
            cur_off = off
        else:
            cur["offsets"].append(off)
            cur["hits"].append(row)
            cur_off = int(np.median(cur["offsets"]))
    return slots


def _effective_width(meme_block: list) -> int:
    """Width with >=0.5 max-letter probability — strips uninformative flanks."""
    rows = []
    for ln in meme_block:
        parts = ln.strip().split()
        if len(parts) == 4:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass
    if not rows:
        return 0
    arr = np.array(rows)
    informative = (arr.max(axis=1) >= 0.5)
    if not informative.any():
        return 0
    lo, hi = np.where(informative)[0][[0, -1]]
    return int(hi - lo + 1)


def _classify(name: str, slots: list, effective_w: int, full_w: int, posneg: str) -> dict:
    """Apply Liu 2026 Fig 3d-style decision tree."""
    fams_per_slot = []
    primary_family, primary_sub, primary_e = "", "", float("nan")
    for s in slots:
        hits = sorted(s["hits"], key=lambda h: float(h["E-value"]))
        top = hits[0]
        fam = _tf_family(str(top["Target_ID"]))
        fams_per_slot.append(fam)
        if primary_family == "":
            primary_family = fam
            primary_sub = str(top["Target_ID"])
            primary_e = float(top["E-value"])

    n_slots = len(slots)
    if n_slots == 0:
        if full_w >= 12 and effective_w < full_w * 0.5:
            cat = "repeat"
        elif effective_w < 8:
            cat = "partial"
        else:
            cat = "unresolved"
    elif n_slots == 1:
        slot_off = int(np.median(slots[0]["offsets"]))
        slot_top = sorted(slots[0]["hits"], key=lambda h: float(h["E-value"]))[0]
        sub_w = int(slot_top.get("Optimal_offset", 0)) if False else None  # unused
        flank_left = slot_off
        flank_right = max(0, full_w - slot_off - 12)  # 12 = canonical core
        cat = "base_with_flank" if (flank_left + flank_right) >= 6 else "base"
    else:
        uniq_fams = {f for f in fams_per_slot if f}
        cat = "homocomposite" if len(uniq_fams) <= 1 else "heterocomposite"

    return {
        "name": name,
        "posneg": posneg,
        "category": cat,
        "n_slots": n_slots,
        "slot_families": "|".join(fams_per_slot),
        "effective_width": effective_w,
        "full_width": full_w,
        "primary_family": primary_family,
        "primary_subname": primary_sub,
        "primary_evalue": primary_e,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog_dir", required=True)
    p.add_argument("--vierstra_meme", required=True)
    p.add_argument("--evalue", type=float, default=100.0)
    p.add_argument("--slot_tol", type=int, default=2)
    args = p.parse_args()

    cdir = Path(args.catalog_dir)
    meme_path = cdir / "combined_mapped.meme"
    out_tsv = cdir / "qc" / "motif_classification.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    tomtom_dir = cdir / "tomtom" / "vierstra_loose"

    # Index posneg via the metadata TSV
    meta_path = cdir / "08_final_averages_metadata.tsv"
    meta = pd.read_csv(meta_path, sep="\t")
    # match on the export-time display name (column 'name' or motif_string fallback);
    # but the MEME uses display names matching what we wrote into name_map — use the
    # name_map to look up posneg via its prefix (pos_/neg_/cluster_final#)
    name_map_path = cdir / "clustered_motifs_name_map_unique.tsv"
    nm = pd.read_csv(name_map_path, sep="\t", header=None, names=["h5_key", "display"])
    nm["posneg"] = nm["h5_key"].str.split(".").str[0].str.replace("_patterns", "")
    posneg_map = dict(zip(nm["display"], nm["posneg"]))

    # Parse each MEME motif block to capture per-motif PWM for width calc
    text = meme_path.read_text()
    blocks = re.split(r'\nMOTIF\s+', text)[1:]
    motif_blocks = {}
    motif_order = []
    for blk in blocks:
        first = blk.splitlines()[0].split()
        name = first[0]
        motif_order.append(name)
        motif_blocks[name] = blk.splitlines()

    # Run TomTom at loose E
    tt_tsv = _run_tomtom(meme_path, Path(args.vierstra_meme), tomtom_dir, args.evalue)
    # Do NOT pass comment='#' to read_csv: tomtom uses '#' inside motif names
    # (e.g. cluster_final#42) and the comment kwarg drops everything after '#'.
    # Strip trailing comment lines manually instead.
    raw = [ln for ln in tt_tsv.read_text().splitlines()
           if ln.strip() and not ln.lstrip().startswith("#")]
    from io import StringIO
    tt = pd.read_csv(StringIO("\n".join(raw)), sep="\t")
    tt.columns = [c.replace("#", "").strip() for c in tt.columns]
    print(f"TomTom rows (all E): {len(tt)} across {tt['Query_ID'].nunique()} queries", flush=True)
    # Filter to strict E for slot detection: E<=100 (the run threshold) is too
    # permissive — 25K rows / 151 queries = ~167 noise hits/motif, every motif
    # becomes "heterocomposite". E<=0.1 is the Vierstra-loose-but-real bar.
    tt_strict = tt[tt["E-value"] <= 0.1].copy()
    # Dedupe within each query by (offset-band, TF family) so that 10 nearly-
    # identical archetype hits at the same slot count as one — best-E wins.
    tt_strict["_band"] = (tt_strict["Optimal_offset"] // 3).astype(int)
    tt_strict["_fam"] = tt_strict["Target_ID"].map(_tf_family)
    tt_strict = (tt_strict.sort_values("E-value")
                          .drop_duplicates(["Query_ID", "_band", "_fam"])
                          .drop(columns=["_band", "_fam"]))
    print(f"After E<=0.1 + per-query (band,family) dedupe: {len(tt_strict)} rows "
          f"across {tt_strict['Query_ID'].nunique()} queries", flush=True)
    # Group once instead of per-motif full-frame filtering
    by_query = dict(list(tt_strict.groupby("Query_ID")))

    # Classify each catalog motif
    out_rows = []
    for i, name in enumerate(motif_order):
        hits = by_query.get(name, tt.iloc[0:0])
        slots = _slotify(hits, slot_tol=args.slot_tol)
        # full width from MEME header line
        header_ln = motif_blocks[name][0] if motif_blocks[name][0].startswith("letter") else motif_blocks[name][1]
        m = re.search(r"w=\s*(\d+)", header_ln)
        full_w = int(m.group(1)) if m else 0
        eff_w = _effective_width(motif_blocks[name])
        posneg = posneg_map.get(name, "?")
        out_rows.append(_classify(name, slots, eff_w, full_w, posneg))
        if (i + 1) % 25 == 0 or i + 1 == len(motif_order):
            print(f"  classified {i+1}/{len(motif_order)}", flush=True)

    df = pd.DataFrame(out_rows)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"wrote {out_tsv} ({len(df)} rows)")
    print("category counts:")
    print(df["category"].value_counts().to_string())
    print()
    print("per posneg:")
    print(df.groupby(["posneg", "category"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
