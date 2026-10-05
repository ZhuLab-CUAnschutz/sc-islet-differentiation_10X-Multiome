"""Rebuild MC cluster catalog naming after the prior fix-script bug.

What happened: my Phase 7.2 fix script (fix_mc_cluster_catalog.py) paired
sorted-alphabetically h5 keys (cluster_final#0, #10, #100, #101, #11, ...) with
display names parsed from step 5.3's MEME (which was written in cluster-id
NUMERICAL order: #0, #1, #2, ...). Result: every name_map row past index 0
got a mis-paired display name, and the rewritten MEME inherited the same
mis-pairing. This contaminated hit-call (since finemo reads the name_map),
marginalization (reads MEME), 7.1 classification, 7.2 aggregation, and 7.3
rollup.

Truth source: per-motif TomTom outputs `tomtom/cluster_final_N.tomtom.txt`
were written by step 5.3 keyed by the actual cluster id N. Walk those + h5
to rebuild name_map and MEME correctly:

  - For each cluster_final#N in h5:
      track = 'pos' or 'neg' from h5 group
      Read tomtom/cluster_final_N.tomtom.txt
      Pick top hit at E <= TOMTOM_E_THRESH (matches step 5.3's 10.0)
      If hit:   display = f"{track}_{target_id}"
      Else:     display = f"cluster_final#{N}"
  - Dedupe collisions with __cfN suffix

Rebuilds:
  combined_mapped.meme              (PPM from h5 CWMs, with correct display)
  clustered_motifs_name_map_unique.tsv

Stashes prior versions as *.broken-naming.bak.
"""
import re, shutil
from pathlib import Path
from collections import Counter
import h5py, numpy as np, pandas as pd

CAT = Path('/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/results/4_multi_task_model/crested/motif_compendium_cluster')
H5  = CAT / 'clustered_motifs.modisco.h5'
TOMTOM_DIR = CAT / 'tomtom'
TOMTOM_E_THRESH = 10.0       # matches step 5.3 strict-E for naming
MEME_OUT = CAT / 'combined_mapped.meme'
NM_OUT   = CAT / 'clustered_motifs_name_map_unique.tsv'


def top_hit_for_cluster(cid: int) -> str | None:
    """Return top Vierstra Target_ID at E <= threshold for cluster_final#cid, else None."""
    fp = TOMTOM_DIR / f"cluster_final_{cid}.tomtom.txt"
    if not fp.exists() or fp.stat().st_size == 0:
        return None
    # Strip trailing comment lines, then parse
    raw = [ln for ln in fp.read_text().splitlines()
           if ln.strip() and not ln.lstrip().startswith("#")]
    if len(raw) < 2:
        return None
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(raw)), sep="\t")
    if "E-value" not in df.columns or df.empty:
        return None
    sub = df[df["E-value"] <= TOMTOM_E_THRESH].sort_values("E-value")
    if sub.empty:
        return None
    return str(sub.iloc[0]["Target_ID"])


def cwm_to_ppm(cwm: np.ndarray) -> np.ndarray:
    """Clip CWM to non-negative + row-normalize. All-zero rows fall back to uniform."""
    ppm = np.clip(cwm, 0, None).astype(np.float64)
    rs = ppm.sum(axis=1, keepdims=True)
    zero_rows = (rs.squeeze() == 0)
    if zero_rows.any():
        ppm[zero_rows, :] = 0.25
        rs = ppm.sum(axis=1, keepdims=True)
    return ppm / rs


def main():
    # --- 1) build canonical mapping ---
    with h5py.File(H5, "r") as h:
        entries = []  # list of (track, cid_int, h5_key, target_or_None)
        for track in ["pos_patterns", "neg_patterns"]:
            if track not in h:
                continue
            t = "pos" if track == "pos_patterns" else "neg"
            for k in sorted(h[track].keys()):
                m = re.match(r"cluster_final#(\d+)", k)
                if not m:
                    continue
                cid = int(m.group(1))
                h5_key = f"{track}.{k}"
                target = top_hit_for_cluster(cid)
                entries.append((t, cid, h5_key, target))

    # Sort by cid (numerical) so the rebuilt MEME matches a sensible order.
    entries.sort(key=lambda r: r[1])

    # Build display names + dedupe
    raw_displays = [(f"{t}_{tgt}" if tgt else f"cluster_final#{cid}") for (t, cid, _, tgt) in entries]
    counts = Counter(raw_displays)
    final_displays = []
    for (t, cid, h5key, tgt), disp in zip(entries, raw_displays):
        if counts[disp] > 1:
            disp = f"{disp}__cf{cid}"
        final_displays.append(disp)

    # Sanity: no duplicates left
    assert len(set(final_displays)) == len(final_displays), \
        f"still have dups: {Counter(final_displays).most_common(5)}"

    # --- 2) stash broken files ---
    for fp in [MEME_OUT, NM_OUT]:
        if fp.exists():
            shutil.copy2(fp, fp.with_suffix(fp.suffix + ".broken-naming.bak"))
            print(f"  stashed {fp.name} -> {fp.name}.broken-naming.bak")

    # --- 3) write name_map ---
    nm_lines = []
    for (t, cid, h5key, tgt), disp in zip(entries, final_displays):
        nm_lines.append(f"{h5key}\t{disp}")
    NM_OUT.write_text("\n".join(nm_lines) + "\n")
    print(f"  wrote {NM_OUT.name} ({len(nm_lines)} rows)")

    # --- 4) rewrite MEME from h5 CWMs with new displays ---
    meme_lines = [
        "MEME version 4", "", "ALPHABET= ACGT", "", "strands: +", "",
        "Background letter frequencies:", "A 0.25 C 0.25 G 0.25 T 0.25", "",
    ]
    n_renorm = 0
    with h5py.File(H5, "r") as h:
        for (t, cid, h5key, tgt), disp in zip(entries, final_displays):
            track, _, key = h5key.partition(".")
            cwm = np.asarray(h[track][key]["contrib_scores"])
            ppm = cwm_to_ppm(cwm)
            W = ppm.shape[0]
            meme_lines.append(f"MOTIF {disp}")
            meme_lines.append(f"letter-probability matrix: alength= 4 w= {W}")
            for row in ppm:
                meme_lines.append(" ".join(f"{v:.6f}" for v in row))
            meme_lines.append("")
    MEME_OUT.write_text("\n".join(meme_lines) + "\n")
    print(f"  wrote {MEME_OUT.name}")

    # --- 5) diagnostic summary ---
    n_pos = sum(1 for e in entries if e[0] == "pos")
    n_neg = sum(1 for e in entries if e[0] == "neg")
    n_named = sum(1 for e in entries if e[3] is not None)
    n_dedup = sum(1 for d in final_displays if "__cf" in d)
    print()
    print(f"  pos motifs: {n_pos}  neg motifs: {n_neg}  total: {len(entries)}")
    print(f"  named via Vierstra TomTom (E <= {TOMTOM_E_THRESH}): {n_named}")
    print(f"  collisions deduped with __cf suffix: {n_dedup}")
    print()
    # spot-check the 4 known previously-mislabeled
    for cid in [1, 43, 55, 147]:
        rows = [(d, e) for e, d in zip(entries, final_displays) if e[1] == cid]
        if rows:
            d, e = rows[0]
            print(f"  cluster_final#{cid}  track={e[0]}  vierstra_top={e[3]}  display={d}")


if __name__ == "__main__":
    main()
