# sc-islet-differentiation_10X-Multiome — Config

Configuration files and helper utilities for the SC-islet differentiation 10X Multiome dataset.

## Metadata Files (single source of truth)

| File | Description |
|------|-------------|
| `cell_type_metadata.tsv` | Cell types, groupings, colors, display order, stage, endocrine, dev_stage, lineage |
| `stage_metadata.tsv` | Differentiation stages, colors, and display order |
| `endocrine_metadata.tsv` | Endocrine classification, colors, and display order |
| `lineage_metadata.tsv` | Lineage facet (progenitor, endocrine_progenitor, beta, alpha, EC, delta, non_pancreatic): colors + order |
| `dev_stage_metadata.tsv` | Developmental-stage facet (DE, GT, PFG, PE, ENP, SC_immature, SC_mature, non_pancreatic): colors + order |
| `batch_metadata.tsv` | Differentiation batches, colors, and display order |
| `loader.py` | Python helper for loading configs and metadata |

### Cell types

| Group | Cell types |
|-------|------------|
| definitive_endoderm | DE |
| posterior_gut_tube | PGT1, PGT2, PGT3 |
| posterior_foregut | PFG1, PFG2 |
| pancreatic_progenitor | PP1, PP2 |
| endocrine_progenitor | ENP_phase1, early_ENP, late_ENP |
| SC_beta | early_SC_beta, late_SC_beta |
| SC_alpha | early_SC_alpha, late_SC_alpha |
| SC_EC | early_SC_EC, late_SC_EC |
| SC_delta | SC_delta_GHRL |
| proliferating | proliferating_endocrine |
| non_pancreatic | exocrine, liver, FB_FLT1 |

### Differentiation stages

D4, D7, D9, D12, D15, D22, D45

### Batches

HZ015, HZ019, HZ021, XW002

## Usage

```python
import sys
sys.path.insert(0, "/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome")
from config.loader import (
    load_yaml_config,
    list_configs,
    load_cell_type_metadata,
    load_cell_type_colors,
    load_grouping,
    get_group_members,
    get_group_names,
    load_stage_colors,
    load_endocrine_colors,
    load_batch_colors,
)

# --- Cell type metadata (single source of truth) ---
meta = load_cell_type_metadata()

# Colors and display order
palette = load_cell_type_colors()
colors = palette["celltype_colors"]   # {cell_type: hex}
order = palette["celltype_order"]     # [cell_type, ...]

# Grouping
grouping = load_grouping()
sc_beta = get_group_members(grouping, "SC_beta")
# ["early_SC_beta", "late_SC_beta"]

# --- Stage colors ---
stage = load_stage_colors()
stage_colors = stage["stage_colors"]  # {stage: hex}
stage_order = stage["stage_order"]    # [stage, ...]

# --- Endocrine colors ---
endo = load_endocrine_colors()

# --- Batch colors ---
batch = load_batch_colors()
```
