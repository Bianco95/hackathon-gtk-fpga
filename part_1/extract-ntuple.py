"""Extract a small ntuple for the tutorial.

Reads the first NUM_EVENTS events of the full ntuple, keeps only the branches
used by the pre-processing and writes them to a new ROOT file with the same
tree path, so the notebook code works on both files unchanged.

Usage (from the repo root):
    python Tutorial/extract-ntuple.py
"""
from pathlib import Path

import uproot

#! Variables
NUM_EVENTS = 20_000
INPUT_PATH = "/home/leonardo/Desktop/ML_GTK/src/data/n-tuple/DataMay2024/dati_maggio24.root"
OUTPUT_PATH = Path(__file__).parent / "data" / "dati_maggio24_small.root"
TREE_PATH = "PnnOneTrackEventSelection/GTtree"
BRANCHES = [
    # per hit
    "cl_x", "cl_y", "cl_time", "GTKstation", "cl_kinepart",
    # per particle (truth)
    "set_cl_kinepart", "truthParent",
    # per event
    "KTAGTime", "eventNumber", "burst",
]

tree = uproot.open(INPUT_PATH)[TREE_PATH]
print(f"Input:  {INPUT_PATH} ({tree.num_entries} events)")

arrays = tree.arrays(BRANCHES, entry_stop=NUM_EVENTS)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with uproot.recreate(OUTPUT_PATH) as f:
    f[TREE_PATH] = {name: arrays[name] for name in BRANCHES}

print(f"Output: {OUTPUT_PATH} ({len(arrays)} events, {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")
