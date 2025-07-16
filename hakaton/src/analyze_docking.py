import os
import re
import pandas as pd

DOCK_DIR = os.path.join(os.path.dirname(__file__), "../results/docking")
GEN_FILE = os.path.join(os.path.dirname(__file__), "../results/generated.csv")
OUT_FILE = os.path.join(os.path.dirname(__file__), "../results/docking_summary.csv")

gen_df = pd.read_csv(GEN_FILE)

records = []
pattern = re.compile(r"^REMARK VINA RESULT:\s+([-0-9.]+)")

for fname in os.listdir(DOCK_DIR):
    if not fname.endswith("_out.pdbqt"):
        continue
    path = os.path.join(DOCK_DIR, fname)
    with open(path, "r") as f:
        energy = None
        for line in f:
            m = pattern.match(line)
            if m:
                energy = float(m.group(1))
                break

    idx = int(re.findall(r"lig(\d+)_out\.pdbqt", fname)[0])
    smiles, score = gen_df.iloc[idx][["smiles", "score"]]
    records.append({"idx": idx, "smiles": smiles, "score": score, "docking_energy": energy})

dock_df = pd.DataFrame(records).sort_values("docking_energy")
dock_df.to_csv(OUT_FILE, index=False)
print(f"Saved docking summary to {OUT_FILE}")
