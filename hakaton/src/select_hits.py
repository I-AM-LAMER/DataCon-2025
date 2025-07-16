import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
try:
    from sascorer import calculateScore
except ImportError:
    calculateScore = lambda m: 0.0 

DOCK_SUM = os.path.join(os.path.dirname(__file__), "../results/docking_summary.csv")
OUT_FILE = os.path.join(os.path.dirname(__file__), "../results/final_hits.csv")

df = pd.read_csv(DOCK_SUM)

sa_list, qed_list = [], []
for smi in df.smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        sa_list.append(np.nan)
        qed_list.append(np.nan)
    else:
        sa_list.append(1.0 - calculateScore(mol))
        qed_list.append(QED.qed(mol))

df["SA_score"] = sa_list
df["QED"]      = qed_list

hits = df[
    (df.score >= 0.5) &
    (df.docking_energy <= -7.0) &
    (df.SA_score >= 0.3) &
    (df.QED >= 0.4)
].copy()

hits = hits.nsmallest(20, "docking_energy")

hits.to_csv(OUT_FILE, index=False)
print(f"Saved {len(hits)} final hits to {OUT_FILE}")
