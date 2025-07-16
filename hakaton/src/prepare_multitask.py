import os
import pandas as pd
from data_loader import load_chembl_activity
from descriptors import compute_rdkit_fps, compute_mordred_descriptors

OUT_PATH = os.path.join(os.path.dirname(__file__), "../data/multitask_clean.csv")

targets = {
    "bace1": "CHEMBL4822",
    "gsk3b": "CHEMBL262",
    "ache": "CHEMBL220"
}

dfs = []
for name, chembl_id in targets.items():
    df = load_chembl_activity(chembl_id).rename(columns={"ic50": f"ic50_{name}"})
    df[f"ic50_{name}"] = pd.to_numeric(df[f"ic50_{name}"], errors="coerce")
    df = df.dropna(subset=[f"ic50_{name}"])
    df = df.set_index("smiles")
    dfs.append(df)

multitask_df = pd.concat(dfs, axis=1, join="outer").reset_index()

fps = compute_rdkit_fps(multitask_df)
descs = compute_mordred_descriptors(multitask_df)

result = pd.concat([multitask_df, fps, descs], axis=1)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
result.to_csv(OUT_PATH, index=False)
