# docking.py
import os
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

LIGANDS = "results/generated.smi"
RECEPTOR = "data/ache.pdbqt"
OUT_DIR  = "results/docking"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(LIGANDS, sep="\t", names=["smiles","score"])
top = df.sort_values("score", ascending=False).head(200)

for idx, row in top.iterrows():
    smi = row.smiles
    mol = Chem.MolFromSmiles(smi)
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
    fout = os.path.join(OUT_DIR, f"lig_{idx}.pdb")
    Chem.MolToPDBFile(mol_h, fout)

    ligand_pdbqt = fout.replace(".pdb",".pdbqt")
    subprocess.run([
        "obabel", fout, "-O", ligand_pdbqt, "--partialcharge", "gasteiger"
    ])
    out_pdbqt = ligand_pdbqt.replace(".pdbqt", "_out.pdbqt")
    subprocess.run([
        "vina",
        "--receptor", RECEPTOR,
        "--ligand", ligand_pdbqt,
        "--center_x", "10", "--center_y", "25", "--center_z", "15",
        "--size_x", "20", "--size_y", "20", "--size_z", "20",
        "--out", out_pdbqt
    ])
print("Docking finished, results in", OUT_DIR)
