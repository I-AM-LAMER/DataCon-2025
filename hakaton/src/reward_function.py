import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from sascorer import calculateScore  

rf_bace1 = joblib.load("models/rf_bace1.joblib")
rf_gsk3b = joblib.load("models/rf_gsk3b.joblib")
rf_ache  = joblib.load("models/rf_ache.joblib")

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
    arr = np.array(fp)
    return arr.astype(float)

def score_smiles(smiles: str) -> float:
    fp = smiles_to_fp(smiles).reshape(1, -1)
    preds = [
        rf_bace1.predict(fp)[0],
        rf_gsk3b.predict(fp)[0],
        rf_ache.predict(fp)[0]
    ]

    norm = [(p - 4) / 4 for p in preds]  
    mean_act = np.mean(norm)
    qed = QED.qed(Chem.MolFromSmiles(smiles))
    sa  = 1 - calculateScore(Chem.MolFromSmiles(smiles))
    return float(0.6*mean_act + 0.2*qed + 0.2*sa)
