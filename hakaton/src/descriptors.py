import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

try:
    from mordred import Calculator, descriptors as mordred_descriptors
except ImportError:
    raise ImportError("Mordred не найден — установите mordred[full]")

def compute_rdkit_fps(df: pd.DataFrame, radius=2, n_bits=2048) -> pd.DataFrame:
    fps = []
    for smi in df.smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(fp.ToBitString())
    arr = [list(map(int, list(bits))) for bits in fps]
    return pd.DataFrame(arr, columns=[f"fp_{i}" for i in range(n_bits)], index=df.index)

def compute_mordred_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    mols = [Chem.MolFromSmiles(smi) for smi in df.smiles]
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    descs = calc.pandas(mols)
    descs.index = df.index
    return descs
