from chembl_webresource_client.new_client import new_client
import pandas as pd

def load_chembl_activity(target_chembl_id: str) -> pd.DataFrame:
    activity = new_client.activity
    records = (
        activity
        .filter(target_chembl_id=target_chembl_id)
        .filter(standard_type="IC50")
        .only(["molecule_chembl_id", "canonical_smiles", "standard_value"])
    )
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["smiles", "ic50"])
    df = df.rename(columns={"canonical_smiles": "smiles", "standard_value": "ic50"})
    df = df.loc[:, ["smiles", "ic50"]]
    df = df.dropna(subset=["smiles", "ic50"]).drop_duplicates(["smiles"])
    df["ic50"] = pd.to_numeric(df["ic50"], errors="coerce")
    df = df.dropna(subset=["ic50"])
    return df


