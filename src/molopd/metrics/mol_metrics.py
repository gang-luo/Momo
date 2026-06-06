from __future__ import annotations

from itertools import combinations
from typing import Iterable

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, QED


def _safe_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def compute_basic_generation_metrics(smiles_list: Iterable[str], train_smiles_set: set[str] | None = None) -> dict[str, float]:
    smiles = [s.strip() for s in smiles_list if isinstance(s, str) and s.strip()]
    total = len(smiles)
    if total == 0:
        return {
            "validity": 0.0,
            "uniqueness": 0.0,
            "novelty": 0.0,
            "diversity": 0.0,
            "qed_mean": 0.0,
            "logp_mean": 0.0,
            "mw_mean": 0.0,
            "tpsa_mean": 0.0,
        }

    mols = [_safe_mol(s) for s in smiles]
    valid_pairs = [(s, m) for s, m in zip(smiles, mols) if m is not None]
    valid_smiles = [s for s, _ in valid_pairs]
    valid_mols = [m for _, m in valid_pairs]

    validity = len(valid_smiles) / total
    uniqueness = (len(set(valid_smiles)) / len(valid_smiles)) if valid_smiles else 0.0

    if train_smiles_set is None:
        novelty = 0.0
    else:
        novelty = (sum(1 for s in valid_smiles if s not in train_smiles_set) / len(valid_smiles)) if valid_smiles else 0.0

    if len(valid_mols) >= 2:
        fps = [_fp(m) for m in valid_mols]
        sims = [DataStructs.TanimotoSimilarity(a, b) for a, b in combinations(fps, 2)]
        diversity = 1.0 - (sum(sims) / len(sims)) if sims else 0.0
    else:
        diversity = 0.0

    qed_mean = sum(QED.qed(m) for m in valid_mols) / len(valid_mols) if valid_mols else 0.0
    logp_mean = sum(Crippen.MolLogP(m) for m in valid_mols) / len(valid_mols) if valid_mols else 0.0
    mw_mean = sum(Descriptors.MolWt(m) for m in valid_mols) / len(valid_mols) if valid_mols else 0.0
    tpsa_mean = sum(Descriptors.TPSA(m) for m in valid_mols) / len(valid_mols) if valid_mols else 0.0

    return {
        "validity": float(validity),
        "uniqueness": float(uniqueness),
        "novelty": float(novelty),
        "diversity": float(diversity),
        "qed_mean": float(qed_mean),
        "logp_mean": float(logp_mean),
        "mw_mean": float(mw_mean),
        "tpsa_mean": float(tpsa_mean),
    }
