from __future__ import annotations

from typing import Iterable

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - rdkit is a declared dependency
    Chem = None

SMILES_FIELDS = ("smiles", "SMILES", "canonical_smiles", "generated_smiles")


def extract_smiles(row: dict) -> str | None:
    """Return the first supported SMILES field from a record."""
    for field in SMILES_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def canonicalize_smiles(smiles: str | None, *, isomeric: bool = True) -> str | None:
    """RDKit canonicalization. Returns None for invalid/empty SMILES."""
    if smiles is None:
        return None
    smiles = str(smiles).strip()
    if not smiles or Chem is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except Exception:
        return None


def canonicalize_many(smiles_list: Iterable[str], *, keep_invalid: bool = False) -> list[str | None] | list[str]:
    values = [canonicalize_smiles(s) for s in smiles_list]
    if keep_invalid:
        return values
    return [s for s in values if s is not None]
