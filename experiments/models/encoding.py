"""Encode peptide tokens without changing residue case or modifications."""

import math

import numpy as np

from ..datasets import DatasetSpec

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TOKENS = ("<PAD>", "<UNK>", *AMINO_ACIDS, *AMINO_ACIDS.lower(), "P'")
TOKEN_IDS = {token: index for index, token in enumerate(TOKENS)}
PAD = TOKEN_IDS["<PAD>"]
UNK = TOKEN_IDS["<UNK>"]


def tokenize(sequence: str) -> list[str]:
    """Read P' as one residue and keep all other characters separate."""
    tokens = []
    index = 0
    while index < len(sequence):
        if sequence[index:index + 2] == "P'":
            tokens.append("P'")
            index += 2
        else:
            tokens.append(sequence[index])
            index += 1
    return tokens


def encode_rows(
    spec: DatasetSpec, rows: list[dict[str, str]], *, one_hot: bool = False
) -> np.ndarray:
    """Encode only declared task inputs.

    Args:
        spec: Task input and sequence length contract.
        rows: Task rows; extra columns are excluded.
        one_hot: Use positional one-hot tokens instead of integer tokens.

    Returns:
        A finite feature matrix with one row per supplied example.

    Raises:
        ValueError: If inputs are absent, non-finite, or exceed the length limit.
    """
    numeric = tuple(name for name in spec.inputs if name != "sequence")
    width = spec.max_length * (len(TOKENS) if one_hot else 1) + len(numeric)
    result = np.zeros((len(rows), width), dtype=np.float32)
    for row_index, row in enumerate(rows):
        if any(name not in row for name in spec.inputs):
            raise ValueError(f"Missing input for {spec.name}, row {row_index}")
        tokens = tokenize(row["sequence"])
        if not tokens or len(tokens) > spec.max_length:
            raise ValueError(f"Invalid sequence length for {spec.name}: {len(tokens)}")
        ids = [TOKEN_IDS.get(token, UNK) for token in tokens]
        ids.extend([PAD] * (spec.max_length - len(ids)))
        if one_hot:
            for position, token_id in enumerate(ids):
                result[row_index, position * len(TOKENS) + token_id] = 1
        else:
            result[row_index, :spec.max_length] = ids
        for offset, name in enumerate(numeric):
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite input: {name}")
            result[row_index, width - len(numeric) + offset] = value
    return result
