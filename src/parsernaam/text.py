"""Shared Latin-script normalization and character encoding."""

import re
import string
import unicodedata
from collections.abc import Sequence

import torch

VOCABULARY = string.ascii_letters + "' -"


def normalize(name: str) -> str:
    """Strip accents, retain Latin name characters, and title-case words.

    Args:
        name: Raw name string.

    Returns:
        Canonical name, or an empty string for unsupported input.
    """
    decomposed = unicodedata.normalize("NFKD", name.casefold().upper())
    unaccented = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = re.sub(r"[^A-Za-z' \s-]", "", unaccented)
    result = " ".join(cleaned.split()).title()
    return result if re.search("[A-Za-z]", result) else ""


def encode(
    names: Sequence[str], seq_len: int, vocabulary: str = VOCABULARY
) -> torch.Tensor:
    """Encode canonical names in a batch with padding 0 and unknown 1.

    Args:
        names: Names already passed through ``normalize``.
        seq_len: Maximum sequence length.
        vocabulary: Characters in artifact order, indexed from 2.

    Returns:
        Integer tensor with shape ``[len(names), seq_len]``.

    Raises:
        ValueError: If the maximum length is not positive.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    indices = {character: index + 2 for index, character in enumerate(vocabulary)}
    result = torch.zeros((len(names), seq_len), dtype=torch.long)
    for row, name in enumerate(names):
        values = [indices.get(character, 1) for character in name[:seq_len]]
        result[row, : len(values)] = torch.tensor(values, dtype=torch.long)
    return result
