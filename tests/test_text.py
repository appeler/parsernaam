"""Fixed expected preprocessing and packed-state contracts."""

import pytest
import torch

from parsernaam.model import LSTM
from parsernaam.text import VOCABULARY, encode, normalize
from training.legacy import encode_legacy


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("JOSÉ", "Jose"),
        ("K. Rao", "K Rao"),
        ("o'connor", "O'Connor"),
        ("देवनागरी", ""),
        ("李明", ""),
        ("\t John   SMITH \n", "John Smith"),
        ("---", ""),
        ("François Müller", "Francois Muller"),
        ("Anne-Marie", "Anne-Marie"),
        ("123!", ""),
        ("Straße", "Strasse"),
        ("Groẞ", "Gross"),
        ("Iş\u0131k", "Isik"),
        ("\u0131", "I"),
    ],
)
def test_normalization(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    "name",
    [
        "José García",
        "O'Connor",
        "K. Rao",
        "Anne-Marie",
        "John Smith",
        "Straße",
        "Groẞ",
        "Iş\u0131k",
        "\u0131",
    ],
)
def test_case_normalization(name):
    assert normalize(name) == normalize(name.upper()) == normalize(name.lower())


def test_encoding_truncates_and_reserves_padding_unknown():
    tokens = encode(["Abcdef", "A?", ""], 3)
    assert tokens.shape == (3, 3)
    assert torch.equal(tokens[0], encode(["Abc"], 3)[0])
    assert tokens[1].tolist() == [VOCABULARY.index("A") + 2, 1, 0]
    assert tokens[2].tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="positive"):
        encode(["John"], 0)


def test_legacy_encoding_contract():
    assert encode_legacy(["Ab?"], 5, "Ab").tolist() == [[0, 1, 3, 3, 3]]


def test_packed_state_does_not_depend_on_trailing_padding():
    torch.manual_seed(13)
    model = LSTM(len(VOCABULARY) + 2, 16, 2, 2).eval()
    with torch.inference_mode():
        assert torch.allclose(
            model(encode(["John"], 5)), model(encode(["John"], 47)), atol=1e-7
        )
        names = ["John", "Smith", "O'Connor"]
        batch = model(encode(names, 47))
        singles = torch.cat([model(encode([n], 47)) for n in names])
        assert torch.allclose(batch, singles, atol=1e-7)
