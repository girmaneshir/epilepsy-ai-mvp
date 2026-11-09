import pytest
from src.translator import Translator

@pytest.fixture(scope="module")
def tr():
    return Translator()

@pytest.mark.slow
def test_roundtrip(tr):
    en   = "Addis Ababa is the capital of Ethiopia."
    am   = tr.translate(en, "en", "am")
    back = tr.translate(am, "am", "en")
    assert "addis" in back.lower()
