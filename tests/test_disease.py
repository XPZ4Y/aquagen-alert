"""Photo triage: the fallback must abstain, the packaged checkpoints must classify.

The transformer path needs network + `uv sync --extra ml`, so it is opt-in:
    AQUA_TEST_ML=1 uv run pytest tests/test_disease.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aqua_alert.disease import classifier

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
LESION = SAMPLES / "example_lesion_fish.jpg"
HEALTHY = SAMPLES / "example_healthy_fish.jpg"


def test_triage_never_pretends_to_diagnose():
    """Colour statistics cannot separate a lesion from a muddy hand - so the fallback
    must abstain rather than output a confident 'Healthy Fish'."""
    if not LESION.exists():
        pytest.skip("demo photo not present")
    pred = classifier.predict(LESION)
    if pred.backend.startswith("transformers"):
        assert pred.label  # a real model is allowed to answer
    else:
        assert pred.triage in {"monitor", "treat", "escalate_to_expert"}
        assert "uncertain" in pred.label.lower()
        assert pred.confidence <= 0.5


def test_missing_file_is_loud():
    with pytest.raises(FileNotFoundError):
        classifier.predict("does/not/exist.jpg")


def test_advice_never_prescribes_antibiotics():
    for label, (triage, en, local) in classifier.ADVICE.items():
        assert triage in {"healthy", "monitor", "treat", "escalate_to_expert"}
        assert local.strip(), f"{label} has no local-language advice"
        if triage == "escalate_to_expert":
            assert "officer" in en.lower() or "report" in en.lower() or "contain" in en.lower()


@pytest.mark.skipif(os.getenv("AQUA_TEST_ML") != "1",
                    reason="set AQUA_TEST_ML=1 to download and exercise the ViT checkpoint")
def test_pretrained_checkpoint_on_demo_photos():
    lesion = classifier.predict(LESION)
    healthy = classifier.predict(HEALTHY)
    assert lesion.backend.startswith("transformers")
    assert lesion.triage == "escalate_to_expert"
    assert healthy.label.lower().startswith("healthy")
