import numpy as np

from nucleus.webapp.server import create_app


def test_status_reports_v10_modalities():
    service = create_app()
    status = service.status()
    assert status["version"] == "1.0.0"
    assert status["modalities"] == ["text", "image", "audio"]


def test_image_service_accepts_8x8_input():
    service = create_app()
    pixels = np.zeros((8, 8)).tolist()
    result = service.handle("/api/image/classify", {"pixels": pixels})
    assert result["label"] in {"horizontal", "vertical"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_audio_service_accepts_waveform():
    service = create_app()
    t = np.arange(512) / 8000.0
    waveform = np.sin(2 * np.pi * 440.0 * t).tolist()
    result = service.handle("/api/audio/classify", {"waveform": waveform})
    assert result["label"] in {"440 Hz tone", "880 Hz tone"}
    assert 0.0 <= result["confidence"] <= 1.0
