"""
Unit & Integration Tests for End-to-End Satellite Damage Inference Pipeline
"""

import os
import base64
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from ml.src.inference.engine import DamageInferenceEngine, inference_engine
from backend.app.main import app

client = TestClient(app)


@pytest.fixture
def sample_tile_pair():
    """Generates synthetic pre- and post-disaster satellite tiles for testing"""
    pre = np.full((128, 128, 3), 70, dtype=np.uint8)
    post = np.full((128, 128, 3), 70, dtype=np.uint8)

    # Intact building on pre
    cv2.rectangle(pre, (20, 20), (60, 60), (180, 140, 120), -1)

    # Destroyed building on post
    cv2.rectangle(post, (20, 20), (60, 60), (60, 50, 45), -1)

    _, pre_png = cv2.imencode(".png", cv2.cvtColor(pre, cv2.COLOR_RGB2BGR))
    _, post_png = cv2.imencode(".png", cv2.cvtColor(post, cv2.COLOR_RGB2BGR))

    return pre_png.tobytes(), post_png.tobytes()


def test_decode_image_formats(sample_tile_pair):
    engine = DamageInferenceEngine()
    pre_bytes, _ = sample_tile_pair

    # 1. Decode from raw bytes
    decoded = engine.decode_image(pre_bytes)
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (128, 128, 3)

    # 2. Decode from base64 data URI
    b64_uri = f"data:image/png;base64,{base64.b64encode(pre_bytes).decode('utf-8')}"
    decoded_b64 = engine.decode_image(b64_uri)
    assert decoded_b64.shape == (128, 128, 3)

    # 3. Decode from numpy array passthrough
    decoded_arr = engine.decode_image(decoded)
    assert decoded_arr.shape == (128, 128, 3)

    # 4. Corrupt input rejection
    with pytest.raises(ValueError):
        engine.decode_image(b"not-a-valid-image-stream-12345")


def test_validate_and_align_pair():
    engine = DamageInferenceEngine()

    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((120, 100, 3), dtype=np.uint8)

    # Aligns mismatched dimensions
    p1, p2 = engine.validate_and_align_pair(img1, img2)
    assert p1.shape == p2.shape == (100, 100, 3)

    # Rejects excessively small tiles
    too_small = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="smaller than required minimum"):
        engine.validate_and_align_pair(too_small, img1)


def test_model_status_reporting():
    status = inference_engine.get_model_status()
    assert "model_version" in status
    assert "weights_status" in status
    assert "notice" in status
    assert status["weights_status"] in ["CALIBRATED_CHECKPOINT", "UNTRAINED_DEMO_INITIALIZATION"]


def test_end_to_end_inference_pipeline(sample_tile_pair):
    engine = DamageInferenceEngine()
    pre_bytes, post_bytes = sample_tile_pair

    result = engine.run_inference_pipeline(
        pre_image=pre_bytes,
        post_image=post_bytes,
        confidence_threshold=0.60,
        meters_per_pixel=0.5,
        origin_lat_lon=(12.3375, 75.8062),
    )

    assert result["status"] == "SUCCESS"
    assert "summary" in result
    assert "model_info" in result
    assert "segmentation_mask_base64" in result
    assert result["segmentation_mask_base64"].startswith("data:image/png;base64,")

    summary = result["summary"]
    assert "total_buildings_detected" in summary
    assert "damage_breakdown" in summary
    assert "damage_rate_percentage" in summary

    features = result["features"]
    assert len(features) > 0
    for f in features:
        assert f["type"] == "Feature"
        assert f["geometry"]["type"] == "Polygon"
        props = f["properties"]
        assert "damage_class" in props
        assert "damage_color" in props
        assert "confidence" in props
        assert "area_sqm" in props


def test_api_damage_assessment_with_imagery(sample_tile_pair):
    pre_bytes, post_bytes = sample_tile_pair
    pre_b64 = f"data:image/png;base64,{base64.b64encode(pre_bytes).decode('utf-8')}"
    post_b64 = f"data:image/png;base64,{base64.b64encode(post_bytes).decode('utf-8')}"

    payload = {
        "pre_image_base64": pre_b64,
        "post_image_base64": post_b64,
        "confidence_threshold": 0.65,
        "meters_per_pixel": 0.50,
        "origin_lat": 12.3375,
        "origin_lon": 75.8062,
        "apply_to_project": False,
    }

    res = client.post("/api/projects/proj-karnataka-wghats-2026/damage-assessment", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "summary" in data
    assert "segmentation_mask_base64" in data
    assert len(data["features"]) > 0


def test_api_damage_assessment_invalid_payload():
    payload = {
        "pre_image_base64": "invalid-base64-data",
        "post_image_base64": "invalid-base64-data",
    }
    res = client.post("/api/projects/proj-karnataka-wghats-2026/damage-assessment", json=payload)
    assert res.status_code == 400


def test_api_model_status():
    res = client.get("/api/models/damage-assessment/status")
    assert res.status_code == 200
    data = res.json()
    assert "weights_status" in data
    assert "localization_architecture" in data


def test_api_sample_tiles():
    res = client.get("/api/samples/satellite-tiles")
    assert res.status_code == 200
    data = res.json()
    assert "pre_image_base64" in data
    assert "post_image_base64" in data
    assert data["pre_image_base64"].startswith("data:image/png;base64,")
    assert data["post_image_base64"].startswith("data:image/png;base64,")

