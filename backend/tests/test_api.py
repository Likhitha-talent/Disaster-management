"""
Unit and Integration Tests for Disaster Decision-Support Backend
"""

import os
import json
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_get_projects():
    response = client.get("/api/projects")
    assert response.status_code == 200
    projects = response.json()
    assert len(projects) > 0
    assert "summary" in projects[0]


def test_get_buildings():
    response = client.get("/api/projects/proj-karnataka-wghats-2026/buildings")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert len(data["features"]) == 50


def test_building_filter():
    response = client.get("/api/projects/proj-karnataka-wghats-2026/buildings?damage_class=destroyed")
    assert response.status_code == 200
    data = response.json()
    features = data["features"]
    assert len(features) > 0
    for f in features:
        assert f["properties"]["damage_class"] == "destroyed"


def test_relocation_analysis():
    response = client.get("/api/projects/proj-karnataka-wghats-2026/relocation-sites")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    features = data["features"]
    # Check that rank 1 exists and has highest TOPSIS score
    ranked = [f for f in features if f["properties"].get("rank") == 1]
    assert len(ranked) == 1
    assert ranked[0]["properties"]["topsis_score"] > 0.0


def test_hazard_zones():
    response = client.get("/api/projects/proj-karnataka-wghats-2026/risk-zones")
    assert response.status_code == 200
    data = response.json()
    assert len(data["features"]) >= 3


def test_post_risk_analysis():
    # 1. Baseline analysis with standard weights
    payload_1 = {
        "damage_density": 0.35,
        "flood_hydro_exposure": 0.25,
        "slope_landslide": 0.18,
        "land_cover_hazard": 0.12,
        "population_density": 0.10,
        "red_threshold": 0.65,
        "amber_threshold": 0.35,
    }
    res1 = client.post("/api/projects/proj-karnataka-wghats-2026/risk-analysis", json=payload_1)
    assert res1.status_code == 200
    data1 = res1.json()
    assert "metadata" in data1
    assert "features" in data1
    meta1 = data1["metadata"]
    assert "consistency_ratio" in meta1
    assert "is_consistent" in meta1
    assert isinstance(meta1["consistency_ratio"], float)
    assert meta1["consistency_ratio"] < 0.10
    assert meta1["is_consistent"] is True

    score_red_1 = data1["features"][0]["properties"]["composite_risk_score"]

    # 2. Shift weights heavily towards flood and verify score recalculation
    payload_2 = {
        "damage_density": 0.10,
        "flood_hydro_exposure": 0.55,
        "slope_landslide": 0.15,
        "land_cover_hazard": 0.10,
        "population_density": 0.10,
        "red_threshold": 0.65,
        "amber_threshold": 0.35,
    }
    res2 = client.post("/api/projects/proj-karnataka-wghats-2026/risk-analysis", json=payload_2)
    assert res2.status_code == 200
    data2 = res2.json()
    score_red_2 = data2["features"][0]["properties"]["composite_risk_score"]
    assert score_red_1 != score_red_2


def test_post_relocation_analysis():
    # 1. Default parameters: slope limit 16 degrees
    res_default = client.post("/api/projects/proj-karnataka-wghats-2026/relocation-analysis", json={
        "planning_area_per_person_sqm": 30.0,
        "usable_area_fraction": 0.65,
        "max_allowable_slope_deg": 16.0,
        "min_hazard_buffer_meters": 250.0,
    })
    assert res_default.status_code == 200
    data_default = res_default.json()
    default_eligible = data_default["metadata"]["eligible_count"]
    default_capacity = data_default["metadata"]["total_relocation_capacity_persons"]

    # 2. Restrictive slope limit: 6 degrees should exclude steeper candidate sites
    res_restricted = client.post("/api/projects/proj-karnataka-wghats-2026/relocation-analysis", json={
        "planning_area_per_person_sqm": 30.0,
        "usable_area_fraction": 0.65,
        "max_allowable_slope_deg": 6.0,
        "min_hazard_buffer_meters": 250.0,
    })
    assert res_restricted.status_code == 200
    data_restricted = res_restricted.json()
    restricted_eligible = data_restricted["metadata"]["eligible_count"]
    restricted_capacity = data_restricted["metadata"]["total_relocation_capacity_persons"]

    assert restricted_eligible < default_eligible
    assert restricted_capacity < default_capacity

    # 3. Verify project summary endpoint reflects the newly calculated relocation capacity
    proj_res = client.get("/api/projects/proj-karnataka-wghats-2026")
    assert proj_res.status_code == 200
    summary = proj_res.json()["summary"]
    assert summary["total_relocation_capacity_available"] == restricted_capacity
    assert summary["eligible_relocation_sites"] == restricted_eligible


def test_building_review_persistence_and_summary_update(tmp_path, monkeypatch):
    import backend.app.services.project_service as ps_module
    import backend.app.api.routes as routes_module
    from backend.app.services.project_service import ProjectDataStore

    # Temporary review store file for clean test isolation
    test_review_file = str(tmp_path / "test_reviewed_buildings.json")
    monkeypatch.setattr(ps_module, "REVIEW_STORE_FILE", test_review_file)

    # Initialize isolated store instance
    test_store = ProjectDataStore()
    monkeypatch.setattr(ps_module, "store", test_store)
    monkeypatch.setattr(routes_module, "store", test_store)

    initial_summary = test_store.get_summary_statistics()
    initial_flagged = initial_summary["flagged_for_human_review"]

    # Target BLD-0007 (flagged for review)
    bldg_id = "BLD-0007"
    review_payload = {
        "verified_damage_class": "minor_damage",
        "reviewer_notes": "Ground verification confirms minor damage only",
        "reviewer_id": "OFFICER-TEST-99"
    }

    res = client.post(f"/api/projects/proj-karnataka-wghats-2026/buildings/{bldg_id}/review", json=review_payload)
    assert res.status_code == 200
    updated_bldg = res.json()["building"]["properties"]
    assert updated_bldg["damage_class"] == "minor_damage"
    assert updated_bldg["review_status"] == "VERIFIED_BY_ANALYST"
    assert updated_bldg["needs_review"] is False

    # Check updated summary
    updated_summary = test_store.get_summary_statistics()
    assert updated_summary["flagged_for_human_review"] == initial_flagged - 1

    # Check file persistence on disk
    assert os.path.exists(test_review_file)
    with open(test_review_file, "r", encoding="utf-8") as f:
        stored_data = json.load(f)
    assert bldg_id in stored_data
    assert stored_data[bldg_id]["verified_damage_class"] == "minor_damage"

    # Verify reloading from disk restores verified state
    reloaded_store = ProjectDataStore()
    reloaded_bldg = next(b for b in reloaded_store.buildings_data if b["properties"]["building_id"] == bldg_id)
    assert reloaded_bldg["properties"]["damage_class"] == "minor_damage"
    assert reloaded_bldg["properties"]["review_status"] == "VERIFIED_BY_ANALYST"
    assert reloaded_bldg["properties"]["needs_review"] is False
