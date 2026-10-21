"""
Unit and Integration Tests for Disaster Decision-Support Backend
"""

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

