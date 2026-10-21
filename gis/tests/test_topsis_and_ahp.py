"""
Tests for AHP Multi-Hazard Overlay and TOPSIS Relocation Algorithms
"""

import numpy as np
import pytest
from gis.hazard_analysis.overlay import compute_ahp_weights, HazardOverlayEngine, DEFAULT_PAIRWISE_MATRIX
from gis.relocation.topsis import RelocationDecisionEngine


def test_ahp_consistency():
    weights, lambda_max, cr, is_consistent = compute_ahp_weights(DEFAULT_PAIRWISE_MATRIX)
    assert np.isclose(np.sum(weights), 1.0)
    assert cr < 0.10, f"AHP matrix inconsistent: CR={cr:.4f}"
    assert is_consistent is True


def test_capacity_estimation():
    engine = RelocationDecisionEngine(default_planning_area_per_person=30.0, default_usable_fraction=0.60)
    # Area: 10,000 sqm * 0.60 = 6,000 sqm / 30 = 200 persons
    capacity, net_area = engine.estimate_capacity(10000.0)
    assert capacity == 200
    assert net_area == 6000.0


def test_hard_constraints_exclusion():
    engine = RelocationDecisionEngine()

    # Case 1: Steep slope should be excluded
    steep_site = {
        "total_area_sqm": 25000.0,
        "terrain_slope_deg": 24.0,  # exceeds 16 deg limit
        "buffer_from_hazard_m": 500.0,
        "in_red_zone": False
    }
    is_eligible, reasons = engine.check_hard_constraints(steep_site)
    assert is_eligible is False
    assert any("slope" in r.lower() for r in reasons)

    # Case 2: Safe site
    safe_site = {
        "total_area_sqm": 25000.0,
        "terrain_slope_deg": 4.5,
        "buffer_from_hazard_m": 600.0,
        "in_red_zone": False
    }
    is_eligible, reasons = engine.check_hard_constraints(safe_site)
    assert is_eligible is True
    assert len(reasons) == 0


def test_topsis_ranking_order():
    engine = RelocationDecisionEngine()
    candidates = [
        {
            "site_id": "SITE-HIGH",
            "properties": {
                "distance_to_road_km": 0.1,       # Lower is better (cost)
                "service_proximity_score": 0.95,   # Higher is better (beneficial)
                "usable_area_ha": 10.0,            # Higher is better
                "terrain_slope_deg": 2.0,          # Lower is better
                "buffer_from_hazard_km": 1.5,      # Higher is better
            }
        },
        {
            "site_id": "SITE-LOW",
            "properties": {
                "distance_to_road_km": 2.5,       # Worse road distance
                "service_proximity_score": 0.20,   # Worse service proximity
                "usable_area_ha": 1.0,            # Smaller
                "terrain_slope_deg": 12.0,        # Steeper
                "buffer_from_hazard_km": 0.3,     # Closer to hazard
            }
        }
    ]

    ranked = engine.run_topsis(candidates)
    assert len(ranked) == 2
    assert ranked[0]["properties"]["site_id"] == "SITE-HIGH"
    assert ranked[0]["properties"]["rank"] == 1
    assert ranked[0]["properties"]["topsis_score"] > ranked[1]["properties"]["topsis_score"]

