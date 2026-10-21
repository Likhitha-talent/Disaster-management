"""
Multi-Hazard Composite Risk Mapping & Red-Zone Screening Module
Implements Analytic Hierarchy Process (AHP) weighted overlay:
  R(x) = sum(w_i * r_i(x))
Generates Red / Amber / Green screening zones with full mathematical transparency,
consistency checks (Consistency Ratio CR < 0.10), and GeoJSON outputs.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd


# Random Index (RI) table for AHP consistency calculation up to n=10
RI_TABLE = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}

DEFAULT_AHP_FACTORS = [
    "damage_density",      # Weight 0.35: Observed structural damage density
    "flood_hydro_exposure", # Weight 0.25: Hydrological flow accumulation / flood exposure
    "slope_landslide",     # Weight 0.18: Terrain slope & landslide susceptibility
    "land_cover_hazard",   # Weight 0.12: Environmental vulnerability / surface permeability
    "population_density",  # Weight 0.10: WorldPop gridded human exposure
]

# Standard pairwise comparison matrix for post-disaster flood/landslide scenario
# Factors: [damage_density, flood_hydro_exposure, slope_landslide, land_cover_hazard, population_density]
DEFAULT_PAIRWISE_MATRIX = np.array([
    [1.00, 1.50, 2.00, 3.00, 3.50],  # damage_density
    [0.67, 1.00, 1.50, 2.00, 2.50],  # flood_hydro_exposure
    [0.50, 0.67, 1.00, 1.50, 2.00],  # slope_landslide
    [0.33, 0.50, 0.67, 1.00, 1.20],  # land_cover_hazard
    [0.29, 0.40, 0.50, 0.83, 1.00],  # population_density
])


def compute_ahp_weights(pairwise_matrix: np.ndarray) -> Tuple[np.ndarray, float, float, bool]:
    """
    Computes priority weights, principal eigenvalue lambda_max, Consistency Index (CI),
    and Consistency Ratio (CR) from an AHP pairwise comparison matrix.

    Returns:
      weights (np.ndarray): Normalized priority weights summing to 1.0
      lambda_max (float): Maximum eigenvalue
      cr (float): Consistency Ratio
      is_consistent (bool): True if CR < 0.10
    """
    n = pairwise_matrix.shape[0]
    # Column normalization method
    col_sums = pairwise_matrix.sum(axis=0)
    norm_matrix = pairwise_matrix / col_sums
    weights = norm_matrix.mean(axis=1)

    # Calculate lambda_max
    weighted_sum_vector = np.dot(pairwise_matrix, weights)
    lambda_max = float(np.mean(weighted_sum_vector / weights))

    # Consistency Index & Ratio
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RI_TABLE.get(n, 1.49)
    cr = float(ci / ri) if ri > 0 else 0.0

    is_consistent = cr < 0.10
    return weights, lambda_max, cr, is_consistent


class HazardOverlayEngine:
    """
    Calculates multi-hazard screening zones using weighted factor overlay.
    Classifies zones into:
      - Red Zone: Composite Score >= red_threshold (Default: 0.65) -> High Risk
      - Amber Zone: amber_threshold <= Score < red_threshold (Default: 0.35 to 0.65) -> Moderate Risk
      - Green Zone: Score < amber_threshold -> Low Risk / Stable
    """
    def __init__(
        self,
        factors: Optional[List[str]] = None,
        pairwise_matrix: Optional[np.ndarray] = None,
        red_threshold: float = 0.65,
        amber_threshold: float = 0.35,
    ):
        self.factors = factors or DEFAULT_AHP_FACTORS
        self.pairwise_matrix = pairwise_matrix if pairwise_matrix is not None else DEFAULT_PAIRWISE_MATRIX
        self.red_threshold = red_threshold
        self.amber_threshold = amber_threshold

        self.weights, self.lambda_max, self.cr, self.is_consistent = compute_ahp_weights(self.pairwise_matrix)

    def evaluate_cell_score(self, factor_scores: Dict[str, float]) -> float:
        """
        Calculates composite score R(x) = sum(w_i * r_i(x)) for a spatial unit.
        All r_i(x) must be normalized to [0, 1].
        """
        score = 0.0
        for i, factor in enumerate(self.factors):
            val = factor_scores.get(factor, 0.0)
            clamped_val = max(0.0, min(1.0, float(val)))
            score += self.weights[i] * clamped_val
        return float(score)

    def classify_score(self, composite_score: float) -> Tuple[str, str]:
        """
        Returns (zone_class, description) based on thresholds.
        """
        if composite_score >= self.red_threshold:
            return "RED", "High Hazard / Permanent Risk Screening Area"
        elif composite_score >= self.amber_threshold:
            return "AMBER", "Moderate Hazard / Cautionary Buffer Zone"
        else:
            return "GREEN", "Low Hazard / Suitable Baseline Zone"

    def process_grid_zones(self, zone_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes a list of spatial polygons with factor ratings, evaluates composite scores,
        and produces a GeoJSON FeatureCollection with risk categories and AHP audit metadata.
        """
        processed_features = []
        red_count = 0
        amber_count = 0
        green_count = 0

        for feat in zone_features:
            props = feat.get("properties", {})
            factor_values = {
                factor: props.get(factor, 0.0) for factor in self.factors
            }
            score = self.evaluate_cell_score(factor_values)
            zone_class, desc = self.classify_score(score)

            if zone_class == "RED":
                red_count += 1
            elif zone_class == "AMBER":
                amber_count += 1
            else:
                green_count += 1

            new_props = {
                **props,
                "composite_risk_score": round(score, 4),
                "zone_class": zone_class,
                "zone_description": desc,
                "factor_breakdown": {k: round(v, 4) for k, v in factor_values.items()},
                "weights_used": {self.factors[i]: round(float(self.weights[i]), 4) for i in range(len(self.factors))},
            }

            processed_features.append({
                "type": "Feature",
                "geometry": feat.get("geometry"),
                "properties": new_props,
            })

        return {
            "type": "FeatureCollection",
            "metadata": {
                "ahp_factors": self.factors,
                "weights": [round(float(w), 4) for w in self.weights],
                "consistency_ratio": round(self.cr, 4),
                "is_consistent": self.is_consistent,
                "thresholds": {
                    "red": self.red_threshold,
                    "amber": self.amber_threshold,
                },
                "summary": {
                    "total_zones": len(processed_features),
                    "red_zones": red_count,
                    "amber_zones": amber_count,
                    "green_zones": green_count,
                }
            },
            "features": processed_features,
        }

