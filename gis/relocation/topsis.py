"""
Relocation Site Screening & Multi-Criteria TOPSIS Decision Engine
1. Enforces hard constraint exclusion filters (hazard proximity, steep slopes, minimal usable area).
2. Calculates preliminary carrying capacity: C_i = (A_i * u_i) / a_i.
3. Performs TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) ranking.
4. Provides full criteria breakdown, explainability metrics, and actionable planning labels.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class RelocationDecisionEngine:
    """
    Multi-criteria decision support for candidate relocation sites following
    disaster displacement.
    """
    def __init__(
        self,
        default_planning_area_per_person: float = 30.0,  # 30 m^2 per person (Sphere standard)
        default_usable_fraction: float = 0.65,          # 65% developable (roads/sanitation take 35%)
        max_allowable_slope_deg: float = 16.0,          # >16 degrees is considered steep/landslide prone
        min_site_area_sqm: float = 1500.0,              # Minimum viable settlement site size
        min_hazard_buffer_meters: float = 250.0,        # Minimum safe distance from active Red Zone
    ):
        self.planning_area_per_person = default_planning_area_per_person
        self.usable_fraction = default_usable_fraction
        self.max_allowable_slope_deg = max_allowable_slope_deg
        self.min_site_area_sqm = min_site_area_sqm
        self.min_hazard_buffer_meters = min_hazard_buffer_meters

        # Default TOPSIS criteria configuration
        # beneficial: True means higher score is better; False means lower (cost) is better.
        self.criteria_config = [
            {"name": "distance_to_road_km", "beneficial": False, "weight": 0.20, "label": "Road Access Distance"},
            {"name": "service_proximity_score", "beneficial": True, "weight": 0.25, "label": "Essential Services Proximity"},
            {"name": "usable_area_ha", "beneficial": True, "weight": 0.25, "label": "Net Usable Area"},
            {"name": "terrain_slope_deg", "beneficial": False, "weight": 0.15, "label": "Terrain Slope"},
            {"name": "buffer_from_hazard_km", "beneficial": True, "weight": 0.15, "label": "Buffer from Hazard Zone"},
        ]

    def estimate_capacity(
        self,
        total_area_sqm: float,
        usable_fraction: Optional[float] = None,
        planning_area_per_person: Optional[float] = None,
    ) -> Tuple[int, float]:
        """
        Calculates capacity C = (A * u) / a
        Returns (estimated_persons_capacity, net_usable_area_sqm)
        """
        u = usable_fraction if usable_fraction is not None else self.usable_fraction
        a = planning_area_per_person if planning_area_per_person is not None else self.planning_area_per_person

        net_area = total_area_sqm * u
        capacity = int(net_area / a) if a > 0 else 0
        return capacity, net_area

    def check_hard_constraints(self, site_props: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Screens candidates against mandatory exclusionary criteria.
        Returns (is_eligible, list_of_exclusion_reasons).
        """
        reasons = []

        # 1. Total area check
        area = site_props.get("total_area_sqm", 0.0)
        if area < self.min_site_area_sqm:
            reasons.append(f"Insufficient total area ({area:.0f} m² < minimum {self.min_site_area_sqm:.0f} m²)")

        # 2. Slope steepness check
        slope = site_props.get("terrain_slope_deg", 0.0)
        if slope > self.max_allowable_slope_deg:
            reasons.append(f"Excessive terrain slope ({slope:.1f}° > limit {self.max_allowable_slope_deg:.1f}°)")

        # 3. Buffer distance to Red Zone check
        hazard_dist = site_props.get("buffer_from_hazard_m", 9999.0)
        if hazard_dist < self.min_hazard_buffer_meters:
            reasons.append(f"Encroaches hazard buffer ({hazard_dist:.0f} m < safe buffer {self.min_hazard_buffer_meters:.0f} m)")

        # 4. Red Zone overlap check
        in_red_zone = site_props.get("in_red_zone", False)
        if in_red_zone:
            reasons.append("Site geometry directly intersects documented Red Hazard Zone")

        # 5. Land status / legal hold check
        legal_status = site_props.get("land_tenure_status", "AVAILABLE")
        if legal_status.upper() in ["DISPUTED", "PROTECTED_FOREST", "RESTRICTED"]:
            reasons.append(f"Land tenure restriction: {legal_status}")

        is_eligible = len(reasons) == 0
        return is_eligible, reasons

    def run_topsis(
        self,
        eligible_candidates: List[Dict[str, Any]],
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Computes TOPSIS rankings for eligible candidate sites.
        Returns sorted list of sites with closeness scores and ranks.
        """
        if not eligible_candidates:
            return []

        # Prepare weights and criteria
        criteria_names = [c["name"] for c in self.criteria_config]
        beneficial_flags = [c["beneficial"] for c in self.criteria_config]

        if custom_weights:
            weights = np.array([custom_weights.get(name, c["weight"]) for c, name in zip(self.criteria_config, criteria_names)])
        else:
            weights = np.array([c["weight"] for c in self.criteria_config])
        # Normalize weights to sum to 1.0
        weights = weights / np.sum(weights)

        # Build raw decision matrix (M x N)
        m = len(eligible_candidates)
        n = len(criteria_names)
        matrix = np.zeros((m, n), dtype=float)

        for i, site in enumerate(eligible_candidates):
            props = site.get("properties", site)
            for j, c_name in enumerate(criteria_names):
                val = float(props.get(c_name, 0.0))
                matrix[i, j] = val

        # 1. Vector Normalization: r_ij = x_ij / sqrt(sum_k x_kj^2)
        norm_matrix = np.zeros_like(matrix)
        for j in range(n):
            col = matrix[:, j]
            norm = np.linalg.norm(col)
            if norm == 0:
                norm_matrix[:, j] = 0.0
            else:
                norm_matrix[:, j] = col / norm

        # 2. Weighted Normalized Decision Matrix: v_ij = w_j * r_ij
        v_matrix = norm_matrix * weights

        # 3. Determine Ideal Best (A+) and Ideal Worst (A-)
        ideal_best = np.zeros(n)
        ideal_worst = np.zeros(n)

        for j in range(n):
            col = v_matrix[:, j]
            if beneficial_flags[j]:
                ideal_best[j] = np.max(col)
                ideal_worst[j] = np.min(col)
            else:
                ideal_best[j] = np.min(col)
                ideal_worst[j] = np.max(col)

        # 4. Calculate Euclidean Distances S+ and S-
        s_plus = np.sqrt(np.sum((v_matrix - ideal_best) ** 2, axis=1))
        s_minus = np.sqrt(np.sum((v_matrix - ideal_worst) ** 2, axis=1))

        # 5. Calculate Relative Closeness to Ideal Solution: C_i = S- / (S+ + S-)
        total_dist = s_plus + s_minus
        closeness = np.zeros(m)
        for i in range(m):
            if total_dist[i] > 0:
                closeness[i] = s_minus[i] / total_dist[i]
            else:
                closeness[i] = 0.5

        # 6. Attach scores and determine ranking
        ranked_sites = []
        for i, site in enumerate(eligible_candidates):
            props = dict(site.get("properties", site))
            if "site_id" in site and "site_id" not in props:
                props["site_id"] = site["site_id"]
            score = float(closeness[i])

            # Planning recommendation label
            if score >= 0.70:
                category = "Immediate Priority (Phase 1)"
            elif score >= 0.45:
                category = "Short-Term Priority (Phase 2)"
            else:
                category = "Medium-Term Contingency (Phase 3)"

            criteria_details = {}
            for j, c in enumerate(self.criteria_config):
                criteria_details[c["name"]] = {
                    "label": c["label"],
                    "raw_value": float(matrix[i, j]),
                    "beneficial": c["beneficial"],
                    "weight": round(float(weights[j]), 3),
                }

            props["topsis_score"] = round(score, 4)
            props["recommendation_category"] = category
            props["separation_ideal_best"] = round(float(s_plus[i]), 4)
            props["separation_ideal_worst"] = round(float(s_minus[i]), 4)
            props["criteria_breakdown"] = criteria_details

            site_entry = {
                **site,
                "properties": props
            }
            ranked_sites.append(site_entry)

        # Sort descending by TOPSIS score
        ranked_sites.sort(key=lambda x: x["properties"]["topsis_score"], reverse=True)

        # Assign ordinal rank
        for rank_idx, site in enumerate(ranked_sites, 1):
            site["properties"]["rank"] = rank_idx

        return ranked_sites

    def screen_and_rank_sites(
        self,
        candidate_features: List[Dict[str, Any]],
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Complete pipeline: Screens raw candidates, excludes violations,
        computes carrying capacities, and performs TOPSIS ranking on eligible sites.
        """
        eligible = []
        excluded = []

        for feat in candidate_features:
            props = feat.get("properties", {})
            total_area = props.get("total_area_sqm", 0.0)

            # Estimate capacity
            cap, net_area = self.estimate_capacity(total_area)
            props["estimated_capacity_persons"] = cap
            props["net_usable_area_sqm"] = net_area
            props["usable_area_ha"] = round(net_area / 10000.0, 2)

            # Derive km units for standard TOPSIS input if provided in meters
            if "distance_to_road_m" in props and "distance_to_road_km" not in props:
                props["distance_to_road_km"] = round(props["distance_to_road_m"] / 1000.0, 2)
            if "buffer_from_hazard_m" in props and "buffer_from_hazard_km" not in props:
                props["buffer_from_hazard_km"] = round(props["buffer_from_hazard_m"] / 1000.0, 2)

            is_eligible, reasons = self.check_hard_constraints(props)
            props["is_eligible"] = is_eligible
            props["exclusion_reasons"] = reasons

            feat_updated = {**feat, "properties": props}

            if is_eligible:
                eligible.append(feat_updated)
            else:
                props["rank"] = None
                props["topsis_score"] = 0.0
                props["recommendation_category"] = "EXCLUDED"
                excluded.append(feat_updated)

        ranked_eligible = self.run_topsis(eligible, custom_weights)

        total_capacity = sum(s["properties"]["estimated_capacity_persons"] for s in ranked_eligible)

        return {
            "type": "FeatureCollection",
            "metadata": {
                "total_candidate_sites": len(candidate_features),
                "eligible_count": len(ranked_eligible),
                "excluded_count": len(excluded),
                "total_relocation_capacity_persons": total_capacity,
                "planning_assumptions": {
                    "planning_area_per_person_sqm": self.planning_area_per_person,
                    "usable_area_fraction": self.usable_fraction,
                    "max_slope_deg": self.max_allowable_slope_deg,
                    "min_hazard_buffer_m": self.min_hazard_buffer_meters,
                }
            },
            "features": ranked_eligible + excluded,
        }
