"""
Project Orchestration & Decision-Support Service
Coordinates ML Inference, GIS Multi-Hazard Overlay, and TOPSIS Relocation Ranking.
"""

import os
import json
import copy
from typing import Dict, List, Any, Optional
import numpy as np

from gis.hazard_analysis.overlay import HazardOverlayEngine
from gis.relocation.topsis import RelocationDecisionEngine


SAMPLE_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../data/samples")
)


class ProjectDataStore:
    """In-memory reactive store initialized with study dataset"""
    def __init__(self):
        self._load_seed_data()

    def _load_seed_data(self):
        meta_file = os.path.join(SAMPLE_DATA_DIR, "project_metadata.json")
        bld_file = os.path.join(SAMPLE_DATA_DIR, "sample_buildings.geojson")
        haz_file = os.path.join(SAMPLE_DATA_DIR, "sample_hazard_zones.geojson")
        rel_file = os.path.join(SAMPLE_DATA_DIR, "sample_relocation_sites.geojson")

        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                self.project_metadata = json.load(f)
        else:
            self.project_metadata = {"id": "proj-default", "name": "Default Project"}

        if os.path.exists(bld_file):
            with open(bld_file, "r", encoding="utf-8") as f:
                self.buildings_data = json.load(f)["features"]
        else:
            self.buildings_data = []

        if os.path.exists(haz_file):
            with open(haz_file, "r", encoding="utf-8") as f:
                self.hazard_zones_data = json.load(f)["features"]
        else:
            self.hazard_zones_data = []

        if os.path.exists(rel_file):
            with open(rel_file, "r", encoding="utf-8") as f:
                self.candidate_sites_data = json.load(f)["features"]
        else:
            self.candidate_sites_data = []

        # Initialize engines
        self.hazard_engine = HazardOverlayEngine()
        self.relocation_engine = RelocationDecisionEngine()

        # Run initial ranking
        self.screened_relocation_results = self.relocation_engine.screen_and_rank_sites(
            self.candidate_sites_data
        )

    def get_project_metadata(self) -> Dict[str, Any]:
        return self.project_metadata

    def get_summary_statistics(self) -> Dict[str, Any]:
        total_bldgs = len(self.buildings_data)
        counts = {"undamaged": 0, "minor_damage": 0, "major_damage": 0, "destroyed": 0}
        flagged_count = 0
        total_area = 0.0

        for b in self.buildings_data:
            props = b.get("properties", {})
            d_class = props.get("damage_class", "undamaged")
            counts[d_class] = counts.get(d_class, 0) + 1
            if props.get("needs_review") or props.get("review_status") == "FLAGGED_FOR_REVIEW":
                flagged_count += 1
            total_area += props.get("area_sqm", 0.0)

        # Relocation capacity summary
        reloc_meta = self.screened_relocation_results.get("metadata", {})
        total_cap = reloc_meta.get("total_relocation_capacity_persons", 0)
        eligible_sites = reloc_meta.get("eligible_count", 0)

        # Displaced person estimate (approx 4.5 people per destroyed/major building)
        severely_damaged = counts.get("destroyed", 0) + counts.get("major_damage", 0)
        estimated_displaced_persons = severely_damaged * 4

        return {
            "total_buildings_assessed": total_bldgs,
            "damage_breakdown": counts,
            "severely_damaged_count": severely_damaged,
            "damage_rate_percentage": round((severely_damaged / total_bldgs * 100.0) if total_bldgs > 0 else 0, 1),
            "flagged_for_human_review": flagged_count,
            "total_structural_footprint_sqm": round(total_area, 1),
            "estimated_displaced_persons": estimated_displaced_persons,
            "total_relocation_capacity_available": total_cap,
            "capacity_adequacy_ratio": round(total_cap / estimated_displaced_persons, 2) if estimated_displaced_persons > 0 else 999.0,
            "eligible_relocation_sites": eligible_sites,
        }

    def get_filtered_buildings(
        self,
        damage_class: Optional[str] = None,
        min_confidence: Optional[float] = None,
        needs_review: Optional[bool] = None,
    ) -> Dict[str, Any]:
        filtered = []
        for feat in self.buildings_data:
            props = feat.get("properties", {})
            if damage_class and props.get("damage_class") != damage_class:
                continue
            if min_confidence is not None and props.get("confidence", 0.0) < min_confidence:
                continue
            if needs_review is not None and props.get("needs_review") != needs_review:
                continue
            filtered.append(feat)

        return {
            "type": "FeatureCollection",
            "metadata": {
                "total_returned": len(filtered),
                "total_assessed": len(self.buildings_data),
            },
            "features": filtered,
        }

    def update_building_review(
        self, building_id: str, new_damage_class: str, reviewer_notes: Optional[str], reviewer_id: str
    ) -> Optional[Dict[str, Any]]:
        colors = {
            "undamaged": "#10B981",
            "minor_damage": "#FBBF24",
            "major_damage": "#F97316",
            "destroyed": "#EF4444",
        }
        level_map = {"undamaged": 0, "minor_damage": 1, "major_damage": 2, "destroyed": 3}

        for feat in self.buildings_data:
            props = feat.get("properties", {})
            if props.get("building_id") == building_id:
                # Save audit trail
                props["original_damage_class"] = props.get("original_damage_class") or props.get("damage_class")
                props["original_confidence"] = props.get("confidence")
                props["damage_class"] = new_damage_class
                props["damage_level"] = level_map.get(new_damage_class, 0)
                props["damage_color"] = colors.get(new_damage_class, "#6B7280")
                props["review_status"] = "VERIFIED_BY_ANALYST"
                props["needs_review"] = False
                props["reviewer_id"] = reviewer_id
                props["reviewer_notes"] = reviewer_notes
                props["reviewed_at"] = "2026-10-21T16:00:00Z"
                return feat
        return None

    def run_hazard_analysis(
        self,
        weights_dict: Dict[str, float],
        red_threshold: float = 0.65,
        amber_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        # Custom weights normalized
        total_w = sum(weights_dict.values())
        norm_weights = [weights_dict.get(f, 0.2) / total_w for f in self.hazard_engine.factors]
        self.hazard_engine.weights = np.array(norm_weights)
        self.hazard_engine.red_threshold = red_threshold
        self.hazard_engine.amber_threshold = amber_threshold

        result = self.hazard_engine.process_grid_zones(self.hazard_zones_data)
        return result

    def get_hazard_zones(self) -> Dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": self.hazard_zones_data
        }

    def run_relocation_analysis(
        self,
        planning_area_sqm: float = 30.0,
        usable_fraction: float = 0.65,
        max_slope_deg: float = 16.0,
        min_hazard_buffer_m: float = 250.0,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        self.relocation_engine.planning_area_per_person = planning_area_sqm
        self.relocation_engine.usable_fraction = usable_fraction
        self.relocation_engine.max_allowable_slope_deg = max_slope_deg
        self.relocation_engine.min_hazard_buffer_meters = min_hazard_buffer_m

        self.screened_relocation_results = self.relocation_engine.screen_and_rank_sites(
            self.candidate_sites_data, custom_weights=custom_weights
        )
        return self.screened_relocation_results

    def get_relocation_sites(self) -> Dict[str, Any]:
        return self.screened_relocation_results


# Singleton project store instance
store = ProjectDataStore()

