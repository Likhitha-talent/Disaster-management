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
from gis.hazard_analysis.weather_service import WeatherHazardService
from gis.relocation.topsis import RelocationDecisionEngine


OUTPUTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../data/outputs")
)
REVIEW_STORE_FILE = os.path.join(OUTPUTS_DIR, "reviewed_buildings.json")

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

        # Load persisted human building reviews if present
        if os.path.exists(REVIEW_STORE_FILE):
            try:
                with open(REVIEW_STORE_FILE, "r", encoding="utf-8") as f:
                    saved_reviews = json.load(f)
                if isinstance(saved_reviews, dict):
                    colors = {
                        "undamaged": "#10B981",
                        "minor_damage": "#FBBF24",
                        "major_damage": "#F97316",
                        "destroyed": "#EF4444",
                    }
                    level_map = {"undamaged": 0, "minor_damage": 1, "major_damage": 2, "destroyed": 3}
                    for feat in self.buildings_data:
                        b_id = feat.get("properties", {}).get("building_id")
                        if b_id in saved_reviews:
                            rev = saved_reviews[b_id]
                            props = feat["properties"]
                            props["original_damage_class"] = rev.get("original_damage_class") or props.get("damage_class")
                            props["original_confidence"] = rev.get("original_confidence", props.get("confidence"))
                            props["damage_class"] = rev["verified_damage_class"]
                            props["damage_level"] = level_map.get(rev["verified_damage_class"], 0)
                            props["damage_color"] = colors.get(rev["verified_damage_class"], "#6B7280")
                            props["review_status"] = "VERIFIED_BY_ANALYST"
                            props["needs_review"] = False
                            props["reviewer_id"] = rev.get("reviewer_id", "OFFICER-SDMA-01")
                            props["reviewer_notes"] = rev.get("reviewer_notes")
                            props["reviewed_at"] = rev.get("reviewed_at", "2026-10-21T16:00:00Z")
            except Exception:
                # Fall back gracefully to in-memory seed data on corrupt/unreadable file
                pass

        # Initialize engines
        self.hazard_engine = HazardOverlayEngine()
        self.relocation_engine = RelocationDecisionEngine()
        self.weather_service = WeatherHazardService()

        # Run initial ranking
        self.screened_relocation_results = self.relocation_engine.screen_and_rank_sites(
            self.candidate_sites_data
        )

    def _persist_building_review(self, building_id: str, review_record: Dict[str, Any]):
        """Persist human review override to disk safely with graceful fallback"""
        try:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            saved_reviews = {}
            if os.path.exists(REVIEW_STORE_FILE):
                try:
                    with open(REVIEW_STORE_FILE, "r", encoding="utf-8") as f:
                        saved_reviews = json.load(f)
                    if not isinstance(saved_reviews, dict):
                        saved_reviews = {}
                except Exception:
                    saved_reviews = {}
            saved_reviews[building_id] = review_record
            with open(REVIEW_STORE_FILE, "w", encoding="utf-8") as f:
                json.dump(saved_reviews, f, indent=2)
        except Exception:
            # File system write failure falls back gracefully to in-memory state
            pass

    def get_live_weather(self) -> Dict[str, Any]:
        return self.weather_service.fetch_live_weather()

    def get_hazard_zones(self, use_live_weather: bool = False) -> Dict[str, Any]:
        if not use_live_weather:
            result = self.hazard_engine.process_grid_zones(self.hazard_zones_data)
            result["metadata"]["mode"] = "SATELLITE_BASELINE"
            result["metadata"]["description"] = "Topographical and structural baseline hazard screening"
            return result

        # Dynamic live weather integration mode
        weather = self.get_live_weather()
        hydro_factor = weather.get("dynamic_hydro_risk_factor", 0.5)

        # Scale flood exposure across zones according to live precipitation alert
        dynamic_zones = copy.deepcopy(self.hazard_zones_data)
        for zone in dynamic_zones:
            props = zone.get("properties", {})
            orig_hydro = props.get("flood_hydro_exposure", 0.5)
            # Blend baseline terrain flow with current live rainfall trigger
            blended_hydro = min(1.0, max(0.0, 0.4 * orig_hydro + 0.6 * hydro_factor))
            props["flood_hydro_exposure"] = round(blended_hydro, 3)
            props["live_weather_sync"] = True
            props["live_rainfall_24h_mm"] = weather.get("cumulative_24h_rainfall_mm")
            props["live_alert_level"] = weather.get("alert_level")

        # Process with AHP engine
        result = self.hazard_engine.process_grid_zones(dynamic_zones)
        result["metadata"]["mode"] = "LIVE_WEATHER_DRIVEN"
        result["metadata"]["weather"] = weather
        return result

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
                orig_class = props.get("original_damage_class") or props.get("damage_class")
                orig_conf = props.get("confidence")
                props["original_damage_class"] = orig_class
                props["original_confidence"] = orig_conf
                props["damage_class"] = new_damage_class
                props["damage_level"] = level_map.get(new_damage_class, 0)
                props["damage_color"] = colors.get(new_damage_class, "#6B7280")
                props["review_status"] = "VERIFIED_BY_ANALYST"
                props["needs_review"] = False
                props["reviewer_id"] = reviewer_id
                props["reviewer_notes"] = reviewer_notes
                props["reviewed_at"] = "2026-10-21T16:00:00Z"

                # Persist to disk safely with graceful in-memory fallback
                self._persist_building_review(building_id, {
                    "original_damage_class": orig_class,
                    "original_confidence": orig_conf,
                    "verified_damage_class": new_damage_class,
                    "reviewer_id": reviewer_id,
                    "reviewer_notes": reviewer_notes,
                    "reviewed_at": "2026-10-21T16:00:00Z"
                })

                return feat
        return None

    def run_hazard_analysis(
        self,
        weights_dict: Dict[str, float],
        red_threshold: float = 0.65,
        amber_threshold: float = 0.35,
        use_live_weather: bool = False,
    ) -> Dict[str, Any]:
        # Custom weights normalized
        total_w = sum(weights_dict.values())
        if total_w <= 0:
            total_w = 1.0
        norm_weights = np.array([weights_dict.get(f, 0.2) / total_w for f in self.hazard_engine.factors], dtype=float)
        self.hazard_engine.update_weights_and_cr(norm_weights)
        self.hazard_engine.red_threshold = red_threshold
        self.hazard_engine.amber_threshold = amber_threshold

        return self.get_hazard_zones(use_live_weather=use_live_weather)

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
