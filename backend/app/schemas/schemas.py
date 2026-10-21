"""
Pydantic Request & Response Schemas for Disaster Decision Support Platform
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str
    disaster_type: str
    event_date: str
    center_lat: float = 12.3375
    center_lon: float = 75.8062
    notes: Optional[str] = None


class BuildingReviewUpdate(BaseModel):
    verified_damage_class: str
    reviewer_notes: Optional[str] = None
    reviewer_id: str = "OFFICER-SDMA-01"


class AHPWeightConfig(BaseModel):
    damage_density: float = 0.35
    flood_hydro_exposure: float = 0.25
    slope_landslide: float = 0.18
    land_cover_hazard: float = 0.12
    population_density: float = 0.10
    red_threshold: float = 0.65
    amber_threshold: float = 0.35


class RelocationParamConfig(BaseModel):
    planning_area_per_person_sqm: float = 30.0
    usable_area_fraction: float = 0.65
    max_allowable_slope_deg: float = 16.0
    min_hazard_buffer_meters: float = 250.0
    criteria_weights: Optional[Dict[str, float]] = None


class DamageAssessmentRequest(BaseModel):
    confidence_threshold: float = 0.65
    model_version: str = "Siamese-UNet-v1.0.0"

