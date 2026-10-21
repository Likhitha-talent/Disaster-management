"""
FastAPI Routes for Disaster Damage Assessment & Relocation Platform
Endpoints align with Section 8 of the Project Implementation Specification.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Optional, Dict, Any
import csv
import io

from backend.app.schemas.schemas import (
    ProjectCreate,
    BuildingReviewUpdate,
    AHPWeightConfig,
    RelocationParamConfig,
    DamageAssessmentRequest,
)
from backend.app.services.project_service import store

router = APIRouter(prefix="/api")


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI-Powered Post-Disaster Damage Assessment & Relocation API",
        "academic_year": "2026-27",
        "institution": "The National Institute of Engineering, Mysuru",
    }


@router.get("/projects")
def list_projects():
    meta = store.get_project_metadata()
    summary = store.get_summary_statistics()
    return [{**meta, "summary": summary}]


@router.get("/projects/{project_id}")
def get_project_details(project_id: str):
    meta = store.get_project_metadata()
    summary = store.get_summary_statistics()
    return {
        "project": meta,
        "summary": summary
    }


@router.post("/projects/{project_id}/damage-assessment")
def run_damage_assessment(project_id: str, payload: DamageAssessmentRequest):
    summary = store.get_summary_statistics()
    return {
        "job_id": f"job-infer-{project_id}-992",
        "status": "COMPLETED",
        "model_version": payload.model_version,
        "confidence_threshold": payload.confidence_threshold,
        "summary": summary
    }


@router.get("/projects/{project_id}/buildings")
def get_buildings(
    project_id: str,
    damage_class: Optional[str] = Query(None, description="undamaged, minor_damage, major_damage, destroyed"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    needs_review: Optional[bool] = Query(None),
):
    return store.get_filtered_buildings(
        damage_class=damage_class,
        min_confidence=min_confidence,
        needs_review=needs_review,
    )


@router.post("/projects/{project_id}/buildings/{building_id}/review")
def update_building_review(project_id: str, building_id: str, payload: BuildingReviewUpdate):
    updated = store.update_building_review(
        building_id=building_id,
        new_damage_class=payload.verified_damage_class,
        reviewer_notes=payload.reviewer_notes,
        reviewer_id=payload.reviewer_id,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Building {building_id} not found")
    return {"status": "SUCCESS", "building": updated}


@router.post("/projects/{project_id}/risk-analysis")
def run_risk_analysis(project_id: str, config: AHPWeightConfig):
    weights_dict = {
        "damage_density": config.damage_density,
        "flood_hydro_exposure": config.flood_hydro_exposure,
        "slope_landslide": config.slope_landslide,
        "land_cover_hazard": config.land_cover_hazard,
        "population_density": config.population_density,
    }
    result = store.run_hazard_analysis(
        weights_dict=weights_dict,
        red_threshold=config.red_threshold,
        amber_threshold=config.amber_threshold,
    )
    return result


@router.get("/projects/{project_id}/risk-zones")
def get_risk_zones(project_id: str):
    return store.get_hazard_zones()


@router.post("/projects/{project_id}/relocation-analysis")
def run_relocation_analysis(project_id: str, config: RelocationParamConfig):
    result = store.run_relocation_analysis(
        planning_area_sqm=config.planning_area_per_person_sqm,
        usable_fraction=config.usable_area_fraction,
        max_slope_deg=config.max_allowable_slope_deg,
        min_hazard_buffer_m=config.min_hazard_buffer_meters,
        custom_weights=config.criteria_weights,
    )
    return result


@router.get("/projects/{project_id}/relocation-sites")
def get_relocation_sites(project_id: str):
    return store.get_relocation_sites()


@router.get("/projects/{project_id}/reports/{report_format}")
def export_project_report(project_id: str, report_format: str = "json"):
    meta = store.get_project_metadata()
    summary = store.get_summary_statistics()
    sites = store.get_relocation_sites().get("features", [])

    if report_format.lower() == "csv":
        # Generate CSV stream of relocation rankings
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Rank", "Site ID", "Site Name", "Eligibility", "TOPSIS Score",
            "Recommendation", "Capacity (Persons)", "Net Usable Area (m2)",
            "Terrain Slope (deg)", "Road Distance (km)", "Hazard Buffer (m)", "Exclusion Reasons"
        ])
        for s in sites:
            p = s.get("properties", {})
            writer.writerow([
                p.get("rank", "N/A"),
                p.get("site_id"),
                p.get("site_name"),
                "ELIGIBLE" if p.get("is_eligible") else "EXCLUDED",
                p.get("topsis_score", 0.0),
                p.get("recommendation_category"),
                p.get("estimated_capacity_persons", 0),
                p.get("net_usable_area_sqm", 0.0),
                p.get("terrain_slope_deg", 0.0),
                p.get("distance_to_road_km", 0.0),
                p.get("buffer_from_hazard_m", 0.0),
                "; ".join(p.get("exclusion_reasons", []))
            ])
        return PlainTextResponse(output.getvalue(), media_type="text/csv", headers={
            "Content-Disposition": f"attachment; filename=relocation_rankings_{project_id}.csv"
        })

    # Default JSON comprehensive summary
    return {
        "project": meta,
        "executive_summary": summary,
        "relocation_shortlist": [
            {
                "rank": s["properties"].get("rank"),
                "site_id": s["properties"].get("site_id"),
                "site_name": s["properties"].get("site_name"),
                "is_eligible": s["properties"].get("is_eligible"),
                "topsis_score": s["properties"].get("topsis_score"),
                "capacity_persons": s["properties"].get("estimated_capacity_persons"),
                "category": s["properties"].get("recommendation_category"),
                "exclusion_reasons": s["properties"].get("exclusion_reasons"),
            }
            for s in sites
        ]
    }

