"""
Sample Data Generator for Post-Disaster Damage Assessment Prototype
Generates georeferenced sample datasets for the study area:
- Pre- and Post-disaster building footprints with 4-level damage assessment and review flags
- Multi-hazard Red / Amber / Green screening zones with AHP factor layers
- Candidate relocation sites with capacity calculations and TOPSIS ranking attributes
"""

import json
import math
import os
from typing import Dict, List, Any


def generate_sample_study_dataset(base_dir: str):
    # Anchor coordinates: Western Ghats / Kodagu - Wayanad disaster screening sector
    CENTER_LAT = 12.3375
    CENTER_LON = 75.8062

    deg_lat = 1.0 / 111320.0
    deg_lon = 1.0 / (111320.0 * math.cos(math.radians(CENTER_LAT)))

    # 1. Project Metadata
    project_metadata = {
        "id": "proj-karnataka-wghats-2026",
        "name": "Kodagu-Wayanad Sector Monsoon Landslide & Flood Assessment",
        "disaster_type": "Monsoon Landslide & Flash Flood",
        "event_date": "2026-08-14",
        "imagery_sources": [
            {"source": "Sentinel-2 & High-Res PlanetScope", "resolution_m": 0.5, "crs": "EPSG:4326"},
            {"source": "Copernicus DEM GLO-30", "resolution_m": 30.0, "crs": "EPSG:4326"}
        ],
        "center": [CENTER_LAT, CENTER_LON],
        "zoom": 15,
        "bounding_box": [
            [CENTER_LAT - 0.015, CENTER_LON - 0.020],
            [CENTER_LAT + 0.015, CENTER_LON + 0.020]
        ]
    }

    # 2. Buildings (50 buildings scattered across flood plain, hill slopes, and stable plateaus)
    buildings = []
    damage_distribution = [
        # (count, class_name, level, conf_range, color)
        (15, "undamaged", 0, (0.84, 0.98), "#10B981"),
        (12, "minor_damage", 1, (0.72, 0.92), "#FBBF24"),
        (13, "major_damage", 2, (0.68, 0.89), "#F97316"),
        (10, "destroyed", 3, (0.79, 0.96), "#EF4444"),
    ]

    bldg_id = 1
    # Distribute buildings with realistic spatial clustering
    for count, d_class, d_level, conf_range, color in damage_distribution:
        for idx in range(count):
            # Cluster destroyed/major near river/slope (southern/eastern sector)
            if d_class in ["destroyed", "major_damage"]:
                dx = -400 + (bldg_id * 53) % 800
                dy = -600 + (bldg_id * 67) % 700
            elif d_class == "minor_damage":
                dx = -600 + (bldg_id * 47) % 1100
                dy = -200 + (bldg_id * 59) % 800
            else:  # undamaged on elevated ridge
                dx = 200 + (bldg_id * 61) % 800
                dy = 200 + (bldg_id * 71) % 700

            # Building footprint geometry (approx 12m x 15m rectangular polygon)
            b_width = 12.0 + (bldg_id % 7) * 2.0
            b_height = 10.0 + (bldg_id % 5) * 3.0

            x1 = dx - b_width / 2
            x2 = dx + b_width / 2
            y1 = dy - b_height / 2
            y2 = dy + b_height / 2

            poly = [
                [round(CENTER_LON + x1 * deg_lon, 7), round(CENTER_LAT + y1 * deg_lat, 7)],
                [round(CENTER_LON + x2 * deg_lon, 7), round(CENTER_LAT + y1 * deg_lat, 7)],
                [round(CENTER_LON + x2 * deg_lon, 7), round(CENTER_LAT + y2 * deg_lat, 7)],
                [round(CENTER_LON + x1 * deg_lon, 7), round(CENTER_LAT + y2 * deg_lat, 7)],
                [round(CENTER_LON + x1 * deg_lon, 7), round(CENTER_LAT + y1 * deg_lat, 7)],
            ]

            # Confidence with 4 low-confidence cases flagged for human-in-the-loop review
            if bldg_id in [7, 19, 32, 44]:
                conf = round(0.52 + (bldg_id % 10) * 0.01, 3)
                needs_rev = True
                rev_status = "FLAGGED_FOR_REVIEW"
            else:
                conf = round(conf_range[0] + (bldg_id % 15) * ((conf_range[1] - conf_range[0]) / 15.0), 3)
                needs_rev = False
                rev_status = "AUTO_VERIFIED"

            # Probabilities distribution
            probs = [0.05, 0.05, 0.05, 0.05]
            probs[d_level] = conf
            rem = (1.0 - conf) / 3.0
            for k in range(4):
                if k != d_level:
                    probs[k] = round(rem, 3)

            buildings.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [poly]
                },
                "properties": {
                    "building_id": f"BLD-{bldg_id:04d}",
                    "damage_class": d_class,
                    "damage_level": d_level,
                    "damage_color": color,
                    "confidence": conf,
                    "review_status": rev_status,
                    "needs_review": needs_rev,
                    "area_sqm": round(b_width * b_height, 1),
                    "elevation_m": round(840.0 + (dy / 20.0), 1),
                    "slope_deg": round(4.5 + (dy < -200) * 12.0 + (dx % 7), 1),
                    "class_probabilities": {
                        "undamaged": probs[0],
                        "minor_damage": probs[1],
                        "major_damage": probs[2],
                        "destroyed": probs[3],
                    },
                    "model_version": "Siamese-UNet-v1.0.0",
                }
            })
            bldg_id += 1

    # 3. Hazard Screening Zones (Red, Amber, Green Polygons with AHP composite risk)
    # Zone 1: Active Valley & Runout Zone (RED ZONE)
    red_poly = [
        [round(CENTER_LON - 900 * deg_lon, 7), round(CENTER_LAT - 800 * deg_lat, 7)],
        [round(CENTER_LON + 200 * deg_lon, 7), round(CENTER_LAT - 800 * deg_lat, 7)],
        [round(CENTER_LON + 150 * deg_lon, 7), round(CENTER_LAT - 200 * deg_lat, 7)],
        [round(CENTER_LON - 750 * deg_lon, 7), round(CENTER_LAT - 150 * deg_lat, 7)],
        [round(CENTER_LON - 900 * deg_lon, 7), round(CENTER_LAT - 800 * deg_lat, 7)],
    ]

    # Zone 2: Moderate Slope Cautionary Buffer (AMBER ZONE)
    amber_poly = [
        [round(CENTER_LON - 950 * deg_lon, 7), round(CENTER_LAT - 150 * deg_lat, 7)],
        [round(CENTER_LON + 300 * deg_lon, 7), round(CENTER_LAT - 200 * deg_lat, 7)],
        [round(CENTER_LON + 450 * deg_lon, 7), round(CENTER_LAT + 250 * deg_lat, 7)],
        [round(CENTER_LON - 850 * deg_lon, 7), round(CENTER_LAT + 200 * deg_lat, 7)],
        [round(CENTER_LON - 950 * deg_lon, 7), round(CENTER_LAT - 150 * deg_lat, 7)],
    ]

    # Zone 3: High Plateau Baseline (GREEN ZONE)
    green_poly = [
        [round(CENTER_LON - 800 * deg_lon, 7), round(CENTER_LAT + 250 * deg_lat, 7)],
        [round(CENTER_LON + 850 * deg_lon, 7), round(CENTER_LAT + 250 * deg_lat, 7)],
        [round(CENTER_LON + 950 * deg_lon, 7), round(CENTER_LAT + 900 * deg_lat, 7)],
        [round(CENTER_LON - 700 * deg_lon, 7), round(CENTER_LAT + 950 * deg_lat, 7)],
        [round(CENTER_LON - 800 * deg_lon, 7), round(CENTER_LAT + 250 * deg_lat, 7)],
    ]

    hazard_zones = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [red_poly]},
            "properties": {
                "zone_id": "HZ-ZONE-01-RED",
                "zone_name": "Perennial River Runout & Inundation Sector",
                "composite_risk_score": 0.842,
                "zone_class": "RED",
                "zone_color": "#DC2626",
                "zone_description": "High Hazard Zone - Permanent structural risk screening recommendation",
                "damage_density": 0.88,
                "flood_hydro_exposure": 0.95,
                "slope_landslide": 0.78,
                "land_cover_hazard": 0.70,
                "population_density": 0.65,
                "area_hectares": 34.2,
                "buildings_inside": 23,
            }
        },
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [amber_poly]},
            "properties": {
                "zone_id": "HZ-ZONE-02-AMBER",
                "zone_name": "Mid-Slope Buffer & Hydrological Runoff Corridor",
                "composite_risk_score": 0.528,
                "zone_class": "AMBER",
                "zone_color": "#D97706",
                "zone_description": "Moderate Hazard Zone - Conditional development / monitored mitigation area",
                "damage_density": 0.45,
                "flood_hydro_exposure": 0.58,
                "slope_landslide": 0.62,
                "land_cover_hazard": 0.48,
                "population_density": 0.42,
                "area_hectares": 41.8,
                "buildings_inside": 12,
            }
        },
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [green_poly]},
            "properties": {
                "zone_id": "HZ-ZONE-03-GREEN",
                "zone_name": "Upland Ridge & Stable Bedrock Plateau",
                "composite_risk_score": 0.184,
                "zone_class": "GREEN",
                "zone_color": "#059669",
                "zone_description": "Low Hazard Baseline - Geotechnically stable zone suitable for habitations",
                "damage_density": 0.08,
                "flood_hydro_exposure": 0.12,
                "slope_landslide": 0.22,
                "land_cover_hazard": 0.25,
                "population_density": 0.30,
                "area_hectares": 58.5,
                "buildings_inside": 15,
            }
        }
    ]

    # 4. Candidate Relocation Sites (6 realistic sites with varying capacities, access, and constraints)
    relocation_candidates = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [round(CENTER_LON + 250 * deg_lon, 7), round(CENTER_LAT + 400 * deg_lat, 7)],
                    [round(CENTER_LON + 600 * deg_lon, 7), round(CENTER_LAT + 400 * deg_lat, 7)],
                    [round(CENTER_LON + 600 * deg_lon, 7), round(CENTER_LAT + 700 * deg_lat, 7)],
                    [round(CENTER_LON + 250 * deg_lon, 7), round(CENTER_LAT + 700 * deg_lat, 7)],
                    [round(CENTER_LON + 250 * deg_lon, 7), round(CENTER_LAT + 400 * deg_lat, 7)],
                ]]
            },
            "properties": {
                "site_id": "SITE-A1-GOVT-PLATEAU",
                "site_name": "Government Revenue Plateau Land",
                "total_area_sqm": 85000.0,
                "usable_fraction": 0.70,
                "terrain_slope_deg": 3.8,
                "distance_to_road_km": 0.25,
                "service_proximity_score": 0.88,  # Near district hospital, PHC & water grid
                "buffer_from_hazard_m": 720.0,
                "in_red_zone": False,
                "land_tenure_status": "AVAILABLE",
                "water_availability": "Verified Borewell + Municipal Line",
                "power_grid_access": "Substation within 1.2 km",
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [round(CENTER_LON - 600 * deg_lon, 7), round(CENTER_LAT + 500 * deg_lat, 7)],
                    [round(CENTER_LON - 250 * deg_lon, 7), round(CENTER_LAT + 500 * deg_lat, 7)],
                    [round(CENTER_LON - 250 * deg_lon, 7), round(CENTER_LAT + 850 * deg_lat, 7)],
                    [round(CENTER_LON - 600 * deg_lon, 7), round(CENTER_LAT + 850 * deg_lat, 7)],
                    [round(CENTER_LON - 600 * deg_lon, 7), round(CENTER_LAT + 500 * deg_lat, 7)],
                ]]
            },
            "properties": {
                "site_id": "SITE-B2-MUNICIPAL-RESERVE",
                "site_name": "Municipal Institutional Reserve Field",
                "total_area_sqm": 62000.0,
                "usable_fraction": 0.65,
                "terrain_slope_deg": 5.2,
                "distance_to_road_km": 0.40,
                "service_proximity_score": 0.75,
                "buffer_from_hazard_m": 580.0,
                "in_red_zone": False,
                "land_tenure_status": "AVAILABLE",
                "water_availability": "River Lift Scheme",
                "power_grid_access": "Distribution transformer within 600 m",
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [round(CENTER_LON + 650 * deg_lon, 7), round(CENTER_LAT + 300 * deg_lat, 7)],
                    [round(CENTER_LON + 900 * deg_lon, 7), round(CENTER_LAT + 300 * deg_lat, 7)],
                    [round(CENTER_LON + 900 * deg_lon, 7), round(CENTER_LAT + 550 * deg_lat, 7)],
                    [round(CENTER_LON + 650 * deg_lon, 7), round(CENTER_LAT + 550 * deg_lat, 7)],
                    [round(CENTER_LON + 650 * deg_lon, 7), round(CENTER_LAT + 300 * deg_lat, 7)],
                ]]
            },
            "properties": {
                "site_id": "SITE-C3-AGRI-EXTENSION",
                "site_name": "Fallow Agriculture Terrace",
                "total_area_sqm": 42000.0,
                "usable_fraction": 0.60,
                "terrain_slope_deg": 7.5,
                "distance_to_road_km": 0.85,
                "service_proximity_score": 0.62,
                "buffer_from_hazard_m": 420.0,
                "in_red_zone": False,
                "land_tenure_status": "AVAILABLE",
                "water_availability": "Groundwater potential good",
                "power_grid_access": "Low-tension line accessible",
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [round(CENTER_LON - 150 * deg_lon, 7), round(CENTER_LAT + 300 * deg_lat, 7)],
                    [round(CENTER_LON + 150 * deg_lon, 7), round(CENTER_LAT + 300 * deg_lat, 7)],
                    [round(CENTER_LON + 150 * deg_lon, 7), round(CENTER_LAT + 480 * deg_lat, 7)],
                    [round(CENTER_LON - 150 * deg_lon, 7), round(CENTER_LAT + 480 * deg_lat, 7)],
                    [round(CENTER_LON - 150 * deg_lon, 7), round(CENTER_LAT + 300 * deg_lat, 7)],
                ]]
            },
            "properties": {
                "site_id": "SITE-D4-COMMUNITY-GROUND",
                "site_name": "Gram Panchayat Open Meadow",
                "total_area_sqm": 28000.0,
                "usable_fraction": 0.65,
                "terrain_slope_deg": 6.0,
                "distance_to_road_km": 0.15,
                "service_proximity_score": 0.82,
                "buffer_from_hazard_m": 350.0,
                "in_red_zone": False,
                "land_tenure_status": "AVAILABLE",
                "water_availability": "Overhead tank adjacent",
                "power_grid_access": "Immediate 3-phase grid",
            }
        },
        # Excluded Site 1: Steep Slope Violation (> 16 degrees)
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [round(CENTER_LON - 850 * deg_lon, 7), round(CENTER_LAT + 50 * deg_lat, 7)],
                    [round(CENTER_LON - 600 * deg_lon, 7), round(CENTER_LAT + 50 * deg_lat, 7)],
                    [round(CENTER_LON - 600 * deg_lon, 7), round(CENTER_LAT + 220 * deg_lat, 7)],
                    [round(CENTER_LON - 850 * deg_lon, 7), round(CENTER_LAT + 220 * deg_lat, 7)],
                    [round(CENTER_LON - 850 * deg_lon, 7), round(CENTER_LAT + 50 * deg_lat, 7)],
                ]]
            },
            "properties": {
                "site_id": "SITE-E5-WEST-CLIFF-EXCLUDED",
                "site_name": "Western Scarp Escarpment",
                "total_area_sqm": 35000.0,
                "usable_fraction": 0.50,
                "terrain_slope_deg": 22.4, # Violation: > 16 deg
                "distance_to_road_km": 1.2,
                "service_proximity_score": 0.35,
                "buffer_from_hazard_m": 120.0, # Violation: < 250 m
                "in_red_zone": False,
                "land_tenure_status": "AVAILABLE",
            }
        },
        # Excluded Site 2: Red Hazard Zone Encroachment
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [round(CENTER_LON - 400 * deg_lon, 7), round(CENTER_LAT - 500 * deg_lat, 7)],
                    [round(CENTER_LON - 100 * deg_lon, 7), round(CENTER_LAT - 500 * deg_lat, 7)],
                    [round(CENTER_LON - 100 * deg_lon, 7), round(CENTER_LAT - 300 * deg_lat, 7)],
                    [round(CENTER_LON - 400 * deg_lon, 7), round(CENTER_LAT - 300 * deg_lat, 7)],
                    [round(CENTER_LON - 400 * deg_lon, 7), round(CENTER_LAT - 500 * deg_lat, 7)],
                ]]
            },
            "properties": {
                "site_id": "SITE-F6-RIVER-BASIN-EXCLUDED",
                "site_name": "Lower Flood Basin Flat",
                "total_area_sqm": 48000.0,
                "usable_fraction": 0.60,
                "terrain_slope_deg": 2.1,
                "distance_to_road_km": 0.30,
                "service_proximity_score": 0.50,
                "buffer_from_hazard_m": 0.0,
                "in_red_zone": True, # Direct violation
                "land_tenure_status": "AVAILABLE",
            }
        }
    ]

    # Save to files
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, "project_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(project_metadata, f, indent=2)

    with open(os.path.join(base_dir, "sample_buildings.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": buildings}, f, indent=2)

    with open(os.path.join(base_dir, "sample_hazard_zones.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": hazard_zones}, f, indent=2)

    with open(os.path.join(base_dir, "sample_relocation_sites.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": relocation_candidates}, f, indent=2)

    print(f"Generated sample datasets in: {base_dir}")
    print(f"  - Buildings: {len(buildings)} items (4 classes, confidence scores, review flags)")
    print(f"  - Hazard Zones: {len(hazard_zones)} zones (Red/Amber/Green with AHP ratings)")
    print(f"  - Relocation Candidates: {len(relocation_candidates)} sites (4 eligible, 2 hard-excluded)")


if __name__ == "__main__":
    generate_sample_study_dataset("c:/Users/LIKHITHA SHREE L C/OneDrive/Desktop/MinorProject/data/samples")

