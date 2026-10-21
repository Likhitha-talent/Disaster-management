# Project System Architecture

**Project Title:** AI-Powered Satellite Image-Based Post-Disaster Damage Assessment with GIS-Based Hazard Mapping and Relocation Decision Support  
**Course & Code:** BIS586 | 5th Semester, Academic Year 2026–27  
**Department:** Information Science and Engineering, The National Institute of Engineering, Mysuru  
**Faculty Guide:** Nandini P, Assistant Professor  
**Student Team:** Likhitha Shree L C, Namratha M Sherughar, Nischitha R N, Panchami Urs S  

---

## 1. High-Level Architecture Overview

The system is organized into a modular 4-tier pipeline:

```
[ Bi-Temporal Satellite Imagery (xBD / Sentinel-2) ] + [ DEM & Ancillary GIS Data ]
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Computer Vision / Deep Learning Tier (PyTorch)           │
│   • Stage A: U-Net (ResNet-34) Building Footprint Localizer │
│   • Stage B: Shared-Weight Siamese U-Net Damage Classifier  │
│   • Output: 4 Classes (Undamaged, Minor, Major, Destroyed)  │
│   • Confidence Calibration & Human-in-the-Loop Review Flag  │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Geospatial Analysis Tier (Rasterio, GeoPandas, Shapely)  │
│   • Footprint Polygonization & Georeferencing               │
│   • Multi-Hazard Weighted Overlay: R(x) = ∑ w_i * r_i(x)    │
│   • AHP Weight Calibration (CR < 0.10)                      │
│   • Red / Amber / Green Hazard Screening Zones              │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Relocation Decision Engine (TOPSIS & MCDA)               │
│   • Hard Exclusion Constraints (Slope > 16°, Red Zone, Area)│
│   • Preliminary Carrying Capacity: C_i = (A_i * u_i) / a_i  │
│   • TOPSIS Closeness Ranking & Factor-wise Explainability   │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Presentation & API Tier (FastAPI + React / Leaflet)      │
│   • RESTful Endpoints with Pydantic & GeoJSON Payloads      │
│   • Interactive SDMA Command Dashboard                      │
│   • Real-Time Building Inspector & Human Review Tool        │
│   • Report Generation (CSV, GeoJSON, Executive Summaries)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Layout & Module Responsibilities

```
MinorProject/
├── backend/
│   ├── app/
│   │   ├── api/routes.py          # FastAPI endpoints (CRUD, analysis, export)
│   │   ├── models/                # Database and data schemas
│   │   ├── schemas/schemas.py     # Pydantic validation models
│   │   ├── services/              # Business logic & engine orchestration
│   │   └── main.py                # App entrypoint & static frontend mount
│   └── tests/test_api.py          # Integration tests for REST endpoints
├── data/
│   ├── raw/                       # Original raw rasters & geo-annotations
│   ├── processed/                 # Tiled, normalized, and preprocessed imagery
│   ├── outputs/                   # Exported masks, predictions, GeoJSONs
│   └── samples/                   # Curated reproducible sample study area
├── docs/
│   ├── ARCHITECTURE.md            # System specification & pipeline design
│   ├── AHP_TOPSIS_METHODOLOGY.md  # Detailed mathematical formulation
│   └── DATA_CATALOGUE.md          # Satellite & terrain data sources
├── frontend/
│   └── index.html                 # Complete interactive Leaflet web GIS dashboard
├── gis/
│   ├── hazard_analysis/overlay.py # AHP multi-hazard composite risk engine
│   ├── relocation/topsis.py       # TOPSIS screening & capacity engine
│   └── tests/                     # Unit tests for GIS algorithms
├── ml/
│   ├── src/
│   │   ├── models/unet.py         # Stage A: U-Net localizer
│   │   ├── models/siamese_unet.py # Stage B: Siamese U-Net damage classifier
│   │   ├── models/losses.py       # Class-weighted CE + Soft Dice loss
│   │   └── inference/engine.py    # Pixel-to-polygon inference & confidence
│   └── tests/test_models.py       # Model forward pass tests
├── reports/                       # Generated CSV summaries & evaluation charts
└── run_platform.py                # One-click platform launcher
```
