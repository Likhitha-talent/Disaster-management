# AI-Powered Satellite Image-Based Post-Disaster Damage Assessment
### with GIS-Based Hazard Mapping and Relocation Decision Support

**Institution:** The National Institute of Engineering, Mysuru  
**Department:** Information Science and Engineering  
**Academic Year:** 2026–27 | **Semester:** V, Section 'B'  
**Course Code:** BIS586 (Minor Project) | **Batch No:** B10  
**Faculty Guide:** Nandini P, Assistant Professor  

#### Project Team
- **Likhitha Shree L C** (USN: `4NI24IS091`)
- **Namratha M Sherughar** (USN: `4NI24IS114`)
- **Nischitha R N** (USN: `4NI24IS119`)
- **Panchami Urs S** (USN: `4NI24IS123`)

---

## 🌟 Executive Summary

Disaster management authorities (SDMAs/NDMA) face significant hurdles in the critical hours following natural catastrophes: manual satellite image interpretation is slow, pure pixel maps cannot be queried for spatial planning, and relocation decisions are often made ad-hoc. 

This platform delivers an **end-to-end, explainable spatial decision-support prototype** that:
1. **Localizes building footprints** using U-Net with ResNet encoders on pre-disaster satellite imagery.
2. **Classifies damage into four tiers** (*undamaged, minor damage, major damage, destroyed*) using a shared-weight Siamese U-Net with feature-difference fusion.
3. **Flags uncertain predictions** (confidence < 0.65) for human-in-the-loop review to ensure auditability.
4. **Delineates multi-hazard screening zones** (*Red, Amber, Green*) using Analytic Hierarchy Process (AHP) weighted overlay across slope, flood corridors, land cover, and population exposure ($CR < 0.10$).
5. **Screens and ranks candidate relocation sites** using mandatory exclusionary constraints, preliminary carrying capacity calculations ($C = A \cdot u / a$), and multi-criteria **TOPSIS** ranking.
6. **Visualizes results in an interactive Web GIS Command Dashboard** with instant GeoJSON and CSV export capabilities.

---

## 🚀 Quick Start Guide

### 1. Requirements
- Python 3.10+ (tested with Python 3.14 on Windows 11)
- Key Libraries: PyTorch, Rasterio, GeoPandas, Shapely, FastAPI, Uvicorn

### 2. Run the Platform
Run the launcher script from the root directory:
```powershell
python run_platform.py
```
This automatically:
- Validates the environment and sample dataset
- Starts the FastAPI REST API server at `http://127.0.0.1:8000`
- Launches the Web GIS Command Dashboard in your browser

Alternatively, start via Uvicorn directly:
```powershell
python -m uvicorn backend.app.main:app --reload --port 8000
```

### 3. Run Automated Tests
Execute the full test suite across ML models, GIS algorithms, and API endpoints:
```powershell
python -m pytest gis/tests backend/tests ml/tests -v
```

---

## 🗺️ Web Dashboard Features

Navigate to `http://127.0.0.1:8000`:
- **Overview Tab:** Summary KPIs, damage distribution breakdown, structural footprint metrics, hostable capacity.
- **Damage Assessment Tab:** Interactive map showing color-coded building damage footprints, confidence thresholds, and a Building Inspector tool with class probability bars and human review override.
- **Hazard Screening Tab:** Red, Amber, and Green zones with live AHP factor weight sliders and Consistency Ratio validation ($CR < 0.10$).
- **Relocation Decision Support Tab:** Candidate relocation sites ranked by TOPSIS relative closeness score, with carrying capacity and exclusion audit details.
- **Human Review & Reports Tab:** Queue of flagged uncertain buildings for analyst verification, and 1-click export for CSV rankings, GeoJSON footprints, and JSON executive summaries.

---

## 📚 Technical Documentation
- **[System Architecture](docs/ARCHITECTURE.md):** End-to-end pipeline and software design.
- **[AHP & TOPSIS Methodology](docs/AHP_TOPSIS_METHODOLOGY.md):** Mathematical formulation, formulas, and proofs.
- **[Data Management Catalogue](docs/DATA_CATALOGUE.md):** Satellite data, DEM, and land cover references.
