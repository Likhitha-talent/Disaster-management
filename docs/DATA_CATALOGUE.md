# Data Sources & Data Management Catalogue

**Course:** BIS586 Minor Project | Dept. of ISE, NIE Mysuru  

---

| Layer | Source / Product | Resolution | Spatial Reference | Verification Checklist |
|---|---|---|---|---|
| **Bi-temporal Building Damage** | xBD / xView2 Dataset (`xview2.org`) | ~0.5 m / 0.8 m GSD | WGS84 / UTM | Check paired-image coregistration, 4-tier annotations (no, minor, major, destroyed), license terms. |
| **Regional Optical Imagery** | Copernicus Sentinel-2 (L2A MSI) | 10 m / 20 m | UTM WGS84 | Filter cloud cover < 10%, verify acquisition dates relative to disaster event timestamp. |
| **Elevation & Hydrology** | Copernicus DEM GLO-30 / SRTM | 30 m | EPSG:4326 | Fill voids, derive slope in degrees, compute D8 flow accumulation corridor. |
| **Land Cover Classification** | ESA WorldCover 10 m | 10 m | EPSG:4326 | Verify class taxonomy: tree cover, shrubland, cropland, built-up, water bodies. |
| **Population Exposure** | WorldPop Gridded Population (UN-adjusted) | 100 m | EPSG:4326 | Raster resolution check; aggregate counts within building buffers. |
| **Infrastructure & Roads** | OpenStreetMap (OSM) via Overpass API | Vector | EPSG:4326 | Extract primary/secondary roads, hospitals, primary health centers (PHCs), shelters. |
| **Relocation Candidate Sites** | SDMA / District Revenue GIS Polygons | Vector Polygon | EPSG:4326 | Ensure verified land tenure, ownership records, accessibility, and utility easements. |

---

### Coordinate Reference System (CRS) Protocol
All spatial layers are projected to consistent local UTM zones (e.g., EPSG:32643 for Karnataka / Western Ghats) for accurate Euclidean distance (meters) and polygonal area ($\text{m}^2$) measurements, and reprojected to EPSG:4326 for Leaflet Web GIS visualization.
