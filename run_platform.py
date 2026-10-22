"""
One-Click Platform Launcher
Starts the FastAPI backend and launches the interactive Decision Support Web Dashboard.
"""

import sys
import os
import time
import webbrowser
import subprocess

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("=" * 70)
    print("  AI-POWERED POST-DISASTER DAMAGE ASSESSMENT & RELOCATION PLATFORM")
    print("  Spatial Decision Support System - Satellite Analytics & TOPSIS")
    print("=" * 70)

    # Add project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("\n[1/3] Verifying core AI & GIS runtime dependencies...")
    try:
        import torch


        import fastapi
        import uvicorn
        print("      [OK] Core dependencies (PyTorch, FastAPI, Uvicorn) are ready.")
    except ImportError as e:
        print(f"      [X] Missing core requirement: {e}")
        sys.exit(1)

    try:
        import rasterio
        import geopandas
        print("      [OK] Optional GeoTIFF drivers (Rasterio, GeoPandas) detected.")
    except ImportError:
        print("      [i] Running in standard vector GeoJSON mode (Rasterio/GeoPandas omitted).")

    print("\n[2/3] Checking dataset files...")
    sample_meta = os.path.join(project_root, "data", "samples", "project_metadata.json")
    if not os.path.exists(sample_meta):
        print("      * Generating sample study dataset...")
        from data.samples.generate_sample_project import generate_sample_study_dataset
        generate_sample_study_dataset(os.path.join(project_root, "data", "samples"))
    print("      [OK] Sample disaster study dataset is loaded.")

    url = "http://127.0.0.1:8000"
    print(f"\n[3/3] Starting Decision Support Server at: {url}")
    print(f"      - Interactive GIS Command Dashboard: {url}")
    print(f"      - Swagger API Docs: {url}/docs")
    print(f"      - Press Ctrl+C to terminate the server.\n")

    # Open browser automatically after a brief delay
    time.sleep(1.0)
    webbrowser.open(url)

    # Launch uvicorn
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()

