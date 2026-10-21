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
    print("=" * 70)
    print("  AI-POWERED POST-DISASTER DAMAGE ASSESSMENT & RELOCATION PLATFORM")
    print("  The National Institute of Engineering, Mysuru • Dept. of ISE")
    print("  Course: BIS586 (Batch B10) | Guide: Nandini P, Assistant Professor")
    print("=" * 70)

    # Add project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("\n[1/3] Verifying core AI & GIS runtime dependencies...")
    try:
        import torch
        import rasterio
        import geopandas
        import fastapi
        import uvicorn
        print("      ✓ PyTorch, Rasterio, GeoPandas, and FastAPI are ready.")
    except ImportError as e:
        print(f"      ✗ Missing requirement: {e}")
        sys.exit(1)

    print("\n[2/3] Checking dataset files...")
    sample_meta = os.path.join(project_root, "data", "samples", "project_metadata.json")
    if not os.path.exists(sample_meta):
        print("      * Generating sample study dataset...")
        from data.samples.generate_sample_project import generate_sample_study_dataset
        generate_sample_study_dataset(os.path.join(project_root, "data", "samples"))
    print("      ✓ Sample disaster study dataset is loaded.")

    url = "http://127.0.0.1:8000"
    print(f"\n[3/3] Starting Decision Support Server at: {url}")
    print(f"      • Interactive GIS Command Dashboard: {url}")
    print(f"      • Swagger API Docs: {url}/docs")
    print(f"      • Press Ctrl+C to terminate the server.\n")

    # Open browser automatically after a brief delay
    time.sleep(1.0)
    webbrowser.open(url)

    # Launch uvicorn
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()

