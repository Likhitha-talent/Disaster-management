"""
FastAPI Main Application Entry Point
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from backend.app.api.routes import router

app = FastAPI(
    title="AI-Powered Post-Disaster Damage Assessment & Relocation Platform",
    description="Spatial Decision Support System combining Siamese U-Net Damage Assessment, AHP Multi-Hazard Screening, and TOPSIS Relocation Ranking.",
    version="1.0.0",
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Mount frontend directory
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend"))
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_dashboard():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "message": "AI-Powered Post-Disaster Damage Assessment API is active.",
        "documentation_url": "/docs",
        "health_check": "/api/health",
    }


if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)

