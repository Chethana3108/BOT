from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

from app import app as recommendation_app
from web import app as knowledge_app
from web import startup_event

main_app = FastAPI(
    title="Honda AI Unified Server"
)

# ==============================
# CORS
# ==============================

main_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# MOUNT APPS
# ==============================

main_app.mount("/chat", recommendation_app)

main_app.mount("/knowledge_chat", knowledge_app)

startup_event()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HONDA_HTML = os.path.join(BASE_DIR, "honda.html")

@main_app.get("/")
def home():
    if os.path.exists(HONDA_HTML):
        return FileResponse(HONDA_HTML)
    return {
        "status": "Unified Honda AI Server Running"
    }

@main_app.get("/status")
def status():
    return {
        "status": "Unified Honda AI Server Running"
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    print(f"\nStarting Honda AI Unified Server on port {port}...")
    uvicorn.run("main:main_app", host="0.0.0.0", port=port, reload=False)
