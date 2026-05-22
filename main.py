from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@main_app.get("/")
def home():
    return {
        "status": "Unified Honda AI Server Running"
    }
