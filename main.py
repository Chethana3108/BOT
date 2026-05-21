from fastapi import FastAPI

from app import app as recommendation_app
from web import app as knowledge_app


main_app = FastAPI(
    title="Honda AI Unified Server"
)


main_app.mount("/recommendation", recommendation_app)

main_app.mount("/knowledge", knowledge_app)


@main_app.get("/")
def home():
    return {
        "status": "Unified Honda AI Server Running"
    }
    
    
    
    uvicorn main:main_app --host 0.0.0.0 --port 8000