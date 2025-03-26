from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.image_routes import router as image_router

def create_app() -> FastAPI:
    app = FastAPI(title="Image Processing API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  #можно указать конкретный домен фронтенда, например: ["http://localhost:3000"]
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(image_router, prefix="/api/images", tags=["Images"])

    return app

app = create_app()
