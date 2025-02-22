from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles 
from app.routes.image_routes import router as image_router
from fastapi.middleware.cors import CORSMiddleware

def create_app() -> FastAPI:
    """
    Функция для создания приложения,
    чтобы в будущем при необходимости передавать настройки.
    """
    app = FastAPI(title="Colorizer Project")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    app.include_router(image_router)
    return app

app = create_app()
