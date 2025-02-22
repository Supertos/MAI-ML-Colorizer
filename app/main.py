

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles 
from app.routes.image_routes import router as image_router
from fastapi.middleware.cors import CORSMiddleware

def create_app() -> FastAPI:
    #Создание приложения
    app = FastAPI(title="Colorizer Project")
    
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    app.include_router(image_router)
    return app

app = create_app()
