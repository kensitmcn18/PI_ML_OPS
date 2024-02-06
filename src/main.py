from fastapi import FastAPI
from src.app.routes import main_endpoints

app = FastAPI()

app.include_router(main_endpoints.router, prefix="/api/v1", tags=["api"])