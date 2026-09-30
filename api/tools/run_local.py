"""Run the API directly on Windows with the local GPU environment."""

import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    print(f"Starting local Manga Translator API on http://{settings.HOST}:{settings.PORT}")
    print(f"Interactive Swagger Docs: http://{settings.HOST}:{settings.PORT}/docs")
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
