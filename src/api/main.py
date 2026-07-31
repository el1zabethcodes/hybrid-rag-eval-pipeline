"""Production entrypoint for launching the FastAPI application."""

from src.api.app import create_app
from src.config.settings import get_settings

settings = get_settings()
app = create_app(settings)
