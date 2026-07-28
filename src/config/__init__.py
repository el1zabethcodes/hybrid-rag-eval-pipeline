"""Configuration package for the Hybrid RAG Evaluation Pipeline.

Exports the ``AppSettings`` class and a convenience ``get_settings()``
function so callers can import from ``src.config`` directly.
"""

from __future__ import annotations

from src.config.settings import AppSettings, get_settings

__all__ = ["AppSettings", "get_settings"]
