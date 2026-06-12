"""Vincio server mode. Requires ``pip install "vincio[server]"``."""

from .app import create_app

__all__ = ["create_app"]
