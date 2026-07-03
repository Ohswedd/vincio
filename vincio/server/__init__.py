"""Vincio server mode. Requires ``pip install "vincio[server]"``.

The AG-UI translators (:mod:`vincio.server.agui`) are dependency-free and import
without FastAPI, so generative-UI events can be produced anywhere a run streams;
only :func:`create_app` needs the ``server`` extra.
"""

from __future__ import annotations

from .agui import (
    AGUIEvent,
    AGUIEventType,
    agent_stream_to_agui,
    agui_sse,
    mcp_ui_event,
    run_stream_to_agui,
)
from .app import create_app

__all__ = [
    "create_app",
    "AGUIEvent",
    "AGUIEventType",
    "run_stream_to_agui",
    "agent_stream_to_agui",
    "agui_sse",
    "mcp_ui_event",
]
