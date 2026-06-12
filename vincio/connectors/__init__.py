"""Vincio connector hub.

Pluggable data connectors that feed the document engine: every connector
returns plain :class:`~vincio.core.types.Document` objects with provenance
(``source_uri``, connector metadata, timestamps), so anything a connector
loads chunks, indexes, budgets, and cites exactly like a local file.

Built-in kinds: ``web``, ``github``, ``sql``, ``s3``, ``gcs``, ``notion``,
``confluence``, ``slack``. Construct directly or via :func:`connect`::

    from vincio.connectors import connect

    docs = await connect("web", urls=["https://example.com/docs"]).load()
    app.add_source("kb", connector=connect("github", repo="acme/handbook"))

Connectors with heavy dependencies import them lazily (S3 needs
``pip install "vincio[s3]"``, GCS ``vincio[gcs]``); the REST connectors run
on the core ``httpx`` dependency and accept an injected client for testing.
Custom connectors register with :func:`register_connector`.
"""

from .base import CONNECTORS, Connector, connect, register_connector

__all__ = ["CONNECTORS", "Connector", "connect", "register_connector"]
