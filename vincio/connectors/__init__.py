"""Vincio connector hub.

Pluggable data connectors that feed the document engine: every connector
returns plain :class:`~vincio.core.types.Document` objects with provenance
(``source_uri``, connector metadata, timestamps), so anything a connector
loads chunks, indexes, budgets, and cites exactly like a local file.

Built-in kinds: ``web``, ``github``, ``sql``, ``s3``, ``gcs``, ``notion``,
``confluence``, ``slack``, ``jira``, ``linear``, ``gdrive``, ``sharepoint``,
``salesforce``, ``zendesk``, ``bigquery``, ``snowflake``. Construct directly or
via :func:`connect`::

    from vincio.connectors import connect

    docs = await connect("web", urls=["https://example.com/docs"]).load()
    app.add_source("kb", connector=connect("github", repo="acme/handbook"))
    app.add_source("issues", connector=connect("jira", base_url="https://acme.atlassian.net"))

The REST connectors (``jira``, ``linear``, ``gdrive``, ``sharepoint``,
``salesforce``, ``zendesk``, plus the originals) run on the core ``httpx``
dependency and accept an injected client for testing. Connectors with heavy
dependencies import them lazily (``s3`` → ``vincio[s3]``, ``gcs`` →
``vincio[gcs]``, ``bigquery`` → ``vincio[bigquery]``, ``snowflake`` →
``vincio[snowflake]``) and also accept an injected client/connection so they
round-trip offline. Third-party connectors register on install via the
``vincio.connectors`` entry-point group (see :mod:`vincio.plugins`); custom
in-process connectors register with :func:`register_connector`.
"""

from .base import CONNECTORS, Connector, connect, register_connector

__all__ = ["CONNECTORS", "Connector", "connect", "register_connector"]
