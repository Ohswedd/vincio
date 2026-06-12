"""Connector hub tests — every connector runs offline via injected
clients/transports."""

import json
import sqlite3

import httpx
import pytest

from vincio.connectors import CONNECTORS, connect, register_connector
from vincio.connectors.confluence import ConfluenceConnector
from vincio.connectors.gcs import GCSConnector
from vincio.connectors.github import GitHubConnector
from vincio.connectors.notion import NotionConnector
from vincio.connectors.s3 import S3Connector
from vincio.connectors.slack import SlackConnector
from vincio.connectors.sql import SQLConnector
from vincio.connectors.web import WebConnector
from vincio.core.errors import ConfigError, LoaderError


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestRegistry:
    def test_connect_builds_builtins(self):
        connector = connect("web", urls=["https://example.com"])
        assert isinstance(connector, WebConnector)

    def test_unknown_kind_raises(self):
        with pytest.raises(ConfigError):
            connect("carrier_pigeon")

    async def test_custom_connector_registers(self):
        from vincio.core.types import Document

        @register_connector("static")
        class StaticConnector:
            name = "static"

            async def load(self):
                return [Document(text="hello")]

        try:
            docs = await connect("static").load()
            assert docs[0].text == "hello"
        finally:
            CONNECTORS.pop("static", None)


class TestWeb:
    async def test_html_page(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                html="<html><head><title>Refund Policy</title></head>"
                "<body><p>Refunds within 30 days.</p></body></html>",
            )

        connector = WebConnector(["https://example.com/policy"], client=mock_client(handler))
        [doc] = await connector.load()
        assert doc.title == "Refund Policy"
        assert "Refunds within 30 days." in doc.text
        assert doc.source_uri == "https://example.com/policy"
        assert doc.metadata["connector"] == "web"

    async def test_http_error_raises_loader_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        connector = WebConnector(["https://example.com"], client=mock_client(handler))
        with pytest.raises(LoaderError):
            await connector.load()


class TestGitHub:
    async def test_loads_repo_files(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "/git/trees/" in request.url.path:
                return httpx.Response(
                    200,
                    json={
                        "tree": [
                            {"path": "README.md", "type": "blob"},
                            {"path": "src/app.py", "type": "blob"},
                            {"path": "logo.png", "type": "blob"},
                            {"path": "src", "type": "tree"},
                        ]
                    },
                )
            if request.url.path.endswith("/contents/README.md"):
                return httpx.Response(200, text="# Handbook\n\nRefunds within 30 days.")
            if request.url.path.endswith("/contents/src/app.py"):
                return httpx.Response(200, text="def refund():\n    return 30\n")
            return httpx.Response(404)

        connector = GitHubConnector("acme/handbook", client=mock_client(handler))
        docs = await connector.load()
        titles = {d.title for d in docs}
        assert titles == {"README.md", "src/app.py"}  # png filtered out
        readme = next(d for d in docs if d.title == "README.md")
        assert readme.sections  # markdown sections extracted
        assert readme.source_uri == "https://github.com/acme/handbook/blob/HEAD/README.md"
        code = next(d for d in docs if d.title == "src/app.py")
        assert code.metadata["language"] == "python"

    def test_bad_repo_raises(self):
        with pytest.raises(LoaderError):
            GitHubConnector("not-a-repo")


class TestSQL:
    async def test_sqlite_connection(self):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE faq (id INTEGER, question TEXT, answer TEXT)")
        connection.execute(
            "INSERT INTO faq VALUES (1, 'Refund window?', 'Refunds within 30 days.')"
        )
        connector = SQLConnector(
            "SELECT * FROM faq",
            connection=connection,
            id_column="id",
            title_column="question",
        )
        [doc] = await connector.load()
        assert doc.title == "Refund window?"
        assert "Refunds within 30 days." in doc.text
        assert doc.metadata["row"]["id"] == "1"

    async def test_sqlite_url(self, tmp_path):
        path = tmp_path / "kb.db"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE notes (body TEXT)")
        connection.execute("INSERT INTO notes VALUES ('SLA is 99.9 percent.')")
        connection.commit()
        connection.close()
        connector = SQLConnector("SELECT * FROM notes", url=f"sqlite:///{path}")
        [doc] = await connector.load()
        assert "SLA is 99.9 percent." in doc.text

    def test_needs_url_or_connection(self):
        with pytest.raises(LoaderError):
            SQLConnector("SELECT 1")


class _StubS3Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _StubS3Client:
    def list_objects_v2(self, **kwargs):
        return {
            "Contents": [{"Key": "docs/policy.md"}, {"Key": "image.png"}],
            "IsTruncated": False,
        }

    def get_object(self, *, Bucket, Key):
        return {"Body": _StubS3Body(b"Refunds within 30 days.")}


class _StubBlob:
    def __init__(self, name: str) -> None:
        self.name = name
        self.updated = None

    def download_as_bytes(self) -> bytes:
        return b"Backups retained 35 days."


class _StubGCSClient:
    def list_blobs(self, bucket, prefix=""):
        return [_StubBlob("docs/security.md"), _StubBlob("photo.jpg")]


class TestCloudStores:
    async def test_s3(self):
        connector = S3Connector("kb-bucket", client=_StubS3Client())
        [doc] = await connector.load()
        assert doc.source_uri == "s3://kb-bucket/docs/policy.md"
        assert "Refunds within 30 days." in doc.text

    async def test_gcs(self):
        connector = GCSConnector("kb-bucket", client=_StubGCSClient())
        [doc] = await connector.load()
        assert doc.source_uri == "gs://kb-bucket/docs/security.md"
        assert "Backups retained 35 days." in doc.text


class TestNotion:
    async def test_database_pages(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/databases/db1/query"):
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "id": "page1",
                                "url": "https://notion.so/page1",
                                "properties": {
                                    "Name": {
                                        "type": "title",
                                        "title": [{"plain_text": "Refund Policy"}],
                                    }
                                },
                            }
                        ],
                        "has_more": False,
                    },
                )
            if request.url.path.endswith("/blocks/page1/children"):
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "type": "paragraph",
                                "paragraph": {"rich_text": [{"plain_text": "Refunds within 30 days."}]},
                            },
                            {"type": "divider", "divider": {}},
                        ],
                        "has_more": False,
                    },
                )
            return httpx.Response(404)

        connector = NotionConnector("secret", database_id="db1", client=mock_client(handler))
        [doc] = await connector.load()
        assert doc.title == "Refund Policy"
        assert doc.text == "Refunds within 30 days."
        assert doc.source_uri == "https://notion.so/page1"

    def test_needs_target(self):
        with pytest.raises(LoaderError):
            NotionConnector("secret")


class TestConfluence:
    async def test_space_pages(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["spaceKey"] == "KB"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "100",
                            "title": "SLA",
                            "body": {"storage": {"value": "<p>Uptime is 99.9 percent.</p>"}},
                            "version": {"when": "2026-06-01T00:00:00.000Z"},
                            "space": {"key": "KB"},
                            "_links": {"webui": "/spaces/KB/pages/100"},
                        }
                    ]
                },
            )

        connector = ConfluenceConnector(
            "https://acme.atlassian.net/wiki", space="KB", client=mock_client(handler)
        )
        [doc] = await connector.load()
        assert doc.title == "SLA"
        assert "Uptime is 99.9 percent." in doc.text
        assert doc.source_uri.endswith("/spaces/KB/pages/100")


class TestSlack:
    async def test_channel_history(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["channel"] == "C123"
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "messages": [
                        {"user": "U2", "text": "Refunds take 30 days.", "ts": "1700000001.0"},
                        {"user": "U1", "text": "How long do refunds take?", "ts": "1700000000.0"},
                    ],
                },
            )

        connector = SlackConnector("xoxb-token", ["C123"], client=mock_client(handler))
        [doc] = await connector.load()
        assert doc.title == "#C123"
        lines = doc.text.splitlines()
        assert lines[0] == "U1: How long do refunds take?"  # oldest first
        assert doc.metadata["message_count"] == 2

    async def test_api_error_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": False, "error": "invalid_auth"})

        connector = SlackConnector("bad", ["C123"], client=mock_client(handler))
        with pytest.raises(LoaderError):
            await connector.load()


class TestAppIntegration:
    def test_add_source_with_connector(self):
        from vincio import ContextApp, VincioConfig
        from vincio.providers import MockProvider

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                html="<html><head><title>KB</title></head>"
                "<body><p>Customers on the Pro plan may request refunds within 30 days.</p></body></html>",
            )

        config = VincioConfig()
        config.storage.metadata = "memory://"
        config.observability.exporter = "none"
        config.security.audit_log = False
        app = ContextApp(name="t", provider=MockProvider(), model="mock-1", config=config)
        connector = WebConnector(["https://kb.example.com"], client=mock_client(handler))
        app.add_source("kb", connector=connector, retrieval="hybrid_full")
        assert app.sources["kb"].document_count == 1
        assert app.sources["kb"].chunk_count >= 1
        assert app._sparse is not None and len(app._sparse) >= 1
        assert app._late_interaction is not None and len(app._late_interaction) >= 1

    def test_connector_metadata_round_trip(self):
        # Document JSON survives store round-trips (metadata is plain data).
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="plain text body")

        connector = WebConnector(["https://example.com/a.txt"], client=mock_client(handler))
        import asyncio

        [doc] = asyncio.run(connector.load())
        payload = json.loads(doc.model_dump_json())
        assert payload["metadata"]["connector"] == "web"
