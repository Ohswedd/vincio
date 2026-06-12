"""GitHub connector: repository files via the GitHub REST API."""

from __future__ import annotations

import httpx

from ..core.concurrency import gather_bounded
from ..core.errors import LoaderError
from ..core.types import Document
from ..documents.parsers import extract_markdown_sections
from .base import managed_client, register_connector

__all__ = ["GitHubConnector"]

_TEXT_EXTENSIONS = (
    ".md", ".markdown", ".txt", ".rst",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
)

_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".jsx": "javascript", ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
}


@register_connector("github")
class GitHubConnector:
    name = "github"

    def __init__(
        self,
        repo: str,
        *,
        ref: str = "HEAD",
        path_prefix: str = "",
        token: str | None = None,
        extensions: tuple[str, ...] = _TEXT_EXTENSIONS,
        max_files: int = 200,
        api_base: str = "https://api.github.com",
        client: httpx.AsyncClient | None = None,
        max_concurrency: int = 4,
    ) -> None:
        if "/" not in repo:
            raise LoaderError(f"github repo must be 'owner/name', got {repo!r}")
        self.repo = repo
        self.ref = ref
        self.path_prefix = path_prefix.strip("/")
        self.token = token
        self.extensions = extensions
        self.max_files = max_files
        self.api_base = api_base.rstrip("/")
        self.client = client
        self.max_concurrency = max_concurrency

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _wanted(self, path: str) -> bool:
        if self.path_prefix and not path.startswith(self.path_prefix):
            return False
        return path.lower().endswith(self.extensions)

    async def _fetch_file(self, client: httpx.AsyncClient, path: str) -> Document:
        response = await client.get(
            f"{self.api_base}/repos/{self.repo}/contents/{path}",
            params={"ref": self.ref},
            headers={**self._headers(), "Accept": "application/vnd.github.raw+json"},
        )
        response.raise_for_status()
        text = response.text
        extension = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        is_markdown = extension in (".md", ".markdown")
        metadata: dict = {"connector": self.name, "repo": self.repo, "path": path, "ref": self.ref}
        if extension in _LANGUAGES:
            metadata["language"] = _LANGUAGES[extension]
        return Document(
            source_uri=f"https://github.com/{self.repo}/blob/{self.ref}/{path}",
            title=path,
            media_type="text/markdown" if is_markdown else f"text/x-{metadata.get('language', 'plain')}",
            text=text,
            sections=[s.model_dump(mode="json") for s in extract_markdown_sections(text)]
            if is_markdown
            else [],
            metadata=metadata,
        )

    async def load(self) -> list[Document]:
        async with managed_client(self.client) as client:
            try:
                response = await client.get(
                    f"{self.api_base}/repos/{self.repo}/git/trees/{self.ref}",
                    params={"recursive": "1"},
                    headers=self._headers(),
                )
                response.raise_for_status()
                tree = response.json().get("tree", [])
                paths = [
                    entry["path"]
                    for entry in tree
                    if entry.get("type") == "blob" and self._wanted(entry.get("path", ""))
                ][: self.max_files]
                return await gather_bounded(
                    (self._fetch_file(client, path) for path in paths),
                    limit=self.max_concurrency,
                )
            except httpx.HTTPError as exc:
                raise LoaderError(f"github connector failed for {self.repo}: {exc}") from exc
