"""Framework interop: use LangChain and LlamaIndex assets inside Vincio, and
Vincio's inside them.

The ``from_*`` adapters are duck-typed and import nothing heavy, so existing
tools, retrievers, loaders, and embeddings drop in without adding a dependency.
The ``to_*`` adapters build real framework objects and need the matching extra
(``vincio[langchain]`` / ``vincio[llamaindex]``)::

    from vincio.interop import add_langchain_tool, from_llamaindex_reader

    add_langchain_tool(app, my_langchain_tool)
    docs = from_llamaindex_reader(SimpleDirectoryReader("./docs"))
"""

from .langchain import (
    LangChainEmbedder,
    LangChainRetriever,
    add_langchain_tool,
    from_langchain_document,
    from_langchain_documents,
    from_langchain_embeddings,
    from_langchain_loader,
    from_langchain_retriever,
    from_langchain_tool,
    register_langchain_tool,
    to_langchain_document,
    to_langchain_documents,
    to_langchain_embeddings,
    to_langchain_retriever,
    to_langchain_tool,
)
from .llamaindex import (
    LlamaIndexEmbedder,
    LlamaIndexRetriever,
    add_llamaindex_tool,
    from_llamaindex_document,
    from_llamaindex_documents,
    from_llamaindex_embedding,
    from_llamaindex_reader,
    from_llamaindex_retriever,
    from_llamaindex_tool,
    register_llamaindex_tool,
    to_llamaindex_document,
    to_llamaindex_documents,
    to_llamaindex_embedding,
    to_llamaindex_retriever,
    to_llamaindex_tool,
)

__all__ = [
    # LangChain
    "LangChainEmbedder",
    "LangChainRetriever",
    "add_langchain_tool",
    "from_langchain_document",
    "from_langchain_documents",
    "from_langchain_embeddings",
    "from_langchain_loader",
    "from_langchain_retriever",
    "from_langchain_tool",
    "register_langchain_tool",
    "to_langchain_document",
    "to_langchain_documents",
    "to_langchain_embeddings",
    "to_langchain_retriever",
    "to_langchain_tool",
    # LlamaIndex
    "LlamaIndexEmbedder",
    "LlamaIndexRetriever",
    "add_llamaindex_tool",
    "from_llamaindex_document",
    "from_llamaindex_documents",
    "from_llamaindex_embedding",
    "from_llamaindex_reader",
    "from_llamaindex_retriever",
    "from_llamaindex_tool",
    "register_llamaindex_tool",
    "to_llamaindex_document",
    "to_llamaindex_documents",
    "to_llamaindex_embedding",
    "to_llamaindex_retriever",
    "to_llamaindex_tool",
]
