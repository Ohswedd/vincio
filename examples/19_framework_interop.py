"""Framework interop + provider/vector-store breadth (0.9).

Drop existing LangChain and LlamaIndex assets straight into Vincio, and reach
any OpenAI-compatible gateway or vector store through one interface. The
adapters are duck-typed, so the fakes below stand in for real LangChain tools
and LlamaIndex readers (which expose the same methods) and the example runs
fully offline.
"""

from _shared import example_provider, json_responder

from vincio import ContextApp
from vincio.interop import add_langchain_tool, from_llamaindex_reader
from vincio.providers import openai_compatible
from vincio.retrieval import build_embedder
from vincio.storage import VECTOR_BACKENDS, build_vector_index


class WeatherTool:
    """A LangChain-style tool — real ``langchain.tools.BaseTool`` works identically."""

    name = "get_weather"
    description = "Look up the current weather for a city."
    args = {"city": {"type": "string"}}

    def invoke(self, payload: dict) -> dict:
        return {"city": payload["city"], "temp_c": 22, "sky": "sunny"}


class TinyReader:
    """A LlamaIndex-style reader — real readers expose ``.load_data()``."""

    def load_data(self):
        class _Node:
            def __init__(self, text, metadata):
                self._text = text
                self.metadata = metadata

            def get_content(self):
                return self._text

        return [_Node("Refunds are processed within 5 business days.", {"file_path": "kb.md"})]


provider, model = example_provider(json_responder({"answer": "Refunds take 5 business days."}))
app = ContextApp(name="interop_demo", provider=provider, model=model)

add_langchain_tool(app, WeatherTool())  # LangChain tool -> Vincio (registered + enabled)
docs = from_llamaindex_reader(TinyReader())  # LlamaIndex reader -> Vincio Documents
app.add_source("kb", documents=docs, retrieval="hybrid")


if __name__ == "__main__":
    # Provider breadth: any OpenAI-compatible gateway via a named preset (no network here).
    print("groq endpoint:", openai_compatible("groq", api_key="demo").base_url)
    print("embedder dim:", build_embedder("local").dim)
    print("vector backends:", VECTOR_BACKENDS)
    memory_index = build_vector_index("memory", build_embedder("local"))
    print("tools:", app.enabled_tools, "| kb docs:", len(docs), "| index:", memory_index.name)
    result = app.run("What is the refund window?")
    print("answer:", result.output)
