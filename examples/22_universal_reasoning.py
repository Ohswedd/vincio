"""Universal reasoning for native and non-reasoning models, fully offline.

Shows adaptive depth, provider-independent passes, deterministic verification,
and automatic integration with ordinary ``app.run``. The public receipt contains
decisions and costs, never chain-of-thought.
"""

from __future__ import annotations

import json
import warnings

from vincio import ContextApp
from vincio.providers import MockProvider
from vincio.stability import VincioExperimentalWarning

warnings.simplefilter("ignore", VincioExperimentalWarning)


def responder(request):
    prompt = "\n".join(message.text for message in request.messages)
    if "semantic request router" in prompt:
        return json.dumps(
            {
                "language": "es",
                "primary_task": "summarization",
                "depth": "direct",
                "difficulty": 0.1,
                "task_kinds": ["summarization"],
                "needs_live_external_information": False,
                "web_preference": "auto",
                "tool_names": [],
                "confidence": 0.98,
                "signals": ["simple_transformation"],
            }
        )
    if "Resume este texto" in prompt:
        return "Resumen breve en español."
    if "bounded answer correction" in prompt:
        return "The checked equality is 2 + 2 = 4."
    if "universal reasoning control" in prompt:
        return "The proposed equality is 2 + 2 = 5."
    return "TITLE"


# This mock deliberately has no native reasoning capability. The in-house
# engine therefore supplies the adaptive reasoning architecture itself.
provider = MockProvider(responder=responder, reasoning=False)
app = ContextApp(name="universal-reasoning", provider=provider, model="mock-1")
app.use_reasoning_engine()

# Easy work stays on the exact one-pass run path.
simple = app.run("Rewrite this title in uppercase")

# Hard work gets two independent candidates; the arithmetic kernel refutes both
# wrong equalities, then a single bounded correction is re-certified.
hard = app.run(
    "Prove logically whether 2 + 2 = 5, calculate the equality, and detect the contradiction."
)
receipt = hard.metadata["universal_reasoning"]

# Non-English requests are classified by the configured model itself. Vincio
# does not need a language allow-list; deterministic policy remains in control.
spanish = app.run("Resume este texto en una frase: Vincio verifica sus respuestas.")

print(
    f"simple passes={simple.metadata['universal_reasoning']['passes']} | "
    f"hard passes={receipt['passes']} strategy={receipt['strategy']} "
    f"corrected={receipt['corrected']} answer={hard.raw_text!r}"
)
print(
    f"language={spanish.metadata['universal_reasoning']['detected_language']} "
    f"semantic_route={spanish.metadata['universal_reasoning']['semantic_routing_succeeded']} "
    f"answer={spanish.raw_text!r}"
)
