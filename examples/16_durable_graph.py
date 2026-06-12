"""Durable stateful graph: checkpoints, human-in-the-loop, time-travel."""

from vincio import ContextApp
from vincio.agents import interrupt

app = ContextApp(name="durable_graph_demo")

# A review pipeline as a stateful graph. Checkpoints persist in the app's
# metadata store after every step, so the thread survives interrupts (and,
# with a SQLite/Postgres store, process restarts).
graph = app.graph("contract_review")
graph.add_node("ingest", lambda s: {"clauses": ["auto-renewal", "late fees"]})
graph.add_node("analyze", lambda s: {"risk": "high" if "auto-renewal" in s["clauses"] else "low"})
graph.add_node(
    "approve",
    lambda s: {"approved": interrupt(s, {"question": f"Risk is {s['risk']} — proceed?"})},
)
graph.add_node("report", lambda s: {"report": f"risk={s['risk']} approved={s['approved']}"})
graph.add_edge("ingest", "analyze")
graph.add_edge("analyze", "approve")
graph.add_edge("approve", "report")

if __name__ == "__main__":
    review = graph.compile()

    # 1. Run until the human gate pauses the graph.
    paused = review.invoke({})
    print("status:", paused.status)
    print("asked:", paused.interrupt_payload)

    # 2. Optionally edit state before resuming (edit-and-resume).
    review.update_state(paused.thread_id, {"risk": "medium"})

    # 3. Resume with the human's answer; the gate re-runs and receives it.
    done = review.resume(paused.thread_id, value=True)
    print("report:", done.state["report"])

    # 4. Time-travel: fork the post-analyze checkpoint and replay the rest.
    history = review.history(paused.thread_id)
    print("checkpoints:", [(c.step, c.status) for c in history])
    forked = review.fork(history[2].id)
    replayed = review.resume(forked, value=False)
    print("replayed report:", replayed.state["report"])

    # 5. Compose the graph into a pipeline with streaming node events.
    from vincio import compose

    pipeline = compose(graph.compile()) | (lambda state: state.get("report", "paused for approval"))
    print("composed:", pipeline.call({"clauses": []}))
