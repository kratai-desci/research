"""E13 v5: the function dependency graph that BFCL's multi-turn set is built on, rebuilt for our writer.

BFCL (v3 multi-turn) built its tasks this way: each function is a node, an edge joins two functions where one's output is
an input of the other, execution paths are sampled from that graph, and a task is written around each path. BFCL drew
its edges by hand; here Gemini proposes them from the tools' descriptions and simulator source, and the names are
checked against the docs. Two kinds of edge:
    data      the first function's output gives a value the second needs (get_user_id -> send_message)
    requires  the simulator only lets the second succeed after the first has changed the state (pressBrakePedal -> startEngine)
Edges inside each tool set come from one call per set; edges across sets from one call per main + companion pair.

    python -m lab.e13.graph --out lab/e13/graph.json      (on a VM, inside bfcl-eval's venv)
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from lab.e13.practice import CLASSES, COMPANION, PRIMARY, functions, retry, tool_text
from lab.vertex import Gemini

PROMPT = """You map how a set of software tools depend on each other. You get the tools' descriptions and the source of
the simulator that runs them. List the directed edges A -> B between functions where:
- "data": a value in A's output is a natural input to B (an id, a symbol, a path, a price, a token, a number to compute with);
- "requires": the simulator only lets B succeed after A has changed the state (for example being logged in, a pedal
  pressed, being in the right folder, a booking made).
{scope}
Be thorough but only list real dependencies a user task would follow. Return JSON:
{{"edges": [{{"from": "<function name>", "to": "<function name>", "kind": "data" | "requires", "why": "<few words>"}}]}}"""


def edges_for(llm, classes: list[str]) -> list[dict]:
    owner = {f["name"]: c for c in classes for f in functions([c])}
    scope = ("Only edges inside this one tool set." if len(classes) == 1 else
             f"Only edges that cross between {classes[0]} and {classes[1]} (either direction); leave out edges inside one set.")
    # parse inside the retry: a reply cut off mid-JSON is asked for again
    out = retry(lambda: json.loads(llm.generate(PROMPT.format(scope=scope), tool_text(classes), json_mode=True,
                                                temperature=0.0, max_tokens=65000, thinking="medium", timeout=900)), 4)
    keep = []
    for e in out.get("edges", []):
        a, b = owner.get(e.get("from")), owner.get(e.get("to"))
        if a and b and e["from"] != e["to"] and e.get("kind") in ("data", "requires") and (len(classes) == 1 or a != b):
            keep.append({"from": f"{a}.{e['from']}", "to": f"{b}.{e['to']}", "kind": e["kind"], "why": e.get("why", "")})
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-3-flash-preview")
    a = ap.parse_args()
    llm = Gemini(a.model)
    groups = [[c] for c in CLASSES] + [[p, c] for p in PRIMARY for c in COMPANION]
    with ThreadPoolExecutor(8) as pool:
        found = list(pool.map(lambda g: edges_for(llm, g), groups))
    edges = list({(e["from"], e["to"]): e for es in found for e in es}.values())
    json.dump({"model": a.model, "edges": sorted(edges, key=lambda e: (e["from"], e["to"]))}, open(a.out, "w"), indent=1)
    for g, es in zip(groups, found):
        print(" + ".join(g), len(es))
    print("edges", len(edges))


if __name__ == "__main__":
    main()
