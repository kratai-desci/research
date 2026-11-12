"""A tiny HTTP bridge to WebShop's text environment, run inside WebShop's own Python 3.8 environment.

Usage (from the repo root):
  cd vendor/WebShop && .venv/bin/python ../../lab/scripts/webshop_server.py [synthetic|human]
  synthetic (default): WebShop's generated instructions, port 3100; human: human-written ones, port 3101
Endpoints:
  POST /reset {"session": <goal index>} -> {"observation", "instruction", "actions"}
  POST /step  {"action": "search[...]" | "click[...]"} -> {"observation", "reward", "done", "actions"}
  GET  /info -> {"goals": <number of goals>}
  GET  /outcome -> the hidden goal and what was bought (for the simulated user; never shown to the creature)
"""

import os
import sys

# WebShop builds goal lists from Python sets, whose order depends on the hash seed. Pin it, or a restart
# can reorder which instruction sits at which goal index and make runs incomparable.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

sys.path.insert(0, ".")

from flask import Flask, jsonify, request  # noqa: E402

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv  # noqa: E402

GOALS = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
PORT = 3101 if GOALS == "human" else 3100
env = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=int(GOALS == "human"))
app = Flask(__name__)


def actions():
    a = env.get_available_actions()
    return {"search": a["has_search_bar"], "clickables": a["clickables"]}


@app.post("/reset")
def reset():
    env.reset(session=int(request.json["session"]))
    return jsonify(observation=env.observation, instruction=env.instruction_text, actions=actions())


@app.post("/step")
def step():
    obs, reward, done, _ = env.step(request.json["action"])
    return jsonify(observation=obs, reward=reward, done=done, actions=actions())


@app.get("/info")
def info():
    return jsonify(goals=len(env.server.goals), goal_set=GOALS)


@app.get("/outcome")
def outcome():
    """The hidden goal and what was bought, for the simulated user. Never shown to the creature."""
    session = env.server.user_sessions.get(env.session, {})
    goal = session.get("goal", {})
    asin = session.get("asin")
    product = env.server.product_item_dict.get(asin, {}) if asin else {}
    text = " ".join([product.get("Title", ""), *product.get("BulletPoints", []), product.get("Description", "")]).lower()
    options = goal.get("goal_options", {})
    return jsonify(
        done=bool(session.get("done")),
        reward=float(session.get("reward", 0.0)),
        info=session.get("verbose_info", {}),
        goal={
            "attributes": goal.get("attributes", []),
            "options": list(options.values()) if isinstance(options, dict) else list(options),
            "price_upper": goal.get("price_upper", 0),
        },
        purchased={
            "asin": asin,
            "title": product.get("Title", ""),
            "options": [str(v) for v in (session.get("options") or {}).values()],
            "price": env.server.product_prices.get(asin) if asin else None,
            "text": text,
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("WEBSHOP_PORT", PORT))
    app.run(host=os.environ.get("WEBSHOP_HOST", "127.0.0.1"), port=port, threaded=False)
