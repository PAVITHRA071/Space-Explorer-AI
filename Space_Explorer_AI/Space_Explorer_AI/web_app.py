from flask import Flask, jsonify, render_template_string, request, session
from uuid import uuid4
import numpy as np

import config
from environment import SpaceEnvironment
from training.trainer import train_and_save
from utils import storage

app = Flask(__name__)
app.secret_key = "space-explorer-ai-local-demo-key"

# Browser game sessions for a local demonstration.
games = {}

ACTIONS = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
}


# ---------------------------------------------------------
# GAME SESSION
# ---------------------------------------------------------

def get_game():
    game_id = session.get("game_id")

    if game_id not in games:
        game_id = str(uuid4())
        session["game_id"] = game_id

        env = SpaceEnvironment()
        initial_state = env.reset()

        games[game_id] = {
            "env": env,
            "state": initial_state,
            "algorithm": None,
            "q_table": None,
        }

    return games[game_id]


def reset_game(game):
    game["state"] = game["env"].reset()


# ---------------------------------------------------------
# GRID DATA FOR THE BROWSER
# ---------------------------------------------------------

def get_game_data(game):
    env = game["env"]

    position = tuple(env.position)
    goal = tuple(env.goal)
    asteroids = {tuple(p) for p in env.asteroids}
    crystals = {tuple(p) for p in env.remaining_crystals()}

    board = []

    for y, row in enumerate(config.DEFAULT_LAYOUT):
        cells = []

        for x, symbol in enumerate(row):
            point = (x, y)

            if point == position:
                cell = "ship"
            elif point in asteroids:
                cell = "asteroid"
            elif point in crystals:
                cell = "crystal"
            elif point == goal:
                cell = "goal"
            elif symbol == "S":
                cell = "start"
            else:
                cell = "empty"

            cells.append(cell)

        board.append(cells)

    return {
        "board": board,
        "position": list(position),
        "goal": list(goal),
        "steps": int(env.steps),
        "reward": float(env.total_reward),
        "status": str(env.status),
        "crystals": int(env.crystals_collected),
        "algorithm": game["algorithm"],
        "ai_loaded": game["q_table"] is not None,
        "actions": {
            "up": "Up",
            "down": "Down",
            "left": "Left",
            "right": "Right",
        },
    }


# ---------------------------------------------------------
# BROWSER INTERFACE
# ---------------------------------------------------------

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Space Explorer AI</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 24px;
    font-family: Arial, sans-serif;
    color: #edf3ff;
    background: #080d20;
}

.container {
    max-width: 1150px;
    margin: auto;
}

header {
    margin-bottom: 24px;
}

h1 {
    margin-bottom: 8px;
    color: #64dcff;
}

.subtitle {
    color: #aab8d7;
    line-height: 1.6;
}

.layout {
    display: grid;
    grid-template-columns: minmax(300px, 1.3fr) minmax(280px, 1fr);
    gap: 20px;
}

.panel {
    background: #111a35;
    border: 1px solid #29375d;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
}

h2 {
    font-size: 19px;
    margin-top: 0;
}

.board {
    display: grid;
    grid-template-columns: repeat(10, minmax(0, 1fr));
    gap: 4px;
    margin: 20px 0;
}

.cell {
    aspect-ratio: 1;
    min-width: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #1b2948;
    border: 1px solid #304267;
    border-radius: 5px;
    font-size: clamp(12px, 2vw, 25px);
}

.ship { background: #155c70; }
.asteroid { background: #572c3c; }
.crystal { background: #493e75; }
.goal { background: #185b42; }
.start { background: #263957; }

.legend {
    color: #c1cce5;
    font-size: 13px;
    line-height: 1.9;
}

.stats {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-bottom: 18px;
}

.stat {
    background: #0b132b;
    border-radius: 8px;
    padding: 12px;
}

.stat small {
    display: block;
    color: #9aaccf;
    margin-bottom: 7px;
}

.stat strong {
    font-size: 20px;
    color: #64dcff;
    overflow-wrap: anywhere;
}

button, select, input {
    font: inherit;
    border-radius: 7px;
    padding: 10px 12px;
}

button {
    border: 1px solid #3e6394;
    background: #1d385d;
    color: white;
    cursor: pointer;
    margin: 4px 3px 4px 0;
}

button:hover {
    background: #2a527e;
}

button:disabled {
    opacity: 0.5;
    cursor: wait;
}

.primary {
    background: #087e91;
    border-color: #159db0;
}

.danger {
    background: #582d40;
}

input, select {
    width: 100%;
    margin: 8px 0 12px;
    background: #0b132b;
    border: 1px solid #33466d;
    color: white;
}

label {
    color: #c7d5f0;
    font-size: 14px;
}

.controls {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 4px;
    max-width: 230px;
    margin: 10px auto;
}

.controls button {
    margin: 0;
}

.empty-control {
    visibility: hidden;
}

.message {
    color: #a9d9e7;
    min-height: 24px;
    line-height: 1.5;
    overflow-wrap: anywhere;
}

.result-row {
    padding: 10px 0;
    border-bottom: 1px solid #29375d;
    line-height: 1.6;
}

.muted {
    color: #9aaccf;
    font-size: 13px;
    line-height: 1.6;
}

@media (max-width: 760px) {
    body { padding: 12px; }
    .layout { grid-template-columns: 1fr; }
    .panel { padding: 14px; }
}
</style>
</head>

<body>
<div class="container">

<header>
    <h1>🚀 Space Explorer AI</h1>
    <div class="subtitle">
        A Comparative Study of Q-Learning and SARSA for Intelligent Space Navigation.
        Collect energy crystals, avoid asteroids, and reach the destination planet.
    </div>
</header>

<div class="layout">

<section>
    <div class="panel">
        <h2>Space Navigation Grid</h2>
        <div id="board" class="board"></div>

        <div class="legend">
            🚀 Spacecraft &nbsp; ☄️ Asteroid &nbsp; 💎 Energy crystal<br>
            🌍 Destination &nbsp; 🛰️ Starting location
        </div>

        <hr style="border-color:#29375d; margin:18px 0;">

        <h2>Manual Controls</h2>
        <div class="controls">
            <button class="empty-control" disabled> </button>
            <button onclick="move('up')">⬆ Up</button>
            <button class="empty-control" disabled> </button>

            <button onclick="move('left')">⬅ Left</button>
            <button onclick="move('down')">⬇ Down</button>
            <button onclick="move('right')">➡ Right</button>
        </div>

        <button onclick="resetGame()">↻ Reset Game</button>
        <p class="muted">Use the buttons to move manually, or train an agent and watch its learned policy.</p>
    </div>

    <div class="panel">
        <h2>Game Status</h2>
        <div class="stats">
            <div class="stat">
                <small>Steps</small>
                <strong id="steps">0</strong>
            </div>
            <div class="stat">
                <small>Total Reward</small>
                <strong id="reward">0</strong>
            </div>
            <div class="stat">
                <small>Crystals Collected</small>
                <strong id="crystals">0</strong>
            </div>
            <div class="stat">
                <small>Game Status</small>
                <strong id="status">Running</strong>
            </div>
        </div>
        <div class="message" id="message">Ready to explore!</div>
    </div>
</section>

<section>
    <div class="panel">
        <h2>Train an AI Agent</h2>

        <label for="algorithm">Select algorithm</label>
        <select id="algorithm">
            <option value="q_learning">Q-Learning</option>
            <option value="sarsa">SARSA</option>
        </select>

        <label for="episodes">Training episodes</label>
        <input id="episodes" type="number" min="10" max="200000" value="2500">

        <button class="primary" id="trainButton" onclick="trainAgent()">Train Selected Agent</button>

        <p class="muted">
            Training may take some time. The trained Q-table and evaluation results
            are saved in your project folder.
        </p>

        <div class="result-row">
            <strong>Training result</strong>
            <div id="trainingResult" class="message">No training run yet.</div>
        </div>
    </div>

    <div class="panel">
        <h2>AI Playback</h2>
        <p class="muted">Load a saved model, then let the agent select actions using its Q-table.</p>

        <label for="playAlgorithm">Model to watch</label>
        <select id="playAlgorithm">
            <option value="q_learning">Q-Learning</option>
            <option value="sarsa">SARSA</option>
        </select>

        <button class="primary" onclick="loadAI()">Load AI Model</button>
        <button id="watchButton" onclick="toggleWatch()" disabled>▶ Watch AI</button>
        <button onclick="resetGame()">↻ Reset</button>

        <div id="aiMessage" class="message">Train an agent first if no model is saved.</div>
    </div>

    <div class="panel">
        <h2>Algorithm Comparison</h2>
        <button onclick="compareAlgorithms()">Compare Saved Models</button>
        <div id="comparison" class="muted" style="margin-top:12px;">
            Train both algorithms to compare their evaluation results.
        </div>
    </div>
</section>

</div>
</div>

<script>
let watching = false;
let watchTimer = null;
let busy = false;

const symbols = {
    ship: "🚀",
    asteroid: "☄️",
    crystal: "💎",
    goal: "🌍",
    start: "🛰️",
    empty: ""
};

async function api(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Something went wrong.");
    }

    return data;
}

function stopWatching() {
    watching = false;

    if (watchTimer !== null) {
        clearTimeout(watchTimer);
        watchTimer = null;
    }

    document.getElementById("watchButton").textContent = "▶ Watch AI";
}

function render(data) {
    const board = document.getElementById("board");
    board.innerHTML = "";

    data.board.forEach(row => {
        row.forEach(type => {
            const cell = document.createElement("div");
            cell.className = "cell " + type;
            cell.textContent = symbols[type] || "";
            board.appendChild(cell);
        });
    });

    document.getElementById("steps").textContent = data.steps;
    document.getElementById("reward").textContent =
        Number(data.reward).toFixed(1);
    document.getElementById("crystals").textContent = data.crystals;
    document.getElementById("status").textContent = data.status;

    if (data.status === "success") {
        document.getElementById("message").textContent =
            "🎉 Mission completed! The spacecraft reached the destination.";
    } else if (data.status === "crashed") {
        document.getElementById("message").textContent =
            "💥 The spacecraft hit an asteroid. Reset to try again.";
    } else if (data.status === "timeout") {
        document.getElementById("message").textContent =
            "⏳ Maximum steps reached. Reset to start a new episode.";
    } else {
        document.getElementById("message").textContent =
            data.algorithm
                ? "Algorithm: " + data.algorithm
                : "Navigate to the destination and collect crystals.";
    }
}

async function refreshState() {
    const data = await api("/api/state");
    render(data);
}

async function move(direction) {
    if (busy) return;

    stopWatching();

    try {
        const data = await api("/api/move", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({direction})
        });

        render(data);
    } catch (error) {
        showMessage(error.message);
    }
}

async function resetGame() {
    stopWatching();

    try {
        const data = await api("/api/reset", {method: "POST"});
        render(data);
    } catch (error) {
        showMessage(error.message);
    }
}

function showMessage(message) {
    document.getElementById("message").textContent = message;
}

async function trainAgent() {
    if (busy) return;

    stopWatching();

    const button = document.getElementById("trainButton");
    const result = document.getElementById("trainingResult");
    const algorithm = document.getElementById("algorithm").value;
    const episodes = Number(document.getElementById("episodes").value);

    if (!Number.isInteger(episodes) || episodes < 10 || episodes > 200000) {
        result.textContent = "Enter a whole number from 10 to 200000.";
        return;
    }

    busy = true;
    button.disabled = true;
    button.textContent = "Training...";
    result.textContent = "Training started. Please wait...";

    try {
        const data = await api("/api/train", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({algorithm, episodes})
        });

        result.textContent =
            data.name + " trained for " + data.episodes + " episodes. " +
            "Success rate: " + data.success_rate.toFixed(2) + "%. " +
            "Average reward: " + data.mean_reward.toFixed(2) + ". " +
            "Time: " + data.duration.toFixed(2) + " seconds.";

        document.getElementById("playAlgorithm").value = algorithm;
        document.getElementById("aiMessage").textContent =
            "Training complete. Click Load AI Model to play.";

        await refreshState();
        await compareAlgorithms();
    } catch (error) {
        result.textContent = error.message;
    } finally {
        busy = false;
        button.disabled = false;
        button.textContent = "Train Selected Agent";
    }
}

async function loadAI() {
    stopWatching();

    const algorithm = document.getElementById("playAlgorithm").value;

    try {
        const data = await api("/api/load-ai", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({algorithm})
        });

        document.getElementById("watchButton").disabled = false;
        document.getElementById("aiMessage").textContent =
            data.name + " model loaded. Click Watch AI to begin.";

        render(data);
    } catch (error) {
        document.getElementById("aiMessage").textContent = error.message;
        document.getElementById("watchButton").disabled = true;
    }
}

function toggleWatch() {
    if (watching) {
        stopWatching();
        return;
    }

    watching = true;
    document.getElementById("watchButton").textContent = "⏸ Pause AI";
    watchStep();
}

async function watchStep() {
    if (!watching) return;

    try {
        const data = await api("/api/ai-step", {method: "POST"});
        render(data);

        if (["success", "crashed", "timeout"].includes(data.status)) {
            stopWatching();
            document.getElementById("aiMessage").textContent =
                "Episode finished. Reset the game to watch another episode.";
            return;
        }

        watchTimer = setTimeout(watchStep, 500);
    } catch (error) {
        stopWatching();
        document.getElementById("aiMessage").textContent = error.message;
    }
}

async function compareAlgorithms() {
    const container = document.getElementById("comparison");
    container.textContent = "Loading comparison...";

    try {
        const data = await api("/api/compare");

        if (data.results.length === 0) {
            container.textContent =
                "No saved evaluation results yet. Train an agent first.";
            return;
        }

        container.innerHTML = "";

        data.results.forEach(item => {
            const row = document.createElement("div");
            row.className = "result-row";

            const title = document.createElement("strong");
            title.textContent = item.name;

            const details = document.createElement("div");

            if (item.error) {
                details.textContent = item.error;
            } else {
                details.textContent =
                    "Episodes: " + item.episodes +
                    " | Success rate: " + item.success_rate.toFixed(2) + "%" +
                    " | Mean reward: " + item.mean_reward.toFixed(2) +
                    " | Mean successful steps: " +
                    (item.mean_steps_success === null
                        ? "N/A"
                        : item.mean_steps_success.toFixed(2));
            }

            row.appendChild(title);
            row.appendChild(details);
            container.appendChild(row);
        });
    } catch (error) {
        container.textContent = error.message;
    }
}

refreshState().catch(error => showMessage(error.message));
compareAlgorithms();
</script>

</body>
</html>
"""


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


# ---------------------------------------------------------
# GAME API
# ---------------------------------------------------------

@app.route("/api/state")
def api_state():
    game = get_game()
    return jsonify(get_game_data(game))


@app.route("/api/move", methods=["POST"])
def api_move():
    game = get_game()
    env = game["env"]

    if env.status != "running":
        return jsonify({"error": "The episode has ended. Reset the game first."}), 400

    data = request.get_json(silent=True) or {}
    direction = data.get("direction")

    if direction not in ACTIONS:
        return jsonify({"error": "Invalid movement direction."}), 400

    next_state, reward, done, info = env.step(ACTIONS[direction])
    game["state"] = next_state

    # Keep the environment's latest state tuple for clarity.
    # The next reset or AI action will update this value.
    return jsonify(get_game_data(game))


@app.route("/api/reset", methods=["POST"])
def api_reset():
    game = get_game()
    reset_game(game)
    return jsonify(get_game_data(game))


# ---------------------------------------------------------
# TRAIN AND SAVE
# ---------------------------------------------------------

@app.route("/api/train", methods=["POST"])
def api_train():
    data = request.get_json(silent=True) or {}

    algorithm = data.get("algorithm")
    if algorithm not in config.ALGORITHM_KEYS:
        return jsonify({"error": "Select Q-Learning or SARSA."}), 400

    try:
        episodes = config.parse_field("episodes", data.get("episodes", 2500))
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400

    settings = storage.load_settings()
    settings.episodes = episodes

    seed = None if settings.seed == -1 else settings.seed

    try:
        result, evaluation = train_and_save(
            algorithm,
            settings,
            seed=seed,
        )
    except Exception as error:
        app.logger.exception("Training failed")
        return jsonify({"error": f"Training failed: {error}"}), 500

    game = get_game()
    reset_game(game)
    game["algorithm"] = config.ALGORITHM_NAMES[algorithm]
    game["q_table"] = result.agent.q_table.copy()

    return jsonify({
        "algorithm": algorithm,
        "name": config.ALGORITHM_NAMES[algorithm],
        "episodes": result.episodes_completed,
        "duration": result.duration,
        "mean_reward": float(evaluation["mean_reward"]),
        "success_rate": float(evaluation["success_rate"]),
        "mean_steps_success": evaluation["mean_steps_success"],
        "collision_rate": float(evaluation["collision_rate"]),
    })


# ---------------------------------------------------------
# LOAD A SAVED MODEL
# ---------------------------------------------------------

@app.route("/api/load-ai", methods=["POST"])
def api_load_ai():
    data = request.get_json(silent=True) or {}
    algorithm = data.get("algorithm")

    if algorithm not in config.ALGORITHM_KEYS:
        return jsonify({"error": "Select Q-Learning or SARSA."}), 400

    game = get_game()
    env = game["env"]

    try:
        model = storage.load_model(
            algorithm,
            env.signature(),
            env.n_states,
            env.n_actions,
        )
    except storage.ModelError as error:
        return jsonify({"error": str(error)}), 400

    reset_game(game)
    game["algorithm"] = config.ALGORITHM_NAMES[algorithm]
    game["q_table"] = model.q_table

    return jsonify(get_game_data(game))


# ---------------------------------------------------------
# AI MOVEMENT USING THE SAVED Q-TABLE
# ---------------------------------------------------------

@app.route("/api/ai-step", methods=["POST"])
def api_ai_step():
    game = get_game()
    env = game["env"]
    q_table = game["q_table"]

    if q_table is None:
        return jsonify({"error": "Load a trained AI model first."}), 400

    if env.status != "running":
        return jsonify({"error": "Episode ended. Reset the game first."}), 400

    state_tuple = game["state"]

    # If no valid tuple is stored, reset the environment.
    if not isinstance(state_tuple, tuple):
        state_tuple = env.reset()
        game["state"] = state_tuple

    state_index = env.state_to_index(state_tuple)
    action = int(np.argmax(q_table[state_index]))

    next_state, reward, done, info = env.step(action)
    game["state"] = next_state

    return jsonify(get_game_data(game))


# ---------------------------------------------------------
# COMPARE SAVED EVALUATION RESULTS
# ---------------------------------------------------------

@app.route("/api/compare")
def api_compare():
    results = []

    for algorithm in config.ALGORITHM_KEYS:
        name = config.ALGORITHM_NAMES[algorithm]
        meta = storage.model_info(algorithm)

        if meta is None:
            continue

        if "error" in meta:
            results.append({
                "name": name,
                "error": "Saved model metadata could not be read.",
            })
            continue

        evaluation = meta.get("evaluation")

        if not evaluation:
            results.append({
                "name": name,
                "error": "No evaluation results available. Retrain this model.",
            })
            continue

        mean_steps = evaluation.get("mean_steps_success")

        results.append({
            "name": name,
            "episodes": int(meta.get("episodes", 0)),
            "success_rate": float(evaluation.get("success_rate", 0)),
            "mean_reward": float(evaluation.get("mean_reward", 0)),
            "mean_steps_success": (
                None if mean_steps is None else float(mean_steps)
            ),
            "collision_rate": float(evaluation.get("collision_rate", 0)),
        })

    return jsonify({"results": results})


# ---------------------------------------------------------
# START FLASK
# ---------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("SPACE EXPLORER AI - BROWSER VERSION")
    print("Open http://127.0.0.1:5000 in Chrome")
    print("=" * 55)

    app.run(host="127.0.0.1", port=5000, debug=True)