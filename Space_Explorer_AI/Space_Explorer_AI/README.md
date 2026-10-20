# Space Explorer AI: A Comparative Study of Q-Learning and SARSA for Intelligent Space Navigation

A complete Reinforcement Learning game written in Python. A spaceship learns, by trial and error, to collect energy crystals, avoid asteroids and reach a destination planet. Two algorithms, **Q-Learning** and **SARSA**, are implemented from scratch (tabular, no deep learning), trained on the same map and compared with real measurements.

## 1. Abstract

This project builds a 10 x 8 grid space-exploration game and trains an AI pilot with two classic temporal-difference algorithms. The environment, agents, training loop, evaluation and a Pygame interface are separate modules. The user can play manually, train either algorithm, watch a trained agent fly, and open a Matplotlib dashboard that compares learning curves, success rates, mission lengths and a final evaluation table. All numbers shown come from actual training and evaluation runs; nothing is hard-coded.

## 2. Problem statement

How can an agent that knows nothing about its world learn a safe and rewarding route through a space full of hazards, and how do two closely related algorithms, Q-Learning and SARSA, differ when they learn the same task?

## 3. Objectives

1. Implement a reusable grid environment with `reset()`, `step()`, `get_state()` and `render()`.
2. Implement Q-Learning and SARSA independently, with correct and different update targets.
3. Train both under identical conditions and evaluate them with a fixed policy.
4. Provide a polished interactive game: human play, AI playback, training screen, comparison dashboard and settings.
5. Save models and CSV results that can be used in a college report.

## 4. Existing system and proposed system

**Existing approach:** many classroom RL demos only print numbers in a console, use a single algorithm, or hard-code the path. It is then hard to see learning happen or to compare algorithms.

**Proposed system:** a graphical game where learning is visible (live training screen), the agent's behaviour can be watched step by step, both algorithms run on the same environment, and the comparison graphs are produced from real data.

## 5. Software and hardware requirements

| | Requirement |
|---|---|
| OS | Windows 10/11 (also works on macOS/Linux) |
| Python | 3.11 or 3.12 recommended (3.13 also worked in testing) |
| Libraries | pygame, numpy, matplotlib (see `requirements.txt`) |
| Hardware | Any ordinary laptop; no GPU. Screen of at least 1024 x 660 pixels |
| Editor | Visual Studio Code or any terminal |

## 6. Technologies used

Python, Pygame (graphics and input), NumPy (Q-tables and maths), Matplotlib (graphs, drawn into the Pygame window), JSON/NPZ/CSV files (saving), `unittest` (tests).

## 7. Reinforcement Learning concepts in simple words

* **Agent**: the learner (the spaceship's pilot).
* **Environment**: the world the agent acts in (the grid).
* **State**: what the agent knows now: its x, y position and which crystals it has already collected.
* **Action**: a choice: Up, Down, Left or Right.
* **Reward**: a number telling the agent how good the last move was.
* **Policy**: the rule for choosing actions.
* **Q-value** `Q(s, a)`: how much total future reward the agent expects if it takes action `a` in state `s` and then keeps behaving well. All Q-values live in a table (the **Q-table**): one row per state, one column per action.
* **Exploration vs exploitation**: sometimes try something new (explore), otherwise use what you know (exploit). **Epsilon-greedy** explores with probability epsilon; epsilon starts high and **decays** each episode.
* **Episode**: one mission, from the start cell until it ends.
* **Learning rate (alpha)**: how strongly new experience changes old knowledge.
* **Discount factor (gamma)**: how much future rewards matter compared with immediate ones.

## 8. Q-Learning

Q-Learning is **off-policy**. After a move it looks at the next state and assumes the agent will take the *best* known action there, whatever it actually does. It therefore learns the value of the greedy policy while still exploring.

## 9. SARSA

SARSA is **on-policy**. Its name is the sequence State, Action, Reward, next State, next Action. After a move it uses the action that the epsilon-greedy policy has *really chosen* for the next step. Because that includes exploratory mistakes, SARSA learns the value of the policy it actually follows, which often makes it more cautious near dangers.

## 10. Update equations

```
Q-Learning:  Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
SARSA:       Q(s,a) <- Q(s,a) + alpha * [ r + gamma *        Q(s',a') - Q(s,a) ]
```

For a terminal state (planet reached or asteroid hit) there is no future, so the target is just `r` for both algorithms. A step-limit timeout is *not* treated as terminal, because the state does not contain the clock, so the future still exists from the agent's point of view.

Code: `agents/q_learning.py` and `agents/sarsa.py` (shared Q-table, epsilon-greedy selection and decay are in `agents/base_agent.py`).

## 11. Game rules and reward structure

Map (10 columns x 8 rows, defined in `config.py`): `S` start (bottom-left), `G` planet (top-right), `C` five crystals, `A` asteroids. A breadth-first search at start-up checks that the planet and every crystal can be reached without touching an asteroid.

| Event | Reward |
|---|---|
| Ordinary move (also bumping into the boundary; the ship stays in its cell) | -1 |
| Collecting a **new** crystal (replaces the -1 for that move) | +20 |
| Reaching the destination planet | +100 |
| Flying into an asteroid | -100 |

The episode ends when the planet is reached, an asteroid is hit, or 100 moves have been made ("out of fuel"). A crystal pays only once per episode because the collected crystals are part of the state.

**Reward tuning:** the initial reward values requested in the project brief worked as specified, so **no reward values were changed**. The only design decisions are those stated above (a crystal or the goal/asteroid reward replaces the -1 step penalty on that move; the 100-move limit).

Controls: arrow keys or WASD in human mode. Pause `P`, restart `R` (or Enter after a mission), menu `Esc`. In playback: `Space` play/pause, `-` / `+` speed.

## 12. Project architecture

```
 interface/ (Pygame screens)          training/trainer.py
   menu, game, training, comparison  -----> Trainer (episodes in small time slices)
            |                                    |
            |                          agents/ (Q-Learning, SARSA)
            |                                    |
            +------------> environment/space_environment.py   (no graphics code)
            |
            +------------> evaluation/ (fixed-policy evaluation, Matplotlib plots)
            +------------> utils/storage.py (models, CSV, settings)
```

The environment never imports Pygame, and training never draws anything. The training screen stays responsive because it trains for about 12 ms per frame and then redraws.

## 13. Folder structure

```
Space_Explorer_AI/
  main.py                  start the game / command-line tools
  config.py                map, rewards, defaults, colours
  requirements.txt  README.md  .gitignore
  environment/space_environment.py
  agents/                  base_agent.py  q_learning.py  sarsa.py
  training/trainer.py
  evaluation/              evaluate_agents.py  plots.py
  interface/               ui.py  menu.py  game_screen.py  training_screen.py  comparison_screen.py
  utils/storage.py
  tests/                   test_environment.py  test_q_learning.py  test_sarsa.py
                           test_training_pipeline.py  test_interface_smoke.py
  models/                  saved Q-tables (.npz) and metadata (.json)
  results/                 CSV files and comparison.png
  assets/README.md         (the game needs no external assets)
```

## 14. Installation on Windows

**Step 1: Extract the ZIP** into a folder such as `C:\Users\YourName\Desktop\Space_Explorer_AI`.

**Step 2: Open the project** by opening that folder in Visual Studio Code (File > Open Folder) and opening a terminal (Terminal > New Terminal).

**Step 3: Create a virtual environment**

```
python -m venv .venv
```

**Step 4: Activate it**

```
:: Command Prompt
.venv\Scripts\activate

# PowerShell
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the script, run `Set-ExecutionPolicy -Scope Process RemoteSigned` once and try again.

**Step 5: Install the dependencies**

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Step 6: Launch the game**

```
python main.py
```

**Step 7: Run the tests**

```
python -m unittest discover -s tests
```

## 15. How to use the game

* **Human play** (menu item 1): fly with the arrow keys or WASD.
* **Train** (items 2, 3, 4): a progress screen shows the algorithm, episode, exploration rate, last reward, 100-episode average reward and success rate, a live reward chart and a progress bar. **Stop** (or `Esc`) cancels safely; an unfinished run is never saved, so an earlier model stays intact. When training finishes, the model, history and evaluation are saved automatically.
* **Watch an agent** (items 5, 6): plays the trained agent with the **greedy policy**: in every state it takes the action with the highest Q-value. Ties are broken the same way every time (lowest action number: Up, Down, Left, Right). Use Play/Pause, Restart and speed buttons. If no model exists yet you get a message that training is required.
* **Compare Algorithms** (item 7): shows Graph 1 (learning curve), Graph 2 (success rate), Graph 3 (steps per successful mission) and Graph 4 (summary table). Use the tab buttons or keys `0`-`4` to view one graph larger, and **Save PNG** to write `results/comparison.png`.
* **Settings** (item 8): episodes, learning rate, discount factor, initial/decay/minimum exploration, playback speed and random seed. Every field is validated and a clear message is shown for bad input. Settings are stored in `settings.json`.

### Command-line tools (no window)

```
python main.py --train both                  :: train both with the saved settings
python main.py --train sarsa --episodes 5000 --seed 7
python main.py --multi-seed 5                :: 5 seeds x 2 algorithms, CSV tables, saved models untouched
python main.py --compare                     :: save results/comparison.png from the saved models
python -m evaluation.evaluate_agents         :: re-evaluate the saved models
```

## 16. Saving, loading and results

* `models/<algorithm>_model.npz`: Q-table and training history. `models/<algorithm>_meta.json`: hyper-parameters, seed, episodes, evaluation, map fingerprint and time.
* Each algorithm has its own files, so retraining one never touches the other.
* Models store a fingerprint of the map and rewards. A model trained for a different setup is refused with a clear message.
* Files are written atomically (temporary file, then rename).
* `.npz` files are loaded with `allow_pickle=False`, so they cannot run code. Still, only use model files you trained yourself.
* `results/training_history_<algorithm>.csv`: one row per training episode.
* `results/evaluation_summary.csv`: one row per algorithm (fixed-policy evaluation).
* `results/multi_seed_results.csv` and `multi_seed_summary.csv`: from `--multi-seed`.

## 17. Experimental evaluation methodology

* Both algorithms use the same map, rewards, hyper-parameters, Q-table initialisation (all zeros) and, when a seed is given, the same random seed.
* **Training behaviour is not used for the final numbers.** After training, a separate evaluation runs **200 episodes** with the greedy policy. With a fixed probability of **5%** a random move replaces the greedy one (engine glitches). Without this noise the map would be deterministic and all 200 episodes would be identical. A fixed evaluation seed (12345) gives both algorithms the same random sequence.
* Metrics: mean reward, success rate, mean steps per successful mission ("N/A" if there was no success), asteroid collision rate, timeout rate and mean crystals collected.
* Default hyper-parameters: alpha 0.1, gamma 0.95, epsilon 1.0 decaying by 0.998 per episode to a minimum of 0.01, 2,500 episodes.
* Learning curves use a trailing rolling average (window 100).

### Results from the author's test runs

These are real results produced in development (2,500 episodes each, defaults above). Yours will differ slightly on other seeds.

Single run, seed 42 (this is what the included models and `results/comparison.png` show):

| Algorithm | Mean reward | Success | Steps / success | Collisions | Crystals |
|---|---|---|---|---|---|
| Q-Learning | 140.1 | 90.0% | 20.9 | 10.0% | 3.77 |
| SARSA | 134.6 | 88.0% | 21.1 | 12.0% | 3.69 |

Five seeds (1-5), mean (standard deviation over seeds), from `results/multi_seed_summary.csv`:

| Algorithm | Mean reward | Success | Steps / success | Collisions | Crystals |
|---|---|---|---|---|---|
| Q-Learning | 136.0 (0.9) | 88.6% (0.4) | 21.0 (0.0) | 11.4% (0.4) | 3.70 (0.01) |
| SARSA | 130.4 (5.0) | 89.1% (1.1) | 19.8 (2.6) | 10.9% (1.1) | 3.33 (0.43) |

**Interpretation (cautious):** on this small map both algorithms reach about 89% success under evaluation noise and behave very similarly. Q-Learning had a somewhat higher mean reward and collected more crystals, and in the seed-42 learning curves it started improving earlier. SARSA's results varied more from seed to seed. The success and collision differences are within the seed-to-seed variation, and five seeds are too few to declare a winner. For a stronger claim, run `python main.py --multi-seed 20` and report the spread. The usual textbook expectation that SARSA keeps a safer distance from hazards was **not** clearly visible here.

## 18. Testing

```
python -m unittest discover -s tests
```

The tests cover: reset and start position, all four actions, boundaries, one-time crystal reward, asteroid and goal termination, the step limit, Q-Learning and SARSA update rules (including that SARSA uses the selected next action and the terminal case), short training runs for both algorithms, reproducibility, safe cancelling, model save/load and error handling, retraining one algorithm without touching the other, evaluation metrics (including "N/A"), graph generation, settings validation and a start-up/smoke test of the real Pygame application using a hidden window.

## 19. Limitations

* Tabular learning only works for small state spaces (here 2,560 states). A larger map or more crystals makes the table grow quickly.
* One fixed map. The agent learns this map, not a general skill of navigation.
* The 5% evaluation noise is a design choice; other noise levels give other numbers.
* Results are for the default hyper-parameters and few seeds; they are not a general statement about the algorithms.
* The interface was verified automatically with a hidden window and by inspecting rendered screenshots; it has not been tested by hand on every Windows screen size or scaling setting.

## 20. Possible future enhancements

* Several maps and a random-map generator, with evaluation on unseen maps.
* Expected SARSA, Double Q-Learning and n-step methods.
* Hyper-parameter search screen with heat-maps.
* A Q-value heat-map overlay on the grid during playback.
* Fuel as a state variable, moving asteroids, sound effects.
* A deep Q-network version for comparison with the tabular results.
