"""Space Explorer AI - entry point.

    python main.py                         open the game window
    python main.py --train both            train from the command line (no window)
    python main.py --train sarsa --episodes 5000 --seed 7
    python main.py --multi-seed 5          train both algorithms with 5 seeds, save CSV tables
    python main.py --compare               save results/comparison.png from the saved models
"""

import argparse
import sys

import config
from utils import storage


class App:
    """Owns the window and switches between screens."""

    def __init__(self):
        import pygame
        from interface import ui

        # Only the display and fonts are needed (the game has no sound).
        pygame.display.init()
        pygame.font.init()
        ui.reset_fonts()
        pygame.display.set_caption("Space Explorer AI - Q-Learning vs SARSA")
        self.pygame = pygame
        self.surface = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.settings = storage.load_settings()
        self.starfield = ui.Starfield(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.running = True
        self.time = 0.0
        self.screen = None
        self.go_menu()

    # --- navigation -------------------------------------------------
    def go_menu(self, message=None):
        from interface.menu import MenuScreen
        self.screen = MenuScreen(self, message)

    def quit(self):
        self.running = False

    def start_human(self):
        from interface.game_screen import GameScreen
        self.screen = GameScreen(self, "human")

    def start_training(self, algorithms):
        from interface.training_screen import TrainingScreen
        problems = self.settings.problems()
        if problems:
            self.go_menu((problems[0] + " Open Settings to fix it.", config.COLORS["red"]))
            return
        self.screen = TrainingScreen(self, algorithms)

    def start_watch(self, algorithm):
        from environment import SpaceEnvironment
        from interface.game_screen import GameScreen
        env = SpaceEnvironment()
        try:
            model = storage.load_model(algorithm, env.signature(), env.n_states, env.n_actions)
        except storage.ModelError as error:
            self.go_menu((str(error), config.COLORS["orange"]))
            return
        self.screen = GameScreen(self, "ai", q_table=model.q_table, algorithm=algorithm)

    def start_compare(self):
        from interface.comparison_screen import ComparisonScreen
        self.screen = ComparisonScreen(self)

    def start_settings(self):
        from interface.menu import SettingsScreen
        self.screen = SettingsScreen(self)

    # --- main loop --------------------------------------------------
    def process_frame(self, dt):
        """Handle events, update and draw one frame."""
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False
            else:
                self.screen.handle_event(event)
        self.time += dt
        self.screen.update(dt)
        self.starfield.draw(self.surface, self.time)
        self.screen.draw(self.surface)
        self.pygame.display.flip()

    def run(self):
        while self.running:
            dt = min(0.1, self.clock.tick(config.FPS) / 1000.0)
            self.process_frame(dt)
        self.pygame.quit()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Space Explorer AI: Q-Learning vs SARSA.")
    parser.add_argument("--train", choices=("q_learning", "sarsa", "both"),
                        help="train from the command line without opening the window")
    parser.add_argument("--episodes", type=int, help="number of training episodes")
    parser.add_argument("--seed", type=int, help="random seed (default: from settings)")
    parser.add_argument("--multi-seed", type=int, metavar="N",
                        help="train both algorithms with N different seeds and save CSV tables")
    parser.add_argument("--compare", action="store_true",
                        help="save results/comparison.png from the saved models")
    return parser.parse_args(argv)


def build_settings(args):
    settings = storage.load_settings()
    if args.episodes is not None:
        settings.episodes = config.parse_field("episodes", args.episodes)
    if args.seed is not None:
        settings.seed = config.parse_field("seed", args.seed)
    return settings


def run_command_line(args):
    """Window-free commands. Returns the process exit code."""
    try:
        settings = build_settings(args)
    except ValueError as error:
        print(f"Error: {error}")
        return 2

    if args.train:
        from training.trainer import train_and_save
        algorithms = config.ALGORITHM_KEYS if args.train == "both" else (args.train,)
        seed = 42 if settings.seed == -1 else settings.seed
        for algorithm in algorithms:
            name = config.ALGORITHM_NAMES[algorithm]

            def progress(snapshot, name=name):
                print(f"\r{name}: episode {snapshot['episode']}/{snapshot['total']} "
                      f"success {snapshot['success_rate']:.0f}%", end="", flush=True)

            result, evaluation = train_and_save(algorithm, settings, seed=seed,
                                                progress_callback=progress)
            from evaluation.evaluate_agents import format_steps
            print(f"\n{name} finished in {result.duration:.1f}s. Evaluation: "
                  f"mean reward {evaluation['mean_reward']:.1f}, "
                  f"success {evaluation['success_rate']:.1f}%, "
                  f"steps/success {format_steps(evaluation['mean_steps_success'])}, "
                  f"collisions {evaluation['collision_rate']:.1f}%")
        print("Saved models to models/ and CSV files to results/.")

    if args.multi_seed:
        from training.trainer import run_multi_seed_experiment
        base = 1 if settings.seed == -1 else settings.seed
        seeds = [base + i for i in range(args.multi_seed)]
        print(f"Multi-seed experiment with seeds {seeds} ({settings.episodes} episodes each)...")
        _, summary = run_multi_seed_experiment(
            settings, seeds, progress_callback=lambda a, s: print(f"  done: {a} seed {s}"))
        for row in summary:
            print(row)
        print("Saved results/multi_seed_results.csv and results/multi_seed_summary.csv")

    if args.compare:
        from environment import SpaceEnvironment
        from evaluation import plots
        from evaluation.evaluate_agents import evaluate_q_table
        env = SpaceEnvironment()
        data = {}
        for algorithm in config.ALGORITHM_KEYS:
            try:
                model = storage.load_model(algorithm, env.signature(), env.n_states, env.n_actions)
            except storage.ModelError as error:
                print(error)
                continue
            data[algorithm] = {"history": model.history,
                               "evaluation": evaluate_q_table(model.q_table, env),
                               "episodes": int(model.meta.get("episodes", 0))}
        if not data:
            print("Nothing to compare. Train first, e.g.:  python main.py --train both")
            return 1
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = plots.save_comparison_png(data, config.RESULTS_DIR / "comparison.png")
        print(f"Saved {path}")
    return 0


def main(argv=None):
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    if args.train or args.multi_seed or args.compare:
        return run_command_line(args)
    try:
        import pygame  # noqa: F401
    except ImportError:
        print("Pygame is not installed. Run:  python -m pip install -r requirements.txt")
        return 1
    App().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
