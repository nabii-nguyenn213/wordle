import argparse
import curses
from collections import defaultdict
from pathlib import Path
from time import perf_counter
import numpy as np
from ai.rl.environment import WordleEnv
from ai.rl.agents import EntropyAgent, RollOutAgent
from ai.rl.mcts import MCTSAgent
from utils.utils import read_dataset

GREEN = "Green"
YELLOW = "Yellow"
GRAY = "Gray"
WIN = (GREEN,) * 5
TOP_N = 5

def load_priors(dataset_path, answers):
    words, probabilities = read_dataset(str(dataset_path))
    weights = defaultdict(float)
    if len(words) != len(probabilities):
        raise ValueError(
            "Dataset words and probabilities have different lengths"
        )
    for word, probability in zip(words, probabilities):
        probability = float(probability)
        if not np.isfinite(probability) or probability < 0:
            raise ValueError("Invalid probability")
        weights[str(word).strip().lower()] += probability
    result = {
        word: weights[word]
        for word in answers
    }
    total = sum(result.values())
    if total <= 0:
        return {
            word: 1.0 / len(answers)
            for word in answers
        }
    return {
        word: probability / total
        for word, probability in result.items()
    }

def posterior(candidates, priors):
    if not candidates:
        return {}
    total = sum(
        priors.get(word, 0.0)
        for word in candidates
    )
    if total <= 0:
        return {
            word: 1.0 / len(candidates)
            for word in candidates
        }
    return {
        word: priors.get(word, 0.0) / total
        for word in candidates
    }

def filter_candidates(env, history):
    candidates = list(env.words)
    for guess, colors in history:
        candidates = [
            answer
            for answer in candidates
            if tuple(
                env.simulate_feedback(guess, answer)
            ) == colors
        ]
    return candidates

def predict_entropy(agent, candidates, top_n=TOP_N):
    """Return top guesses ranked by information entropy."""

    if len(candidates) == 1:
        return [(candidates[0], 0.0)]

    guesses = agent.get_possible_guesses(candidates)

    results = []

    for guess in guesses:
        entropy = agent.calculate_entropy(
            guess,
            candidates
        )
        results.append((guess, entropy))

    results.sort(
        key=lambda x: (-x[1], x[0])
    )

    return results[:top_n]


def predict_rollout(
    agent,
    candidates,
    remaining,
    top_n=TOP_N
):
    """Return top guesses ranked by expected future cost."""

    if len(candidates) == 1:
        return "direct", [(candidates[0], 1.0)]

    # Use entropy for large candidate sets.
    if len(candidates) > agent.max_rollout_candidates:
        results = predict_entropy(
            agent.base_agent,
            candidates,
            top_n
        )
        return "entropy fallback", results

    observation = {
        "candidates": list(candidates),
        "guesses_remaining": remaining,
    }

    baseline = agent.base_agent.select_action(
        observation
    )

    guesses = list(dict.fromkeys(
        [baseline] +
        list(
            agent.base_agent.get_possible_guesses(
                candidates
            )
        )
    ))

    results = []

    for guess in guesses:
        cost = agent._action_cost(
            tuple(candidates),
            remaining,
            guess
        )

        results.append((guess, cost))

    results.sort(
        key=lambda x: (x[1], x[0])
    )

    return "rollout", results[:top_n]


def predict_mcts(
    agent,
    candidates,
    remaining,
    top_n=TOP_N
):
    """
    Run MCTS and return words ranked by search visits.

    Returns:
        mode, [(word, visit_fraction, estimated_cost), ...]
    """

    if len(candidates) == 1:
        return "direct", [
            (candidates[0], 1.0, 1.0)
        ]

    if len(candidates) > agent.max_mcts_candidates:
        results = predict_entropy(
            agent.base_agent,
            candidates,
            top_n
        )

        return "entropy fallback", [
            (word, 0.0, entropy)
            for word, entropy in results
        ]

    root = agent._new_node(
        tuple(candidates),
        remaining
    )

    for _ in range(agent.simulations):
        # Sample a hypothetical answer, not the real one.
        answer = agent.rng.choice(candidates)

        agent._simulate(root, answer)

    total_visits = sum(
        stats.visits
        for stats in root.moves.values()
    )

    results = []

    for guess, stats in root.moves.items():
        visits = (
            stats.visits / total_visits
            if total_visits else 0.0
        )

        expected_cost = (
            -stats.total_reward / stats.visits
            if stats.visits else float("inf")
        )

        results.append(
            (guess, visits, expected_cost)
        )

    results.sort(
        key=lambda x: (-x[1], x[2], x[0])
    )

    return "mcts", results[:top_n]


# ============================================================
# 3. LOAD ALL THREE AGENTS
# ============================================================

class WordleAdvisor:
    def __init__(
        self,
        dataset=None,
        checkpoint=None,
        max_guesses=20,
        rollout_threshold=12,
        mcts_threshold=10,
        simulations=200,
        max_actions=12,
        seed=42,
    ):
        if dataset is None:
            dataset = (
                Path(__file__).resolve().parents[1]
                / "dataset"
                / "wordle.csv"
            )

        self.dataset = Path(dataset)

        self.env = WordleEnv(
            dataset_path=str(self.dataset),
            render_mode="train",
            seed=seed,
        )

        self.priors = load_priors(
            self.dataset,
            self.env.words
        )

        # Model 1: Entropy
        self.entropy = EntropyAgent(
            env=self.env,
            max_guesses=max(max_guesses, TOP_N),
        )

        # Model 2: Rollout
        self.rollout = RollOutAgent(
            env=self.env,
            max_guesses=max(max_guesses, TOP_N),
            max_rollout_candidates=rollout_threshold,
        )

        # Optional trained value network
        value_model = None

        if checkpoint is not None:
            from ai.rl.value import ValueEstimator

            value_model = ValueEstimator(
                env=self.env,
                checkpoint_path=checkpoint,
            )

        # Model 3: MCTS
        self.mcts = MCTSAgent(
            env=self.env,
            simulations=simulations,
            max_guesses=max(max_guesses, TOP_N),
            max_actions=max(max_actions, TOP_N),
            max_mcts_candidates=mcts_threshold,
            seed=seed,
            value_model=value_model,
        )

        self.mcts_name = (
            "LEARNED MCTS"
            if value_model is not None
            else "MCTS"
        )

    def predict(self, history):
        """Compute predictions from all three agents."""

        candidates = filter_candidates(
            self.env,
            history
        )

        remaining = 6 - len(history)

        if not candidates:
            raise ValueError(
                "No possible answer matches your feedback."
            )

        # Shared answer probabilities
        probabilities = posterior(
            candidates,
            self.priors
        )

        top_answers = sorted(
            probabilities.items(),
            key=lambda x: (-x[1], x[0])
        )[:TOP_N]

        results = {
            "candidates": candidates,
            "remaining": remaining,
            "answers": top_answers,
            "models": [],
        }

        if remaining <= 0:
            return results

        if history and history[-1][1] == WIN:
            return results

        # ---------------- ENTROPY ----------------
        start = perf_counter()

        entropy_results = predict_entropy(
            self.entropy,
            candidates
        )

        results["models"].append({
            "name": "1. ENTROPY",
            "mode": "Information gain",
            "time": perf_counter() - start,
            "items": [
                (
                    word,
                    f"H={score:.3f} bits",
                    probabilities.get(word, 0.0)
                )
                for word, score in entropy_results
            ],
        })

        # ---------------- ROLLOUT ----------------
        start = perf_counter()

        mode, rollout_results = predict_rollout(
            self.rollout,
            candidates,
            remaining
        )

        rollout_items = []

        for word, score in rollout_results:
            if mode == "rollout":
                metric = f"Cost={score:.2f}"
            elif mode == "entropy fallback":
                metric = f"H={score:.3f}"
            else:
                metric = "Direct answer"

            rollout_items.append((
                word,
                metric,
                probabilities.get(word, 0.0)
            ))

        results["models"].append({
            "name": "2. ROLLOUT",
            "mode": mode,
            "time": perf_counter() - start,
            "items": rollout_items,
        })

        # ---------------- MCTS ----------------
        start = perf_counter()

        mode, mcts_results = predict_mcts(
            self.mcts,
            candidates,
            remaining
        )

        mcts_items = []

        for word, visits, cost in mcts_results:
            if mode == "mcts":
                metric = f"Visits={visits:.1%}"
            elif mode == "entropy fallback":
                metric = f"H={cost:.3f}"
            else:
                metric = "Direct answer"

            mcts_items.append((
                word,
                metric,
                probabilities.get(word, 0.0)
            ))

        results["models"].append({
            "name": f"3. {self.mcts_name}",
            "mode": mode,
            "time": perf_counter() - start,
            "items": mcts_items,
        })

        return results


# ============================================================
# 4. TERMINAL USER INTERFACE
# ============================================================

class WordleTUI:
    LEFT = 2
    RIGHT = 48
    BOARD_TOP = 3

    def __init__(self, advisor):
        self.advisor = advisor

        self.history = []

        self.letters = [""] * 5
        self.colors = [GRAY] * 5

        self.cursor = 0
        self.mode = "word"

        self.result = None
        self.finished = False

        self.status = "Enter a five-letter word."
        self.error = False

    # --------------------------------------------------------
    # Terminal colors
    # --------------------------------------------------------

    def setup_colors(self):
        curses.start_color()

        try:
            curses.use_default_colors()
        except curses.error:
            pass

        bg = -1

        curses.init_pair(
            1, curses.COLOR_BLACK, curses.COLOR_GREEN
        )
        curses.init_pair(
            2, curses.COLOR_BLACK, curses.COLOR_YELLOW
        )
        curses.init_pair(
            3, curses.COLOR_WHITE, curses.COLOR_BLACK
        )
        curses.init_pair(
            4, curses.COLOR_BLACK, curses.COLOR_CYAN
        )
        curses.init_pair(
            5, curses.COLOR_GREEN, bg
        )
        curses.init_pair(
            6, curses.COLOR_RED, bg
        )
        curses.init_pair(
            7, curses.COLOR_CYAN, bg
        )

    @staticmethod
    def put(win, y, x, text, style=0):
        height, width = win.getmaxyx()

        if y < 0 or y >= height:
            return

        if x < 0 or x >= width:
            return

        try:
            win.addnstr(
                y, x,
                str(text),
                max(0, width - x - 1),
                style
            )
        except curses.error:
            pass

    def tile_style(
        self,
        color,
        selected=False,
        submitted=False
    ):
        if selected:
            return (
                curses.color_pair(4)
                | curses.A_BOLD
            )

        styles = {
            GREEN: (
                curses.color_pair(1)
                | curses.A_BOLD
            ),
            YELLOW: (
                curses.color_pair(2)
                | curses.A_BOLD
            ),
            GRAY: (
                curses.color_pair(3)
                | (
                    0 if submitted
                    else curses.A_DIM
                )
            ),
        }

        return styles[color]

    # --------------------------------------------------------
    # Wordle board
    # --------------------------------------------------------

    def draw_board(self, win):
        x = self.LEFT

        title = [
            "+-----------------------+",
            "|         WORDLE        |",
            "+-----------------------+",
        ]

        for i, line in enumerate(title):
            self.put(
                win, i, x, line,
                curses.color_pair(5)
                | curses.A_BOLD
            )

        for row in range(6):
            top = self.BOARD_TOP + row * 3

            current = (
                row == len(self.history)
                and not self.finished
            )

            if row < len(self.history):
                word, color_tuple = self.history[row]

                letters = list(word)
                colors = list(color_tuple)

            elif current:
                letters = self.letters
                colors = self.colors

            else:
                letters = [""] * 5
                colors = [GRAY] * 5

            self.put(
                win, top, x,
                "+---+" * 5,
                curses.color_pair(5)
            )

            self.put(
                win, top + 2, x,
                "+---+" * 5,
                curses.color_pair(5)
            )

            for col in range(5):
                selected = (
                    current
                    and col == self.cursor
                )

                style = self.tile_style(
                    colors[col],
                    selected=selected,
                    submitted=row < len(self.history)
                )

                letter = (
                    letters[col].upper()
                    if letters[col] else " "
                )

                self.put(
                    win,
                    top + 1,
                    x + col * 5,
                    f"| {letter} |",
                    style
                )

        # Instructions
        y = self.BOARD_TOP + 19

        self.put(
            win, y, x,
            f"Guess {min(len(self.history) + 1, 6)}/6"
            f" | Mode: {self.mode.upper()}",
            curses.color_pair(7)
        )

        instructions = [
            "KEYBOARD CONTROLS",
            "A-Z       Enter word",
            "Left/Right Select tile",
            "G/Y/X     Set tile color",
            "Space     Cycle colors",
            "Enter     Submit feedback",
            "Tab       Switch mode",
            "Ctrl+U    Undo guess",
            "Ctrl+R    Reset game",
            "Esc       Quit",
        ]

        for i, line in enumerate(instructions):
            self.put(
                win, y + 2 + i, x, line
            )

        self.put(
            win, y + 13, x,
            self.status,
            curses.color_pair(6)
            if self.error
            else curses.color_pair(5)
        )

    # --------------------------------------------------------
    # Probability bars
    # --------------------------------------------------------

    @staticmethod
    def probability_bar(probability, width=11):
        probability = max(
            0.0,
            min(1.0, probability)
        )

        filled = round(probability * width)

        return (
            "█" * filled
            + "░" * (width - filled)
        )

    # --------------------------------------------------------
    # Prediction panel
    # --------------------------------------------------------

    def draw_predictions(self, win):
        x = self.RIGHT

        self.put(
            win, 1, x,
            "PREDICTED NEXT GUESSES",
            curses.color_pair(5)
            | curses.A_BOLD
        )

        if self.result is None:
            self.put(
                win, 3, x,
                "Enter a guess and press ENTER."
            )
            return

        result = self.result

        self.put(
            win, 2, x,
            f"Candidates: {len(result['candidates'])}"
            f" | Attempts left: {result['remaining']}"
        )

        if not result["models"]:
            self.put(
                win, 4, x,
                "Game finished. Ctrl+U to undo."
            )

        for index, model in enumerate(
            result["models"]
        ):
            y = 4 + index * 8

            self.put(
                win, y, x,
                model["name"],
                curses.color_pair(5)
                | curses.A_BOLD
            )

            self.put(
                win, y + 1, x,
                f"{model['mode']}"
                f" | {model['time']:.2f}s",
                curses.A_DIM
            )

            for rank, item in enumerate(
                model["items"], start=1
            ):
                word, metric, probability = item

                bar = self.probability_bar(
                    probability
                )

                text = (
                    f"{rank}. {word.upper():5s} "
                    f"[{bar}] "
                    f"P={probability:6.2%} "
                    f"{metric}"
                )

                self.put(
                    win, y + 1 + rank, x, text
                )

        # Shared answer probabilities
        y = 4 + 8 * len(result["models"]) + 1

        self.put(
            win, y, x,
            "MOST LIKELY ANSWERS (SHARED)",
            curses.color_pair(7)
            | curses.A_BOLD
        )

        for rank, (word, probability) in enumerate(
            result["answers"], start=1
        ):
            self.put(
                win, y + rank, x,
                f"{rank}. {word.upper():5s} "
                f"P(answer)={probability:6.2%}"
            )

        self.put(
            win, y + 7, x,
            "Bars = P(answer), not model confidence.",
            curses.A_DIM
        )

        if result["models"]:
            fallback = any(
                model["mode"] == "entropy fallback"
                for model in result["models"]
            )

            if fallback:
                self.put(
                    win, y + 8, x,
                    "Large belief: some agents use entropy.",
                    curses.A_DIM
                )

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------


    def draw(self, win):
        win.erase()

        height, width = win.getmaxyx()

        if width < 125 or height < 39:
            self.put(
                win, 0, 0,
                "Terminal too small."
            )
            self.put(
                win, 1, 0,
                "Resize to at least 125 x 39."
            )
            self.put(
                win, 2, 0,
                f"Current: {width} x {height}"
            )
            win.refresh()
            return

        # Dynamically position the right panel
        self.RIGHT = max(44, width // 3)

        self.draw_board(win)
        self.draw_predictions(win)

        win.refresh()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    def predict(self, win):
        self.status = "Computing predictions..."
        self.error = False

        self.draw(win)

        try:
            self.result = self.advisor.predict(
                self.history
            )

            self.status = "Ready for next guess."
            self.error = False

        except Exception as exc:
            self.result = None
            self.status = f"Error: {exc}"
            self.error = True

    # --------------------------------------------------------
    # Submit guess
    # --------------------------------------------------------

    def submit(self, win):
        if self.finished or len(self.history) >= 6:
            self.status = "Game already finished."
            self.error = True
            return

        if not all(self.letters):
            self.status = "Enter all five letters."
            self.error = True
            return

        guess = "".join(self.letters)
        colors = tuple(self.colors)

        trial_history = (
            self.history + [(guess, colors)]
        )

        candidates = filter_candidates(
            self.advisor.env,
            trial_history
        )

        if not candidates:
            self.status = (
                "No matching answer. "
                "Check your feedback."
            )
            self.error = True
            return

        # Save submitted guess
        self.history = trial_history

        self.finished = (colors == WIN)

        # Reset current input
        self.letters = [""] * 5
        self.colors = [GRAY] * 5
        self.cursor = 0
        self.mode = "word"

        self.predict(win)

        if self.finished:
            self.status = (
                f"Solved {guess.upper()} "
                f"in {len(self.history)} guesses!"
            )

        elif len(self.history) == 6:
            self.status = "Six guesses reached."

    # --------------------------------------------------------
    # Undo / Reset
    # --------------------------------------------------------

    def undo(self, win):
        if not self.history:
            self.status = "Nothing to undo."
            return

        self.history.pop()

        self.finished = False
        self.letters = [""] * 5
        self.colors = [GRAY] * 5
        self.cursor = 0
        self.mode = "word"

        self.predict(win)
        self.status = "Previous guess removed."

    def reset(self, win):
        self.history = []
        self.finished = False

        self.letters = [""] * 5
        self.colors = [GRAY] * 5

        self.cursor = 0
        self.mode = "word"

        self.predict(win)
        self.status = "New game started."

    # --------------------------------------------------------
    # Keyboard input
    # --------------------------------------------------------

    def handle_key(self, key, win):
        # Exit
        if key in (27, 3):
            return False

        # Ctrl+U
        if key == 21:
            self.undo(win)
            return True

        # Ctrl+R
        if key == 18:
            self.reset(win)
            return True

        # Arrow left
        if key == curses.KEY_LEFT:
            self.cursor = max(
                0, self.cursor - 1
            )
            return True

        # Arrow right
        if key == curses.KEY_RIGHT:
            self.cursor = min(
                4, self.cursor + 1
            )
            return True

        # TAB
        if key == 9:
            if self.mode == "color":
                self.mode = "word"

            elif all(self.letters):
                self.mode = "color"

            else:
                self.status = (
                    "Complete the word first."
                )
                self.error = True

            return True

        # Backspace
        if key in (
            curses.KEY_BACKSPACE,
            127,
            8
        ):
            if self.mode == "color":
                self.mode = "word"
                self.cursor = 4

            else:
                self.letters[self.cursor] = ""

                if self.cursor > 0:
                    self.cursor -= 1

            return True

        # ENTER
        if key in (
            10,
            13,
            curses.KEY_ENTER
        ):
            self.submit(win)
            return True

        if self.finished or len(self.history) >= 6:
            return True

        # Word input
        if self.mode == "word":
            if (
                65 <= key <= 90
                or 97 <= key <= 122
            ):
                self.letters[self.cursor] = (
                    chr(key).lower()
                )

                blanks = [
                    i for i, letter
                    in enumerate(self.letters)
                    if not letter
                ]

                if not blanks:
                    self.mode = "color"
                    self.cursor = 0

                    self.status = (
                        "Set colors: G/Y/X. "
                        "Press ENTER to submit."
                    )
                    self.error = False

                else:
                    later = [
                        i for i in blanks
                        if i > self.cursor
                    ]

                    self.cursor = (
                        later[0] if later
                        else blanks[0]
                    )

                return True

        # Color input
        if self.mode == "color":
            if key in (
                ord("g"),
                ord("y"),
                ord("x"),
                ord("0"),
                ord(" ")
            ):
                if key == ord("g"):
                    self.colors[self.cursor] = GREEN

                elif key == ord("y"):
                    self.colors[self.cursor] = YELLOW

                elif key in (
                    ord("x"),
                    ord("0")
                ):
                    self.colors[self.cursor] = GRAY

                else:
                    cycle = {
                        GRAY: YELLOW,
                        YELLOW: GREEN,
                        GREEN: GRAY,
                    }

                    old = self.colors[self.cursor]
                    self.colors[self.cursor] = cycle[old]

                self.cursor = (
                    self.cursor + 1
                ) % 5

                self.error = False

        return True

    # --------------------------------------------------------
    # Main UI loop
    # --------------------------------------------------------

    def run(self, win):
        self.setup_colors()

        try:
            curses.curs_set(0)
        except curses.error:
            pass

        win.keypad(True)

        self.predict(win)

        while True:
            self.draw(win)

            key = win.getch()

            if not self.handle_key(key, win):
                break


# ============================================================
# 5. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Wordle AI Terminal Solver"
    )

    parser.add_argument(
        "--dataset",
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional trained value.pt for Learned MCTS",
    )

    parser.add_argument(
        "--max-guesses",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--rollout-threshold",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--mcts-threshold",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()
    advisor = WordleAdvisor(
        **vars(args)
    )
    curses.wrapper(
        WordleTUI(advisor).run
    )

if __name__ == "__main__":
    main()
