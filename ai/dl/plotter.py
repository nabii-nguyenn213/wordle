import argparse
import csv
from collections import Counter
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai.rl.environment import WordleEnv
from ai.dl.policy import DeepWordleAgent


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "Entropy DL": (
        "entropy_demonstrations.npz",
        "entropy_student.pt",
    ),
    "Rollout DL": (
        "rollout_demonstrations.npz",
        "rollout_student.pt",
    ),
    "MCTS DL": (
        "learned-mcts_demonstrations.npz",
        "mcts_student.pt",
    ),
}


# ============================================================
# 1. LOAD COLLECTED TEACHER DATASETS
# ============================================================

def load_datasets(data_dir):
    datasets = {}
    excluded = set()
    vocabulary = None

    for name, (dataset_name, _) in CONFIG.items():
        path = data_dir / dataset_name

        if not path.exists():
            print(f"[Missing dataset] {path}")
            continue

        with np.load(path, allow_pickle=False) as data:
            words = tuple(
                data["vocabulary"].astype(str)
            )

            if vocabulary is None:
                vocabulary = words

            elif vocabulary != words:
                raise ValueError(
                    "Teacher datasets use different "
                    "word lists or ordering."
                )

            # Exclude every answer used by any teacher.
            for key in (
                "train_answers",
                "val_answers",
                "test_answers",
            ):
                excluded.update(
                    data[key].astype(str).tolist()
                )

            methods = Counter(
                data["method"].astype(str).tolist()
            )

            datasets[name] = {
                "count": len(data["y"]),
                "methods": methods,
                "candidate_count": (
                    data["candidate_count"].copy()
                ),
                "remaining": (
                    data["remaining"].copy()
                ),
            }

    return datasets, excluded, vocabulary


# ============================================================
# 2. PLOT DATASET STATISTICS
# ============================================================

def plot_dataset_summary(datasets, output_dir):
    if not datasets:
        return

    names = list(datasets)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        constrained_layout=True,
    )

    # Number of demonstrations
    amounts = [
        datasets[name]["count"]
        for name in names
    ]

    bars = axes[0].bar(names, amounts)

    axes[0].bar_label(
        bars,
        padding=3,
    )

    axes[0].set_title(
        "Unique State-Action Demonstrations"
    )
    axes[0].set_ylabel(
        "Number of samples"
    )
    axes[0].tick_params(
        axis="x",
        rotation=15,
    )

    # Teacher decision methods
    all_methods = sorted(
        set().union(
            *(
                data["methods"]
                for data in datasets.values()
            )
        )
    )

    bottoms = np.zeros(len(names))

    for method in all_methods:
        values = np.array([
            datasets[name]["methods"].get(
                method, 0
            )
            for name in names
        ])

        axes[1].bar(
            names,
            values,
            bottom=bottoms,
            label=method,
        )

        bottoms += values

    axes[1].set_title(
        "Recorded Teacher Decision Methods"
    )
    axes[1].set_ylabel(
        "Number of unique states"
    )
    axes[1].tick_params(
        axis="x",
        rotation=15,
    )

    if all_methods:
        axes[1].legend(fontsize=8)

    fig.savefig(
        output_dir / "teacher_datasets.png",
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# 3. PLOT TRAINING AND VALIDATION CURVES
# ============================================================

def plot_learning_curves(models_dir, output_dir):
    histories = {}

    for name, (_, filename) in CONFIG.items():
        history_path = (
            models_dir
            / filename.replace(
                ".pt",
                "_history.csv",
            )
        )

        if not history_path.exists():
            print(
                f"[No history] {history_path}"
            )
            continue

        with history_path.open(newline="") as file:
            rows = list(csv.DictReader(file))

        if rows:
            histories[name] = rows

    if not histories:
        print(
            "No training histories found. "
            "Skipping training curves."
        )
        return

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        constrained_layout=True,
    )

    for name, rows in histories.items():
        epochs = [
            int(row["epoch"])
            for row in rows
        ]

        train_loss = [
            float(row["train_ce"])
            for row in rows
        ]

        val_loss = [
            float(row["val_ce"])
            for row in rows
        ]

        val_accuracy = [
            100 * float(
                row["val_sampled_top1"]
            )
            for row in rows
        ]

        axes[0].plot(
            epochs,
            train_loss,
            label=f"{name} Train",
        )

        axes[0].plot(
            epochs,
            val_loss,
            linestyle="--",
            label=f"{name} Val",
        )

        axes[1].plot(
            epochs,
            val_accuracy,
            label=name,
        )

    axes[0].set_title(
        "Policy Imitation Loss"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].legend(fontsize=8)

    axes[1].set_title(
        "Validation Agreement (Sampled Actions)"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(
        "Teacher Action Top-1 (%)"
    )
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8)

    fig.savefig(
        output_dir / "training_curves.png",
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# 4. CREATE A COMMON UNSEEN TEST SET
# ============================================================

def common_test_answers(
    env,
    excluded,
    games,
    seed,
):
    # No hidden answer from any teacher dataset.
    eligible = sorted(
        set(env.words) - excluded
    )

    if not eligible:
        raise ValueError(
            "No unused answers available "
            "for shared evaluation."
        )

    rng = np.random.default_rng(seed)

    count = min(
        games,
        len(eligible),
    )

    if count < games:
        print(
            f"[Warning] Only {count} "
            "unseen answers available."
        )

    selected = rng.choice(
        eligible,
        size=count,
        replace=False,
    ).tolist()

    print(
        f"Shared test answers: {len(selected)} "
        f"(excluded {len(excluded)} "
        "previously recorded answers)"
    )

    return selected


# ============================================================
# 5. EVALUATE THE THREE STUDENT AGENTS
# ============================================================

def evaluate_students(
    env,
    models_dir,
    answers,
):
    logs = []

    for name, (_, filename) in CONFIG.items():
        checkpoint = models_dir / filename

        if not checkpoint.exists():
            print(
                f"[Missing model] {checkpoint}"
            )
            continue

        student = DeepWordleAgent(
            env,
            checkpoint=str(checkpoint),
        )

        print(
            f"\nEvaluating {name} "
            f"on {len(answers)} answers...",
            flush=True,
        )

        for index, answer in enumerate(
            answers, 1
        ):
            observation = env.reset(
                answer=answer
            )

            guessed = set()
            timings = []
            done = False

            while not done:
                start = perf_counter()

                action = student.select_action(
                    observation,
                    exclude_guesses=guessed,
                )

                elapsed_ms = (
                    perf_counter() - start
                ) * 1000

                timings.append(elapsed_ms)
                guessed.add(action)

                (
                    observation,
                    reward,
                    done,
                    info,
                ) = env.step(action)

            win = bool(info["win"])

            guesses = len(
                env.colors_hist
            )

            logs.append({
                "model": name,
                "answer": answer,
                "win": int(win),
                "guesses": guesses,
                "penalized_guesses": (
                    guesses if win else 7
                ),
                "mean_decision_ms": float(
                    np.mean(timings)
                ),
                "total_decision_ms": float(
                    np.sum(timings)
                ),
                "decisions": len(timings),
            })

            if (
                index % 50 == 0
                or index == len(answers)
            ):
                print(
                    f"  {index}/{len(answers)}",
                    flush=True,
                )

    return logs


# ============================================================
# 6. COMPUTE GAMEPLAY METRICS
# ============================================================

def summarize(logs):
    rows = []

    for model in CONFIG:
        games = [
            game for game in logs
            if game["model"] == model
        ]

        if not games:
            continue

        won = [
            game for game in games
            if game["win"]
        ]

        win_rate = (
            len(won) / len(games)
        )

        avg_winning_guesses = (
            float(np.mean([
                game["guesses"]
                for game in won
            ]))
            if won else float("nan")
        )

        avg_penalty = float(
            np.mean([
                game["penalized_guesses"]
                for game in games
            ])
        )

        avg_latency = (
            sum(
                game["total_decision_ms"]
                for game in games
            )
            / sum(
                game["decisions"]
                for game in games
            )
        )

        rows.append({
            "model": model,
            "games": len(games),
            "wins": len(won),
            "win_rate": win_rate,
            "avg_guesses_on_wins": (
                avg_winning_guesses
            ),
            "avg_penalized_guesses": (
                avg_penalty
            ),
            "avg_decision_ms": avg_latency,
        })

    return rows


def save_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 7. PLOT GAMEPLAY BENCHMARK
# ============================================================

def plot_gameplay(
    summary,
    logs,
    output_dir,
):
    if not summary:
        return

    names = [
        row["model"]
        for row in summary
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        constrained_layout=True,
    )

    metrics = [
        (
            "win_rate",
            "Win Rate",
            "Percent",
            100,
        ),
        (
            "avg_guesses_on_wins",
            "Guesses on Successful Games",
            "Guesses (lower is better)",
            1,
        ),
        (
            "avg_penalized_guesses",
            "Penalized Guesses (Failure = 7)",
            "Score (lower is better)",
            1,
        ),
        (
            "avg_decision_ms",
            "Average Decision Time",
            "Milliseconds (lower is better)",
            1,
        ),
    ]

    for ax, (
        field,
        title,
        ylabel,
        multiplier,
    ) in zip(axes.flat, metrics):

        values = [
            row[field] * multiplier
            for row in summary
        ]

        bars = ax.bar(
            names,
            values,
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.tick_params(
            axis="x",
            rotation=10,
        )

        labels = [
            f"{value:.1f}"
            if np.isfinite(value)
            else "N/A"
            for value in values
        ]

        ax.bar_label(
            bars,
            labels=labels,
            padding=3,
        )

    fig.suptitle(
        "Wordle Students: Common Unseen Test Set"
    )

    fig.savefig(
        output_dir / "student_benchmark.png",
        dpi=180,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Guess distribution
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 5),
        constrained_layout=True,
    )

    x = np.arange(1, 8)
    bar_width = 0.8 / len(names)

    for index, name in enumerate(names):
        selected = [
            row for row in logs
            if row["model"] == name
        ]

        counts = [
            sum(
                row["penalized_guesses"] == n
                for row in selected
            )
            for n in x
        ]

        positions = (
            x - 0.4
            + bar_width * (index + 0.5)
        )

        ax.bar(
            positions,
            counts,
            width=bar_width,
            label=name,
        )

    ax.set_xticks(
        x,
        ["1", "2", "3", "4", "5", "6", "Failure"],
    )

    ax.set_xlabel("Number of Guesses")
    ax.set_ylabel("Number of Games")
    ax.set_title(
        "Solve Attempts and Failures"
    )

    ax.legend()

    fig.savefig(
        output_dir / "guess_distribution.png",
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# 8. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot Wordle training and "
            "student evaluation results"
        )
    )

    parser.add_argument(
        "--data-dir",
        default="dataset/data_from_rl",
    )

    parser.add_argument(
        "--models-dir",
        default="results",
    )

    parser.add_argument(
        "--output-dir",
        default="results/dl_plots",
    )

    parser.add_argument(
        "--games",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help=(
            "Plot saved datasets and "
            "histories without playing games"
        ),
    )
    args = parser.parse_args()
    if args.games < 1:
        parser.error(
            "--games must be positive"
        )

    data_dir = Path(
        args.data_dir
    )

    models_dir = Path(
        args.models_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Teacher data statistics
    datasets, excluded, vocabulary = (
        load_datasets(data_dir)
    )

    plot_dataset_summary(
        datasets,
        output_dir,
    )

    # 2. Training histories, if saved
    plot_learning_curves(
        models_dir,
        output_dir,
    )

    # 3. Common held-out gameplay test
    if not args.skip_eval:
        if not datasets:
            raise FileNotFoundError(
                "No teacher datasets found. "
                "Cannot create unseen test set."
            )

        env = WordleEnv(
            render_mode="train",
            seed=args.seed,
        )

        if tuple(env.words) != vocabulary:
            raise ValueError(
                "Current environment vocabulary "
                "differs from teacher datasets."
            )

        answers = common_test_answers(
            env,
            excluded,
            args.games,
            args.seed,
        )

        logs = evaluate_students(
            env,
            models_dir,
            answers,
        )

        summary = summarize(logs)

        save_csv(
            output_dir / "gameplay_per_game.csv",
            logs,
        )

        save_csv(
            output_dir / "student_summary.csv",
            summary,
        )

        plot_gameplay(
            summary,
            logs,
            output_dir,
        )

        print("\n=== STUDENT PERFORMANCE ===")

        for row in summary:
            print(
                f"\n{row['model']}"
                f"\n  Win rate: "
                f"{row['win_rate']:.2%}"
                f"\n  Avg guesses on wins: "
                f"{row['avg_guesses_on_wins']:.3f}"
                f"\n  Penalized guesses: "
                f"{row['avg_penalized_guesses']:.3f}"
                f"\n  Decision latency: "
                f"{row['avg_decision_ms']:.2f} ms"
            )

    print(
        "\nPlots and CSVs saved to:",
        output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
