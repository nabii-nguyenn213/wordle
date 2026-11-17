import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FILES = {
    "Entropy DL": "entropy_student_history.csv",
    "Rollout DL": "rollout_student_history.csv",
    "MCTS DL": "mcts_student_history.csv",
}

METRICS = (
    "epoch",
    "train_ce",
    "val_ce",
    "train_sampled_top1",
    "val_sampled_top1",
    "val_sampled_top5",
    "val_sampled_mrr",
    "best_val_ce",
    "best_epoch",
    "learning_rate",
)


# ============================================================
# 1. LOAD TRAINING HISTORIES
# ============================================================

def load_histories(input_dir):
    histories = {}

    for name, filename in FILES.items():
        path = input_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing history file: {path}"
            )

        with path.open(
            newline="",
            encoding="utf-8-sig",
        ) as stream:
            reader = csv.DictReader(stream)

            missing = (
                set(METRICS)
                - set(reader.fieldnames or ())
            )

            if missing:
                raise ValueError(
                    f"{path}: missing columns {sorted(missing)}"
                )

            records = list(reader)

        if not records:
            raise ValueError(
                f"{path}: empty training history"
            )

        history = {
            key: np.asarray(
                [float(row[key]) for row in records]
            )
            for key in METRICS
        }

        if any(
            not np.isfinite(values).all()
            for values in history.values()
        ):
            raise ValueError(
                f"{path}: NaN or Infinity detected"
            )

        histories[name] = history

    return histories


def best_row(history):
    """Locate the minimum validation CE epoch."""
    return int(np.argmin(history["val_ce"]))


# ============================================================
# 2. PLOTTING UTILITIES
# ============================================================

def setup_axis(
    ax,
    title,
    ylabel,
    percentage=False,
    log_scale=False,
):
    ax.set_title(title, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.24)

    if percentage:
        ax.set_ylim(0, 102)

    if log_scale:
        ax.set_yscale("log")


def draw_metric(
    ax,
    histories,
    key,
    title,
    ylabel,
    percent=False,
    log_scale=False,
    show_best=False,
):
    for name, history in histories.items():
        epochs = history["epoch"]

        values = history[key] * (
            100 if percent else 1
        )

        line, = ax.plot(
            epochs,
            values,
            label=name,
            linewidth=1.9,
        )

        if show_best:
            i = best_row(history)

            ax.scatter(
                epochs[i],
                values[i],
                marker="*",
                s=125,
                color=line.get_color(),
                edgecolors="white",
                zorder=5,
            )

    setup_axis(
        ax,
        title,
        ylabel,
        percentage=percent,
        log_scale=log_scale,
    )

    ax.legend(
        fontsize=8,
        frameon=False,
    )


def save(fig, output_dir, filename, dpi):
    path = output_dir / filename

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)
    print("Saved:", path)


# ============================================================
# 3. COMPLETE DASHBOARD
# ============================================================

def plot_dashboard(histories, output_dir, dpi):
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(15, 13),
        constrained_layout=True,
    )

    specs = [
        (
            "train_ce",
            "Training Cross-Entropy",
            "Cross-Entropy",
            False, True, False,
        ),
        (
            "val_ce",
            "Validation Cross-Entropy (Stars = Best)",
            "Cross-Entropy",
            False, True, True,
        ),
        (
            "train_sampled_top1",
            "Training Sampled Top-1",
            "Agreement (%)",
            True, False, False,
        ),
        (
            "val_sampled_top1",
            "Validation Sampled Top-1",
            "Agreement (%)",
            True, False, False,
        ),
        (
            "val_sampled_top5",
            "Validation Sampled Top-5",
            "Agreement (%)",
            True, False, False,
        ),
        (
            "val_sampled_mrr",
            "Validation Sampled MRR",
            "Mean Reciprocal Rank",
            False, False, False,
        ),
    ]

    for ax, spec in zip(axes.flat, specs):
        (
            key,
            title,
            ylabel,
            percent,
            log_scale,
            show_best,
        ) = spec

        draw_metric(
            ax,
            histories,
            key,
            title,
            ylabel,
            percent=percent,
            log_scale=log_scale,
            show_best=show_best,
        )

    axes[2, 1].set_ylim(0, 1.02)

    fig.suptitle(
        "Wordle - Deep Learning Training Metrics",
        fontsize=17,
        weight="bold",
    )

    save(
        fig,
        output_dir,
        "training_dashboard.png",
        dpi,
    )


# ============================================================
# 4. CROSS-ENTROPY LOSS
# ============================================================

def plot_loss(histories, output_dir, dpi):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=True,
    )

    draw_metric(
        axes[0],
        histories,
        "train_ce",
        "Training Cross-Entropy",
        "Cross-Entropy (log scale)",
        log_scale=True,
    )

    draw_metric(
        axes[1],
        histories,
        "val_ce",
        "Validation Cross-Entropy",
        "Cross-Entropy (log scale)",
        log_scale=True,
        show_best=True,
    )

    save(
        fig,
        output_dir,
        "loss_curves.png",
        dpi,
    )


# ============================================================
# 5. TOP-1 AGREEMENT
# ============================================================

def plot_accuracy(histories, output_dir, dpi):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=True,
    )

    draw_metric(
        axes[0],
        histories,
        "train_sampled_top1",
        "Training Sampled Top-1",
        "Agreement (%)",
        percent=True,
    )

    draw_metric(
        axes[1],
        histories,
        "val_sampled_top1",
        "Validation Sampled Top-1",
        "Agreement (%)",
        percent=True,
    )

    save(
        fig,
        output_dir,
        "top1_agreement.png",
        dpi,
    )


# ============================================================
# 6. TOP-5 AND MRR
# ============================================================

def plot_ranking(histories, output_dir, dpi):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=True,
    )

    draw_metric(
        axes[0],
        histories,
        "val_sampled_top5",
        "Validation Sampled Top-5",
        "Agreement (%)",
        percent=True,
    )

    draw_metric(
        axes[1],
        histories,
        "val_sampled_mrr",
        "Validation Sampled MRR",
        "Mean Reciprocal Rank",
    )

    axes[1].set_ylim(0, 1.02)

    save(
        fig,
        output_dir,
        "top5_and_mrr.png",
        dpi,
    )


# ============================================================
# 7. GENERALIZATION GAP
# ============================================================

def plot_gap(histories, output_dir, dpi):
    fig, ax = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    for name, history in histories.items():
        gap = (
            history["val_ce"]
            - history["train_ce"]
        )

        ax.plot(
            history["epoch"],
            gap,
            linewidth=1.9,
            label=name,
        )

    ax.axhline(
        0,
        color="grey",
        linestyle="--",
        linewidth=1,
    )

    setup_axis(
        ax,
        "Cross-Entropy Generalization Gap",
        "Validation CE - Training CE",
    )

    ax.legend(frameon=False)

    save(
        fig,
        output_dir,
        "generalization_gap.png",
        dpi,
    )


# ============================================================
# 8. SAVE BEST MODEL METRICS
# ============================================================

def save_summary(histories, output_dir):
    rows = []

    for name, history in histories.items():
        i = best_row(history)

        rows.append({
            "model": name,
            "epochs_ran": len(history["epoch"]),
            "best_epoch": int(history["epoch"][i]),
            "best_val_ce": float(
                history["val_ce"][i]
            ),
            "train_ce_at_best": float(
                history["train_ce"][i]
            ),
            "val_sampled_top1_at_best": float(
                history["val_sampled_top1"][i]
            ),
            "val_sampled_top5_at_best": float(
                history["val_sampled_top5"][i]
            ),
            "val_sampled_mrr_at_best": float(
                history["val_sampled_mrr"][i]
            ),
            "learning_rate_at_best": float(
                history["learning_rate"][i]
            ),
        })

    path = output_dir / "metrics_summary.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nBEST-CHECKPOINT METRICS")

    for row in rows:
        print(
            f"{row['model']:<12} "
            f"epoch={row['best_epoch']:>3}  "
            f"val CE={row['best_val_ce']:.4f}  "
            f"Top-1={row['val_sampled_top1_at_best']:.2%}  "
            f"Top-5={row['val_sampled_top5_at_best']:.2%}  "
            f"MRR={row['val_sampled_mrr_at_best']:.4f}"
        )

    print("Saved:", path)


# ============================================================
# 9. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot Wordle DL training metrics"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./results/dl_checkpoints/"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results/dl_plots/"),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    histories = load_histories(args.input_dir)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    plot_dashboard(
        histories,
        args.output_dir,
        args.dpi,
    )

    plot_loss(
        histories,
        args.output_dir,
        args.dpi,
    )

    plot_accuracy(
        histories,
        args.output_dir,
        args.dpi,
    )

    plot_ranking(
        histories,
        args.output_dir,
        args.dpi,
    )

    plot_gap(
        histories,
        args.output_dir,
        args.dpi,
    )

    save_summary(
        histories,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
