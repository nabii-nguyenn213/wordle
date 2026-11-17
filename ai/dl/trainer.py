import argparse
import csv
from pathlib import Path
import numpy as np 
import torch 
from ai.dl.policy import DeepWordleAgent, StudentPolicy, dictionary_fingerprint, make_guess_features
from ai.rl.environment import WordleEnv
from ai.rl.value import FEATURE_DIM

def action_choices(labels, vocab_size, negatives, rng):
    if vocab_size <= 1:
        raise ValueError("Need at least 2 allowed guesses")
    other = rng.integers(
        0,
        vocab_size - 1,
        size=(len(labels), negatives),
        dtype=np.int64,
    )
    # Ensure no negative equals the positive action.
    other += other >= labels[:, None]
    return np.concatenate(
        (labels[:, None], other),
        axis=1,
    )


def batch_logits(model, states, action_ids, action_features):
    actions = action_features[action_ids]
    batch, options, guess_dim = actions.shape
    repeated_states = (
        states[:, None, :]
        .expand(-1, options, -1)
    )
    scores = model(
        repeated_states.reshape(batch * options, -1),
        actions.reshape(batch * options, guess_dim),
    )
    return scores.reshape(batch, options)

def evaluate_checkpoint(checkpoint, answers, max_games=20):
    env = WordleEnv(render_mode="train")
    student = DeepWordleAgent(
        env,
        checkpoint=str(checkpoint),
    )
    wins = 0
    turns = []
    for answer in answers[:max_games]:
        observation = env.reset(answer=str(answer))
        guessed = set()
        done = False
        while not done:
            guess = student.select_action(
                observation,
                exclude_guesses=guessed,
            )
            guessed.add(guess)

            observation, reward, done, info = env.step(
                guess
            )
        wins += int(info["win"])
        turns.append(len(env.colors_hist))
    if turns:
        print("\n=== HELD-OUT EVALUATION ===")
        print("Games:", len(turns))
        print(f"Win rate: {wins / len(turns):.2%}")
        print(f"Average guesses: {np.mean(turns):.3f}")


def train(
    input_path,
    output_path,
    epochs=150,
    batch_size=128,
    negatives=63,
    learning_rate=3e-4,
    weight_decay=1e-4,
    hidden=256,
    val_fraction=0.1,
    seed=42,
    eval_games=0,
    device="cpu",
    patience=15,
    min_delta=1e-4,
):
    """
    Offline policy distillation for Wordle.

    Trains StudentPolicy using teacher demonstrations.

    Metrics:
        - Training Cross Entropy
        - Validation Cross Entropy
        - Sampled Top-1 Accuracy
        - Sampled Top-5 Accuracy
        - Sampled Mean Reciprocal Rank

    Features:
        - Early stopping
        - Best checkpoint saving
        - CSV training history
        - Optional held-out gameplay evaluation
    """

    # ========================================================
    # 1. REPRODUCIBILITY
    # ========================================================

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rng = np.random.default_rng(seed)

    device = torch.device(device)

    if epochs < 1:
        raise ValueError("epochs must be positive")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if negatives < 1:
        raise ValueError("negatives must be positive")

    if patience < 1:
        raise ValueError("patience must be positive")

    if hidden < 2:
        raise ValueError("hidden must be at least 2")

    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")

    if min_delta < 0:
        raise ValueError("min_delta must be nonnegative")

    # ========================================================
    # 2. LOAD DATASET
    # ========================================================

    print("\n" + "=" * 60)
    print("LOADING TEACHER DATASET")
    print("=" * 60)

    with np.load(
        input_path,
        allow_pickle=False,
    ) as data:

        X = np.asarray(
            data["X"],
            dtype=np.float32,
        )

        y = np.asarray(
            data["y"],
            dtype=np.int64,
        )

        vocabulary = tuple(
            data["vocabulary"].astype(str)
        )

        test_answers = (
            data["test_answers"].astype(str).tolist()
            if "test_answers" in data.files
            else []
        )

    if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected features (N, {FEATURE_DIM}), "
            f"received {X.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            "Features and labels have different lengths"
        )

    if len(X) < 10:
        raise ValueError(
            "At least 10 demonstrations are required"
        )

    if len(vocabulary) < 2:
        raise ValueError(
            "Vocabulary must contain at least two words"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Dataset contains NaN or infinite features"
        )

    if not np.all(
        (y >= 0) & (y < len(vocabulary))
    ):
        raise ValueError(
            "Action labels contain invalid word indices"
        )

    vocab_size = len(vocabulary)

    print(f"Dataset        : {input_path}")
    print(f"Samples        : {len(X):,}")
    print(f"Vocabulary     : {vocab_size:,}")
    print(f"State features : {X.shape[1]}")
    print(f"Unique actions : {len(np.unique(y)):,}")

    # ========================================================
    # 3. TRAIN / VALIDATION SPLIT
    # ========================================================

    order = rng.permutation(len(y))

    n_val = max(
        1,
        int(len(y) * val_fraction),
    )

    val_ids = order[:n_val]
    train_ids = order[n_val:]

    X_train = torch.from_numpy(
        X[train_ids]
    ).to(device)

    X_val = torch.from_numpy(
        X[val_ids]
    ).to(device)

    y_train = y[train_ids]
    y_val = y[val_ids]

    print(f"Training states: {len(train_ids):,}")
    print(f"Validation states: {len(val_ids):,}")

    # ========================================================
    # 4. ACTION FEATURES
    # ========================================================

    action_features = torch.from_numpy(
        make_guess_features(vocabulary)
    ).to(device)

    # Fixed validation negatives:
    # every epoch evaluates on the same sampled choices.
    val_choices = torch.from_numpy(
        action_choices(
            y_val,
            vocab_size,
            negatives,
            rng,
        )
    ).to(device)

    # ========================================================
    # 5. INITIALIZE NEURAL NETWORK
    # ========================================================

    model = StudentPolicy(
        hidden=hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    checkpoint = Path(output_path)

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = checkpoint.with_name(
        f"{checkpoint.stem}_history.csv"
    )

    # ========================================================
    # 6. TRAINING CONFIGURATION
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    print(f"Device         : {device}")
    print(f"Architecture   : {FEATURE_DIM + action_features.shape[1]} -> "
          f"{hidden} -> {hidden // 2} -> 1")
    print(f"Optimizer      : AdamW")
    print(f"Learning rate  : {learning_rate}")
    print(f"Weight decay   : {weight_decay}")
    print(f"Batch size     : {batch_size}")
    print(f"Negative words : {negatives}")
    print(f"Maximum epochs : {epochs}")
    print(f"Early patience : {patience}")
    print(f"Minimum delta  : {min_delta}")
    print(f"Checkpoint     : {checkpoint}")
    print(f"History CSV    : {history_path}")

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history_rows = []

    # ========================================================
    # 7. TRAINING LOOP
    # ========================================================

    print("\n" + "=" * 60)
    print("START TRAINING")
    print("=" * 60)

    for epoch in range(1, epochs + 1):

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        model.train()

        shuffled = rng.permutation(
            len(y_train)
        )

        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for start in range(
            0,
            len(shuffled),
            batch_size,
        ):
            ids = shuffled[
                start:start + batch_size
            ]

            ids_tensor = torch.as_tensor(
                ids,
                dtype=torch.long,
                device=device,
            )

            states = X_train[ids_tensor]

            # Create positive + negative actions
            choices = action_choices(
                y_train[ids],
                vocab_size,
                negatives,
                rng,
            )

            action_ids = torch.from_numpy(
                choices
            ).to(device)

            logits = batch_logits(
                model,
                states,
                action_ids,
                action_features,
            )

            # Positive teacher action is always column 0.
            targets = torch.zeros(
                len(ids),
                dtype=torch.long,
                device=device,
            )

            loss = torch.nn.functional.cross_entropy(
                logits,
                targets,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            # Record training metrics
            batch_size_actual = len(ids)

            train_loss_sum += (
                loss.item() * batch_size_actual
            )

            train_correct += (
                logits.argmax(dim=1) == 0
            ).sum().item()

            train_samples += batch_size_actual

        train_loss = (
            train_loss_sum / train_samples
        )

        train_accuracy = (
            train_correct / train_samples
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss_sum = 0.0
        val_top1_correct = 0
        val_top5_correct = 0
        val_mrr_sum = 0.0
        val_samples = 0

        with torch.no_grad():

            for start in range(
                0,
                len(y_val),
                batch_size,
            ):
                end = min(
                    start + batch_size,
                    len(y_val),
                )

                states = X_val[start:end]

                selected_choices = val_choices[
                    start:end
                ]

                logits = batch_logits(
                    model,
                    states,
                    selected_choices,
                    action_features,
                )

                targets = torch.zeros(
                    end - start,
                    dtype=torch.long,
                    device=device,
                )

                loss = torch.nn.functional.cross_entropy(
                    logits,
                    targets,
                )

                current_batch = end - start

                val_loss_sum += (
                    loss.item() * current_batch
                )

                # --------------------------------------------
                # TOP-1 ACCURACY
                # --------------------------------------------

                predictions = logits.argmax(
                    dim=1
                )

                val_top1_correct += (
                    predictions == 0
                ).sum().item()

                # --------------------------------------------
                # TOP-5 ACCURACY
                # --------------------------------------------

                k = min(
                    5,
                    logits.shape[1],
                )

                top_indices = logits.topk(
                    k=k,
                    dim=1,
                ).indices

                val_top5_correct += (
                    top_indices == 0
                ).any(dim=1).sum().item()

                # --------------------------------------------
                # MEAN RECIPROCAL RANK
                # --------------------------------------------

                # Rank of the teacher action (column 0).
                teacher_scores = logits[:, :1]

                ranks = (
                    (logits > teacher_scores)
                    .sum(dim=1)
                    + 1
                )

                val_mrr_sum += (
                    1.0 / ranks.float()
                ).sum().item()

                val_samples += current_batch

        # ----------------------------------------------------
        # FINAL EPOCH METRICS
        # ----------------------------------------------------

        val_loss = (
            val_loss_sum / val_samples
        )

        val_top1 = (
            val_top1_correct / val_samples
        )

        val_top5 = (
            val_top5_correct / val_samples
        )

        val_mrr = (
            val_mrr_sum / val_samples
        )

        # ====================================================
        # 8. BEST CHECKPOINT + EARLY STOPPING
        # ====================================================

        improved = (
            val_loss < best_val_loss - min_delta
        )

        if improved:

            best_val_loss = val_loss
            best_epoch = epoch

            epochs_without_improvement = 0

            # Save the best-performing model.
            torch.save(
                {
                    "weights": {
                        key: value.detach().cpu()
                        for key, value
                        in model.state_dict().items()
                    },
                    "hidden": hidden,
                    "vocab_hash": dictionary_fingerprint(
                        vocabulary
                    ),
                    "source": str(input_path),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "train_size": len(train_ids),
                    "val_size": len(val_ids),
                },
                checkpoint,
            )

        else:
            epochs_without_improvement += 1

        # ====================================================
        # 9. SAVE TRAINING HISTORY
        # ====================================================

        history_rows.append({
            "epoch": epoch,

            "train_ce": train_loss,
            "val_ce": val_loss,

            "train_sampled_top1": train_accuracy,
            "val_sampled_top1": val_top1,
            "val_sampled_top5": val_top5,
            "val_sampled_mrr": val_mrr,

            "best_val_ce": best_val_loss,
            "best_epoch": best_epoch,

            "learning_rate": (
                optimizer.param_groups[0]["lr"]
            ),
        })

        # Save after every epoch.
        # This also preserves progress if training
        # is manually interrupted later.
        with history_path.open(
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    history_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(history_rows)

        # ====================================================
        # 10. PRINT TRAINING METRICS
        # ====================================================

        marker = " *" if improved else ""

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train CE={train_loss:.4f} | "
            f"Val CE={val_loss:.4f} | "
            f"Val Top1={val_top1:.2%} | "
            f"Val Top5={val_top5:.2%} | "
            f"Val MRR={val_mrr:.4f}"
            f"{marker}",
            flush=True,
        )

        # ====================================================
        # 11. EARLY STOPPING CHECK
        # ====================================================

        if epochs_without_improvement >= patience:

            print("\n" + "=" * 60)
            print("EARLY STOPPING TRIGGERED")
            print("=" * 60)

            print(f"Stopped at epoch : {epoch}")
            print(f"Best epoch       : {best_epoch}")
            print(f"Best val CE      : {best_val_loss:.4f}")

            break

    # ========================================================
    # 12. TRAINING SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(f"Total epochs   : {len(history_rows)}")
    print(f"Best epoch     : {best_epoch}")
    print(f"Best val CE    : {best_val_loss:.4f}")

    print(f"Best checkpoint: {checkpoint}")
    print(f"Training history: {history_path}")

    # ========================================================
    # 13. OPTIONAL GAMEPLAY EVALUATION
    # ========================================================

    if eval_games > 0:

        if not test_answers:
            print(
                "No test answers saved in the dataset. "
                "Skipping gameplay evaluation."
            )

        else:
            print("\nEvaluating best student checkpoint...")

            evaluate_checkpoint(
                checkpoint,
                test_answers,
                max_games=eval_games,
            )

    return str(checkpoint)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output", dest="output_path", required=True)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--negatives", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    train(**vars(parser.parse_args()))

if __name__ == "__main__":
    main()
