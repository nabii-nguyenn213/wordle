import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from ai.rl.environment import WordleEnv
from ai.rl.agents import EntropyAgent, RollOutAgent
from ai.rl.mcts import MCTSAgent
from ai.rl.value import encode_state


def make_teacher(env, name, args):
    if name == "entropy":
        return EntropyAgent(
            env=env,
            max_guesses=args.max_guesses,
        )

    if name == "rollout":
        return RollOutAgent(
            env=env,
            max_guesses=args.max_guesses,
            max_rollout_candidates=args.rollout_threshold,
        )

    if name in ("mcts", "learned-mcts"):
        kwargs = dict(
            env=env,
            simulations=args.simulations,
            max_guesses=args.max_guesses,
            max_mcts_candidates=args.mcts_threshold,
            max_actions=args.max_actions,
            seed=args.seed,
        )

        if name == "learned-mcts":
            if not args.checkpoint:
                raise ValueError(
                    "--checkpoint is required for learned-mcts"
                )

            from ai.rl.value import ValueEstimator

            kwargs["value_model"] = ValueEstimator(
                env=env,
                checkpoint_path=args.checkpoint,
            )

        return MCTSAgent(**kwargs)

    raise ValueError(name)


def play_episodes(
    env,
    teacher,
    answers,
    collect_samples=False,
):
    word_to_id = {
        word: i for i, word in enumerate(env.words)
    }

    # Reuse teacher decisions on identical belief states.
    policy_cache = {}
    seen = set()

    samples = []
    logs = []
    methods = Counter()

    wins = 0
    lengths = []

    for episode, answer in enumerate(answers):
        obs = env.reset(answer=answer)

        done = False
        guessed = set()

        while not done:
            candidates = tuple(obs["candidates"])
            remaining = int(obs["guesses_remaining"])

            state_key = (candidates, remaining)

            if state_key not in policy_cache:
                action = teacher.select_action(obs)

                method = getattr(
                    teacher,
                    "last_decision",
                    {},
                ).get("method", "entropy")

                policy_cache[state_key] = (
                    action,
                    method,
                )

            action, method = policy_cache[state_key]

            if action not in word_to_id:
                raise ValueError(
                    f"Unknown teacher action: {action}"
                )

            if action in guessed:
                raise RuntimeError(
                    f"Teacher repeated guess {action}"
                )

            guessed.add(action)
            methods[method] += 1

            # Save one action label per distinct state.
            if collect_samples and state_key not in seen:
                seen.add(state_key)

                features = encode_state(
                    obs,
                    len(env.words),
                )

                samples.append((
                    features,
                    word_to_id[action],
                    remaining,
                    len(candidates),
                    method,
                ))

            next_obs, reward, done, info = env.step(
                action
            )

            # Ensure candidate filtering is correct.
            if answer not in next_obs["candidates"]:
                raise AssertionError(
                    "Actual answer eliminated from belief state"
                )

            if collect_samples:
                logs.append({
                    "episode": episode,
                    "answer_debug_only": answer,
                    "guess": action,
                    "feedback": "/".join(info["feedback"]),
                    "remaining_before": remaining,
                    "candidate_count_before": len(candidates),
                    "teacher_method": method,
                    "reward": float(reward),
                })

            obs = next_obs

        wins += int(info["win"])
        lengths.append(len(env.colors_hist))

        if (
            (episode + 1) % 25 == 0
            or episode + 1 == len(answers)
        ):
            print(
                f"{episode + 1}/{len(answers)} episodes",
                flush=True,
            )

    return {
        "samples": samples,
        "logs": logs,
        "wins": wins,
        "mean_guesses": float(np.mean(lengths)),
        "methods": dict(methods),
    }


def collect_dataset(args):
    if args.n_games < 10:
        raise ValueError(
            "Use at least 10 games for the dataset split"
        )

    env = WordleEnv(
        render_mode="train",
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)

    words = list(env.words)
    rng.shuffle(words)

    selected = words[:min(args.n_games, len(words))]

    if len(selected) < 10:
        raise ValueError("Not enough distinct answers")

    # 80% train, 10% validation, 10% test.
    n_train = int(0.8 * len(selected))
    n_val = int(0.1 * len(selected))

    train_answers = selected[:n_train]
    val_answers = selected[n_train:n_train + n_val]
    test_answers = selected[n_train + n_val:]

    teacher = make_teacher(
        env,
        args.teacher,
        args,
    )

    print(
        f"Teacher: {args.teacher} | "
        f"Train: {len(train_answers)} | "
        f"Validation: {len(val_answers)} | "
        f"Test: {len(test_answers)}"
    )

    # Collect training demonstrations.
    print("\nCollecting training demonstrations...")

    training = play_episodes(
        env,
        teacher,
        train_answers,
        collect_samples=True,
    )

    # Evaluate the teacher without collecting
    # validation labels for student training.
    print("\nEvaluating teacher on validation answers...")

    validation = play_episodes(
        env,
        teacher,
        val_answers,
        collect_samples=False,
    )

    # Save the training dataset.
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = out / (
        f"{args.teacher}_demonstrations.npz"
    )

    samples = training["samples"]

    np.savez_compressed(
        path,
        X=np.stack([
            sample[0] for sample in samples
        ]).astype(np.float32),

        y=np.asarray([
            sample[1] for sample in samples
        ], dtype=np.int64),

        remaining=np.asarray([
            sample[2] for sample in samples
        ], dtype=np.int8),

        candidate_count=np.asarray([
            sample[3] for sample in samples
        ], dtype=np.int32),

        method=np.asarray([
            sample[4] for sample in samples
        ]),

        vocabulary=np.asarray(env.words),
        train_answers=np.asarray(train_answers),
        val_answers=np.asarray(val_answers),
        test_answers=np.asarray(test_answers),
    )

    # Save gameplay for inspection.
    csv_path = out / (
        f"{args.teacher}_gameplay.csv"
    )

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(training["logs"][0]),
        )

        writer.writeheader()
        writer.writerows(training["logs"])

    print("\n=== DATA COLLECTION RESULTS ===")
    print("Saved samples:", len(samples))
    print("Dataset:", path)
    print("Gameplay log:", csv_path)

    print(
        f"Train win rate: "
        f"{training['wins'] / len(train_answers):.1%}"
    )

    print(
        f"Validation win rate: "
        f"{validation['wins'] / len(val_answers):.1%}"
    )

    print(
        f"Train average guesses: "
        f"{training['mean_guesses']:.3f}"
    )

    print(
        f"Validation average guesses: "
        f"{validation['mean_guesses']:.3f}"
    )

    print("Decision methods:", training["methods"])
    print("Test answers remain untouched.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--teacher",
        choices=[
            "entropy",
            "rollout",
            "mcts",
            "learned-mcts",
        ],
        required=True,
    )

    parser.add_argument(
        "--n-games",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        "--max-actions",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output-dir",
        default="./dataset/data_from_rl",
    )

    args = parser.parse_args()
    collect_dataset(args)


if __name__ == "__main__":
    main()
