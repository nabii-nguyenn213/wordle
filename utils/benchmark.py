from time import perf_counter
from statistics import mean
import numpy as np
from ai.rl.environment import WordleEnv
from ai.rl.agents import EntropyAgent, RollOutAgent

def evaluate(agent_class, answers, **kwargs):
    env = WordleEnv(render_mode="train")
    agent = agent_class(env=env, **kwargs)
    wins = 0
    guesses = []
    winning_guesses = []
    planning_times = []
    rollout_moves = 0
    for answer in answers:
        observation = env.reset(answer=answer)
        done = False
        while not done:
            start = perf_counter()
            guess = agent.select_action(observation)
            planning_times.append(perf_counter() - start)
            if hasattr(agent, "last_decision"):
                if agent.last_decision.get("method") == "rollout":
                    rollout_moves += 1
            observation, reward, done, info = env.step(guess)
            # A correct belief state must contain the answer.
            assert answer in observation["candidates"]
        n_guesses = len(env.colors_hist)
        guesses.append(n_guesses)
        if info["win"]:
            wins += 1
            winning_guesses.append(n_guesses)
    return {
        "games": len(answers),
        "wins": wins,
        "win_rate": wins / len(answers),
        "avg_guesses": mean(guesses),
        "avg_guesses_won": (
            mean(winning_guesses)
            if winning_guesses else float("nan")
        ),
        "avg_planning_time": mean(planning_times),
        "rollout_moves": rollout_moves,
    }


def benchmark(n_games=20, seed=42):
    env = WordleEnv(render_mode="train")
    rng = np.random.default_rng(seed)
    answers = rng.choice(
        env.words,
        size=min(n_games, len(env.words)),
        replace=False,
    ).tolist()
    results = {}
    print("Evaluating Entropy Agent...", flush=True)
    results["Entropy"] = evaluate(
        EntropyAgent,
        answers,
        max_guesses=20,
    )
    print("Evaluating Rollout Agent...", flush=True)
    results["Rollout"] = evaluate(RollOutAgent, answers, max_guesses=20, max_rollout_candidates=12)
    print("\n=== BENCHMARK RESULTS ===")
    for name, result in results.items():
        print(f"\n{name}")
        print(f"Games: {result['games']}")
        print(f"Wins: {result['wins']}")
        print(f"Win rate: {result['win_rate']:.2%}")
        print(f"Average guesses: {result['avg_guesses']:.3f}")
        print(
            "Average guesses (wins):",
            f"{result['avg_guesses_won']:.3f}",
        )
        print(
            "Average planning time:",
            f"{result['avg_planning_time']:.3f}s",
        )
        print("Rollout moves:", result["rollout_moves"])

if __name__ == "__main__":
    benchmark(n_games=20)
