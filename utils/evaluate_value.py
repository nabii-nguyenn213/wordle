import numpy as np
from ai.rl.environment import WordleEnv
from ai.rl.agents import EntropyAgent
from ai.rl.value import ValueEstimator

def evaluate_value(checkpoint_path="./results/wordle_value.pt", seed=42,
                   training_games=1000, test_games=100,max_guesses=20):
    env = WordleEnv(render_mode="train", seed=seed)
    agent = EntropyAgent(env=env, max_guesses=max_guesses)
    value_model = ValueEstimator(env=env, checkpoint_path=checkpoint_path)
    rng = np.random.default_rng(seed)
    words = list(env.words)
    rng.shuffle(words)
    if training_games >= len(words):
        raise ValueError("No held-out words available for evaluation")
    test_answers = words[training_games:training_games + test_games]
    predictions = []
    targets = []
    action_cache = {}
    wins = 0
    for answer in test_answers:
        observation = env.reset(answer=answer)
        done = False
        episode_states = []
        episode_rewards = []
        while not done:
            candidates = tuple(observation["candidates"])
            remaining = observation["guesses_remaining"]
            episode_states.append((candidates, remaining))
            key = (candidates, remaining)
            if key not in action_cache:
                action_cache[key] = agent.select_action(observation)
            guess = action_cache[key]
            observation, reward, done, info = env.step(guess)
            episode_rewards.append(reward)
        wins += int(info["win"])
        actual_cost = 0.0
        for state, reward in zip(reversed(episode_states), reversed(episode_rewards)):
            actual_cost -= reward
            candidates, remaining = state
            predicted_cost = value_model.predict_cost(candidates, remaining)
            predictions.append(predicted_cost)
            targets.append(actual_cost)
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    print("\n=== VALUE NETWORK EVALUATION ===")
    print(f"Test games: {len(test_answers)}")
    print(f"Test states: {len(targets)}")
    print(f"Win rate: {wins / len(test_answers):.2%}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("\nSample predictions:")
    for actual, predicted in list(zip(targets, predictions))[:10]:
        print(
            f"Actual: {actual:.2f} | "
            f"Predicted: {predicted:.2f}"
        )
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "wins": wins,
        "games": len(test_answers),
    }

if __name__ == "__main__":
    evaluate_value(
        checkpoint_path="./results/wordle_value.pt",
        seed=42,
        training_games=1000,
        test_games=100,
        max_guesses=20,
    )
