from pathlib import Path
import numpy as np
import torch
from torch import nn
from ai.rl.environment import WordleEnv
from ai.rl.agents import EntropyAgent
from ai.rl.value import ValueNet, encode_state

def collect_experience(env, agent, answers, action_cache):
    features = []
    targets = []
    wins = 0
    for i, answer in enumerate(answers):
        observation = env.reset(answer=answer)
        done = False
        episode_features = []
        episode_rewards = []
        while not done:
            episode_features.append(encode_state(observation, len(env.words)))
            key = (tuple(observation["candidates"]), observation["guesses_remaining"])
            if key not in action_cache:
                action_cache[key] = agent.select_action(observation)
            guess = action_cache[key]
            observation, reward, done, info = env.step(guess)
            episode_rewards.append(reward)
        wins += int(info["win"])
        future_cost = 0.0
        for state, reward in zip(reversed(episode_features), reversed(episode_rewards)):
            future_cost -= reward
            features.append(state)
            targets.append(future_cost)
        print(
            f"\rEpisodes: {i + 1}/{len(answers)}",
            end="",
            flush=True,
        )
    print()
    return (
        np.stack(features).astype(np.float32),
        np.array(targets, dtype=np.float32),
        wins / len(answers),
    )


def train(
    # Environment / Agent
    seed=42,
    render_mode="train",
    max_guesses=20,
    # Dataset
    n_games=60,
    train_ratio=0.8,
    # Training
    epochs=100,
    learning_rate=1e-3,
    weight_decay=0.0,
    optimizer_cls=torch.optim.Adam,
    criterion=None,
    # Logging / Saving
    log_interval=10,
    checkpoint_path=None,
):
    torch.manual_seed(seed)
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if epochs < 1 or log_interval < 1:
        raise ValueError("epochs and log_interval must be positive")
    env = WordleEnv(
        render_mode=render_mode,
        seed=seed,
    )
    agent = EntropyAgent(
        env=env,
        max_guesses=max_guesses,
    )
    rng = np.random.default_rng(seed)
    words = list(env.words)
    rng.shuffle(words)
    n_games = min(n_games, len(words))
    if n_games < 2:
        raise ValueError("At least two games are required")
    n_train = max(
        1,
        min(n_games - 1, int(train_ratio * n_games))
    )
    train_answers = words[:n_train]
    val_answers = words[n_train:n_games]
    action_cache = {}
    # Collect experience
    print("Collecting training experience...")
    X_train, y_train, train_win_rate = collect_experience(
        env, agent, train_answers, action_cache
    )
    print("Collecting validation experience...")
    X_val, y_val, val_win_rate = collect_experience(
        env, agent, val_answers, action_cache
    )
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print(f"Training win rate: {train_win_rate:.2%}")
    print(f"Validation win rate: {val_win_rate:.2%}")
    # Model
    model = ValueNet()
    optimizer = optimizer_cls(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    if criterion is None:
        criterion = nn.MSELoss()
    # Convert to tensors
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val)
    # Checkpoint
    if checkpoint_path is None:
        checkpoint = Path(__file__).with_name("value.pt")
    else:
        checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        prediction = model(X_train)
        loss = criterion(prediction, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            val_prediction = model(X_val)
            val_loss = criterion(val_prediction, y_val)
            val_mae = torch.mean(
                torch.abs(val_prediction - y_val)
            ).item()
        # Save best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), checkpoint)
        # Logging
        if epoch % log_interval == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train MSE: {loss.item():.4f} | "
                f"Val MSE: {val_loss.item():.4f} | "
                f"Val MAE: {val_mae:.4f}"
            )
    print(f"\nBest model saved to: {checkpoint}")
    return model

if __name__ == "__main__":
    train(
        seed=42,
        render_mode="train",
        max_guesses=30,
        n_games=1000,
        train_ratio=0.8,
        epochs=200,
        learning_rate=3e-4,
        weight_decay=1e-5,
        optimizer_cls=torch.optim.Adam,
        criterion=nn.MSELoss(),
        log_interval=10,
        checkpoint_path="./results/wordle_value.pt",
         )
