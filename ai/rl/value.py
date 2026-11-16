import numpy as np
import torch
from torch import nn

FEATURE_DIM = 158

def encode_state(observation, dictionary_size):
    candidates = observation["candidates"]
    remaining = observation["guesses_remaining"]
    if not candidates:
        raise ValueError("Empty belief state")
    n = len(candidates)
    # Letter frequencies at each of the five positions
    positions = np.zeros((5, 26), dtype=np.float32)
    # Probability that each letter occurs in a word
    presence = np.zeros(26, dtype=np.float32)
    for word in candidates:
        for i, letter in enumerate(word):
            positions[i, ord(letter) - ord("a")] += 1
        for letter in set(word):
            presence[ord(letter) - ord("a")] += 1
    positions /= n
    presence /= n
    extra = np.array([np.log1p(n) / np.log1p(dictionary_size), remaining / 6.0], dtype=np.float32)
    return np.concatenate([positions.flatten(), presence, extra])

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class ValueEstimator:
    def __init__(self, env, checkpoint_path):
        self.env = env
        self.model = ValueNet()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.model.eval()

    @torch.no_grad()
    def predict_cost(self, candidates, remaining):
        observation = {
            "candidates": list(candidates),
            "guesses_remaining": remaining,
        }
        features = encode_state(observation, len(self.env.words))
        x = torch.from_numpy(features).unsqueeze(0)
        prediction = self.model(x).item()
        return float(np.clip(prediction, 1.0, remaining + 6.0))
