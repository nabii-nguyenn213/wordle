import numpy as np 
import torch 
import hashlib
from ai.rl.value import FEATURE_DIM, encode_state
from ai.rl.environment import WordleEnv

GUESS_DIM = 5 * 26 + 26 

def make_guess_features(words): 
    features = np.zeros((len(words), GUESS_DIM), dtype=np.float32)
    for row, word in enumerate(words): 
        if len(word) != 5: 
            raise ValueError(f"Invalid word: {word}")
        for position, letter in enumerate(word): 
            index = ord(letter) - ord("a")
            if not 0 <= index <= 26: 
                raise ValueError(f"Invalid letter: {letter}")
            features[row, position * 26 + index] = 1.0
            features[row, 130 + index] = 1.0
    return features

def dictionary_fingerprint(words): 
    return hashlib.sha256("\n".join(words).encode("utf-8")).hexdigest()

class StudentPolicy(torch.nn.Module): 
    def __init__(self, hidden=128): 
        super().__init__()
        self.scorer = torch.nn.Sequential(torch.nn.Linear(FEATURE_DIM+GUESS_DIM, hidden), 
                                          torch.nn.ReLU(), 
                                          torch.nn.Linear(hidden, hidden // 2), 
                                          torch.nn.ReLU(), 
                                          torch.nn.Linear(hidden//2, 1)
                                         )

    def forward(self, state_features, guess_features): 
        x = torch.cat([state_features, guess_features], dim=-1)
        return self.scorer(x).squeeze(-1)

class DeepWordleAgent: 
    def __init__(self, env: WordleEnv, checkpoint="./results/dl/student.pt", batch_size=2048): 
        self.env = env 
        self.words = tuple(env.words)
        self.batch_size = batch_size

        if batch_size < 1: 
            raise ValueError("batch_size must be positive")
        # Load checkpoint 
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected_hash = dictionary_fingerprint(self.words)
        if saved["vocab_hash"] != expected_hash: 
            raise ValueError("Dictionary differs from training dictionary")
        # Initial the network 
        self.model = StudentPolicy(hidden=saved["hidden"])
        self.model.eval()

        self.action_features = torch.from_numpy(make_guess_features(self.words))
    
    @torch.no_grad()
    def top_k(self, observation, k=5, exclude_guesses=()): 
        candidates = observation["candidates"]
        if not candidates: 
            raise ValueError("No possible answers remaining")
        excluded = set(exclude_guesses)

        if len(candidates) == 1 : 
            if candidates[0] not in excluded: 
                return [(candidates[0], 0.0)]
            return []
        # Encode current state
        state_features = encode_state(observation, len(self.words))
        state = torch.from_numpy(state_features).float().unsqueeze(0)

        scores = []

        for start in range(0, len(self.words), self.batch_size): 
            action_batch = self.action_features[start:start+self.batch_size]
            state_batch = state.expand(len(action_batch), -1)
            batch_scores = self.model(state_batch, action_batch)
            scores.append(batch_scores)
        scores = torch.cat(scores).numpy()

        # Exclude already guessed words
        for index, word in enumerate(self.words):
            if word in excluded: 
                scores[index] = -np.inf 
        # Highest scores first 
        best_indices = np.argsort(-scores)[:k]
        return [(self.words[int(index)], float(scores[index])) for index in best_indices 
                if np.isfinite(scores[index])]

    def select_action(self, observation, exclude_guesses=()): 
        predictions = self.top_k(observation, k=1, exclude_guesses=exclude_guesses)
        if not predictions: 
            raise RuntimeError("No legal guess available")
        return predictions[0][0]
