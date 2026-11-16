import string 
from collections import Counter
from games.wordle import Wordle
import numpy as np

class WordleEnv(Wordle):
    def __init__(self, dataset_path="./dataset/wordle.csv", seed=None, render_mode="train"): 
        super().__init__(mode="ai", render_mode=render_mode, 
                         dataset_path=dataset_path, seed=seed)
        # Normalize Dictionary
        self.words = tuple(sorted({str(word).strip().lower() for word in self.words
                                   if len(str(word).strip())==5 
                                   and str(word).strip().isascii()
                                   and str(word).strip().isalpha()}))
        if not self.words: 
            raise ValueError("Word Dictionary is empty")
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self, seed=None, answer=None): 
        if seed is not None: 
            self.rng = np.random.default_rng(seed)
        if answer is None: 
            self.given_word = str(self.rng.choice(self.words))
        else: 
            answer = answer.lower()
            if answer not in self.words: 
                raise ValueError("Answer not in Dictionary")
            self.given_word = answer
        self.guess_remaining = 6
        self.player_guess = ""
        self.guessed_words = [""]*self.guess_remaining
        self.colors_hist = []
        self.guessed_letter = {char: "Reset" for char in string.ascii_lowercase}
        # Belief state: Possible hidden answer 
        self.candidates = list(self.words)
        self.done = False 
        return self.get_observation()
        
    def get_observation(self): 
        return {"candidates": self.candidates.copy(), 
                "guesses_remaining": self.guess_remaining}

    @staticmethod
    def simulate_feedback(guess:str, answer:str): 
        guess = guess.lower()
        answer = answer.lower()
        if len(guess) != 5 or len(answer) != 5: 
            raise ValueError("Both words must have 5 letters")
        colors = ["Gray"] * 5 
        remaining = Counter(answer)
        # Green 
        for i in range(5): 
            if guess[i] == answer[i]: 
                colors[i] = "Green"
                remaining[guess[i]] -= 1 
        # Yellow 
        for i in range(5): 
            if colors[i] == "Green": 
                continue  
            if remaining[guess[i]] > 0: 
                colors[i] = "Yellow"
                remaining[guess[i]] -= 1
        return tuple(colors)

    def step(self, action:str): 
        if self.done: 
            raise RuntimeError("Gameover. Call reset().")
        guess = action.strip().lower()
        if guess not in self.words: 
            raise ValueError(f"Invalid guess: {guess}")
        self.player_guess = guess 
        colors = self.assign_color()
        # Record history 
        idx = 6 - self.guess_remaining
        self.guessed_words[idx] = guess 
        self.colors_hist.append(colors)
        self.guess_remaining -= 1
        # Update possible hidden answers 
        observed = tuple(colors)
        self.candidates = [word for word in self.candidates 
                           if self.simulate_feedback(guess, word) == observed]

        # Check termination 
        won = self.is_win()
        self.done = won or self.guess_remaining == 0 
       
        # Reward Calculation 
        reward = -1.0 
        if self.done and not won: 
            reward -= 6.0 
        
        info = {
               "guess": guess, 
               "feedback": observed, 
               "win": won, 
               "candidate_count": len(self.candidates), 
               }
        if self.render_mode != "train": 
            self.render(colors)
        return (self.get_observation(), reward, self.done, info)
