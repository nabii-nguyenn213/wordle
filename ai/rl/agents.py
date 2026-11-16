import math 
from collections import Counter
from ai.rl.environment import WordleEnv

class EntropyAgent: 
    def __init__(self, env: WordleEnv, max_guesses=100): 
        self.env = env
        self.max_guesses = max_guesses

    def calculate_entropy(self, guess, candidates): 
        feedback_counts = Counter()

        for answer in candidates: 
            feedback = self.env.simulate_feedback(guess, answer)
            feedback_counts[feedback] += 1
        total = len(candidates)
        entropy = 0.0

        for count in feedback_counts.values(): 
            probability = count/total 
            entropy -= probability * math.log2(probability)
        return entropy

    def get_possible_guesses(self, candidates): 
        if len(self.env.words) <= self.max_guesses: 
            return self.env.words 
        # Prefer words containing common letters 
        frequency = Counter(letter for word in candidates for letter in set(word))
        ranked = sorted(self.env.words, 
                        key=lambda word:sum(frequency[letter] for letter in set(word)), 
                        reverse=True)
        if len(candidates) <= self.max_guesses: 
            candidate_set = set(candidates)
            others = [word for word in ranked if word not in candidate_set]
            return list(candidates) + others[:self.max_guesses-len(candidates)]
        return ranked[:self.max_guesses]

    def select_action(self, observation): 
        candidates = observation["candidates"]
        if not candidates: 
            raise ValueError("No candidates remaining")
        if len(candidates) == 1: 
            return candidates[0]
        possible_guesses = self.get_possible_guesses(candidates)
        candidate_set = set(candidates)
        best_guess = None
        best_score = (-1.0, False)
        for guess in possible_guesses: 
            entropy = self.calculate_entropy(guess, candidates)
            score = (entropy, guess in candidate_set)
            if score > best_score: 
                best_score = score 
                best_guess = guess
        return best_guess
