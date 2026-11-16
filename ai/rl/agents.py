import math 
from functools import lru_cache
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

class RollOutAgent: 
    def __init__(self, env: WordleEnv, max_guesses=100, max_rollout_candidates=12, loss_penalty=6.0): 
        self.env = env 
        # Base policy used for future decision
        self.base_agent = EntropyAgent(env, max_guesses)
        self.max_rollout_candidates = max_rollout_candidates
        self.loss_penalty = loss_penalty
        self.last_decision = {}

    @lru_cache(maxsize=200_000)
    def _feedback(self, guess, answer): 
        return tuple(self.env.simulate_feedback(guess, answer))

    @lru_cache(maxsize=50_000)
    def _partitions(self, candidates, guess): 
        groups = {}
        for answer in candidates: 
            pattern = self._feedback(guess, answer)
            groups.setdefault(pattern, []).append(answer)
        return tuple((pattern, tuple(words)) for pattern, words in groups.items())

    def _action_cost(self, candidates, remaining, guess): 
        total = len(candidates)
        cost = 1.0 
        for pattern, group in self._partitions(candidates, guess): 
            if pattern == ("Green", )*5: 
                continue 
            probability = len(group) / total
            if remaining == 1:
                # No guesses left after this wrong answer
                future_cost = self.loss_penalty
            else:
                future_cost = self._policy_value(group, remaining - 1)
            cost += probability * future_cost
        return cost

    @lru_cache(maxsize=50_000)
    def _policy_value(self, candidates, remaining):
        if remaining <= 0:
            return self.loss_penalty
        observation = {"candidates": list(candidates), "guesses_remaining": remaining}
        guess = self.base_agent.select_action(observation)
        return self._action_cost(candidates, remaining, guess)

    def select_action(self, observation):
        candidates = tuple(observation["candidates"])
        remaining = observation["guesses_remaining"]
        if not candidates:
            raise ValueError("No candidates remaining")
        # Always compute the baseline action
        base_guess = self.base_agent.select_action(observation)
        # Avoid expensive rollout on large belief states
        if len(candidates) > self.max_rollout_candidates:
            self.last_decision = {
                "method": "entropy",
                "evaluated": 0,
            }
            return base_guess
        if len(candidates) == 1:
            self.last_decision = {
                "method": "direct",
                "evaluated": 0,
            }
            return candidates[0]
        # Include baseline guess to guarantee comparison
        possible_guesses = list(dict.fromkeys([base_guess] + list(self.base_agent.get_possible_guesses(list(candidates)))))
        best_guess = base_guess
        best_cost = self._action_cost(candidates, remaining, base_guess)
        for guess in possible_guesses:
            if guess == base_guess:
                continue
            cost = self._action_cost(candidates, remaining, guess)
            if cost < best_cost - 1e-12:
                best_cost = cost
                best_guess = guess
        self.last_decision = {
            "method": "rollout",
            "evaluated": len(possible_guesses),
            "expected_cost": best_cost,
        }
        return best_guess
