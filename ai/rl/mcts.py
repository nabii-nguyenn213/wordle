from dataclasses import dataclass, field
from functools import lru_cache 
from math import log, sqrt 
from random import Random 
from ai.rl.agents import EntropyAgent
from ai.rl.environment import WordleEnv

@dataclass
class MoveStats: 
    visits:int = 0
    total_reward:float = 0.0
    children:dict = field(default_factory=dict)

@dataclass
class BeliefNode: 
    candidates: tuple 
    remaining: int 
    moves: dict 
    visits: int = 0

class MCTSAgent: 
    WIN = ("Green", )*5
    def __init__(self, env: WordleEnv, simulations=200, max_guesses=20, max_actions=12, 
                 max_mcts_candidates=10, exploration=2.0, loss_penalty=6.0, seed:int|None=42, 
                 value_model=None): 
        self.env = env

        self.base_agent = EntropyAgent(
            env=env,
            max_guesses=max_guesses,
        )

        self.simulations = simulations
        self.max_actions = max_actions
        self.max_mcts_candidates = max_mcts_candidates
        self.exploration = exploration
        self.loss_penalty = loss_penalty
        self.rng = Random(seed)
        self.last_decision = {}
        self.value_model = value_model

    @lru_cache(maxsize=300_000)
    def _feedback(self, guess, answer):
        return tuple(self.env.simulate_feedback(guess, answer))

    @lru_cache(maxsize=30_000)
    def _base_guess(self, candidates, remaining):
        observation = {
            "candidates": list(candidates),
            "guesses_remaining": remaining,
        }
        return self.base_agent.select_action(observation)

    @lru_cache(maxsize=30_000)
    def _actions(self, candidates):
        if len(candidates) == 1:
            return candidates
        possible = list(dict.fromkeys(
            list(candidates)
            + list(self.base_agent.get_possible_guesses(
                list(candidates)
            ))
        ))
        candidate_set = set(candidates)
        ranked = sorted(possible,
                        key=lambda word: (self.base_agent.calculate_entropy(word, candidates),word in candidate_set),
                        reverse=True)
        return tuple(ranked[:self.max_actions])

    def _new_node(self, candidates, remaining):
        actions = self._actions(candidates)
        return BeliefNode(
            candidates=candidates,
            remaining=remaining,
            moves= {action: MoveStats() for action in actions},
        )

    def _select_ucb(self, node):
        # Explore unvisited actions first.
        for guess, stats in node.moves.items():
            if stats.visits == 0:
                return guess
        def ucb(guess):
            stats = node.moves[guess]
            exploitation = (stats.total_reward / stats.visits)
            exploration = self.exploration * sqrt(log(node.visits + 1) / stats.visits)
            return exploitation + exploration
        return max(node.moves, key=ucb)

    def _rollout(self, candidates, remaining, hypothetical_answer):
        total_reward = 0.0
        for attempts in range(remaining, 0, -1):
            guess = self._base_guess(candidates, attempts)
            pattern = self._feedback(guess, hypothetical_answer)
            total_reward -= 1.0
            if pattern == self.WIN:
                return total_reward
            if attempts == 1:
                return total_reward - self.loss_penalty
            candidates = tuple(word for word in candidates if self._feedback(guess, word) == pattern)
        return total_reward - self.loss_penalty

    def _simulate(self, node, hypothetical_answer):
        guess = self._select_ucb(node)
        stats = node.moves[guess]
        pattern = self._feedback(guess, hypothetical_answer)
        if pattern == self.WIN:
            reward = -1.0
        elif node.remaining == 1:
            reward = -1.0 - self.loss_penalty
        else:
            child = stats.children.get(pattern)
            if child is None:
                # Expansion: update hypothetical belief.
                next_candidates = tuple(word for word in node.candidates if self._feedback(guess, word) == pattern)
                child = self._new_node(next_candidates, node.remaining - 1)
                stats.children[pattern] = child
                # First visit: evaluate the newly expanded node.
                if self.value_model is not None:
                    future_reward = -self.value_model.predict_cost(
                        next_candidates,
                        child.remaining,
                    )
                else:
                    future_reward = self._rollout(
                        next_candidates,
                        child.remaining,
                        hypothetical_answer,
                    )
            else:
                # Existing branch: continue tree search.
                future_reward = self._simulate(child, hypothetical_answer)
            reward = -1.0 + future_reward
        # Backpropagation
        node.visits += 1
        stats.visits += 1
        stats.total_reward += reward
        return reward

    def select_action(self, observation):
        candidates = tuple(observation["candidates"])
        remaining = int(observation["guesses_remaining"])
        if not candidates or remaining < 1:
            raise ValueError("Invalid observation")
        if len(candidates) == 1:
            self.last_decision = {
                "method": "direct",
                "simulations": 0,
            }
            return candidates[0]
        # Avoid expensive search on large belief states.
        if len(candidates) > self.max_mcts_candidates:
            self.last_decision = {
                "method": "entropy",
                "simulations": 0,
            }
            return self.base_agent.select_action(observation)
        # Build a fresh search tree.
        root = self._new_node(candidates, remaining)
        for _ in range(self.simulations):
            # Sample a possible hidden answer.
            answer = self.rng.choice(candidates)
            self._simulate(root, answer)
        # Select the action visited most frequently.
        best_guess = max(root.moves,
                         key=lambda guess: (root.moves[guess].visits, 
                                           (root.moves[guess].total_reward / root.moves[guess].visits 
                                            if root.moves[guess].visits else float("-inf"))))
        stats = root.moves[best_guess]
        self.last_decision = {
            "method": "mcts",
            "simulations": self.simulations,
            "actions": len(root.moves),
            "expected_cost_estimate": (
                -stats.total_reward / stats.visits
            ),
        }
        return best_guess
