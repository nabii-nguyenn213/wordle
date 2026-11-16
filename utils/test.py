import numpy 
from time import perf_counter
from utils.utils import read_dataset
from ai.rl.environment import WordleEnv
from ai.rl.agents import EntropyAgent, RollOutAgent

# Test loading dataset
def test_dataset_loading(): 
    # Dataset is taken from : https://github.com/steve-kasica/wordle-words/blob/master/wordle.csv
    dataset_path = "./dataset/wordle.csv"

    words, probabilities = read_dataset(dataset_path)

    print(words.shape, probabilities.shape)
    print(type(words), type(probabilities))
    # print(f"Total probabilities: {numpy.sum(probabilities, dtype=numpy.float128)}")

def test_wordle_env(): 
    env = WordleEnv(render_mode="rich", seed=42)
    state = env.reset(seed=42)
    print("Candidates:", len(state["candidates"]))
    print("Guesses left:", state["guesses_remaining"])
    guess = "state"
    if guess in env.words: 
        state, reward, done, info = env.step(guess)
        print("Feedback:", info["feedback"])
        print("Reward:", reward)
        print("Done:", done)
        print("Candidates left:", info["candidate_count"])

def test_entropy_agent(total_games=10, seed=None): 
    number_of_win=0
    for i in range(total_games): 
        env = WordleEnv(render_mode="rich", seed=seed)
        agent = EntropyAgent(env=env, max_guesses=20)
        observation = env.reset(seed=seed)
        done = False 
        while not done: 
            guess = agent.select_action(observation)
            observation, reward, done, info = env.step(action=guess)

            print("Guess:", guess)
            print("Feedback:", info["feedback"])
            print("Candidates:", info["candidate_count"])
            print("Reward:", reward)
            print()

        print("Win:", info["win"])
        if info["win"]:
            number_of_win+=1
        print("Total guesses:", 6 - env.guess_remaining)
    print("Win rate:", number_of_win/total_games)

def test_rollout_agent(total_games=10, seed=None): 
    number_of_win=0
    for i in range(total_games): 
        env = WordleEnv(render_mode="rich", seed=seed)
        agent = RollOutAgent(env=env, max_guesses=20, max_rollout_candidates=12)
        observation = env.reset(seed=seed)
        done = False
        while not done:
            candidates = len(observation["candidates"])
            start = perf_counter()
            guess = agent.select_action(observation)
            elapsed = perf_counter() - start
            observation, reward, done, info = env.step(action=guess)
            print(f"\nGuess: {guess}")
            print("Feedback:", info["feedback"])
            print("Candidates:", info["candidate_count"])
            print("Method:", agent.last_decision["method"])
            print(f"Planning time: {elapsed:.3f}s")
            if agent.last_decision["method"] == "rollout":
                print(
                    "Expected cost:",
                    agent.last_decision["expected_cost"]
                )
        print("\nWin:", info["win"])
        if info["win"]:
            number_of_win+=1
        print("Total guesses:", len(env.colors_hist))
    print("Win rate:", number_of_win/total_games)

if __name__ == "__main__": 
    # test_dataset_loading()
    # test_wordle_env()
    # test_entropy_agent(total_games=20)
    test_rollout_agent(total_games=20)
