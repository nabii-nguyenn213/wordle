import numpy 
from utils.utils import read_dataset
from ai.rl.environment import WordleEnv

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

if __name__ == "__main__": 
    # test_dataset_loading()
    test_wordle_env()
