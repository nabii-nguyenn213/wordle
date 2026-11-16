import numpy 
from utils import read_dataset

# Test loading dataset
def test_dataset_loading(): 
    # Dataset is taken from : https://github.com/steve-kasica/wordle-words/blob/master/wordle.csv
    dataset_path = "./dataset/wordle.csv"

    words, probabilities = read_dataset(dataset_path)

    print(words.shape, probabilities.shape)
    print(type(words), type(probabilities))
    # print(f"Total probabilities: {numpy.sum(probabilities, dtype=numpy.float128)}")

if __name__ == "__main__": 
    test_dataset_loading()
