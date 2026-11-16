import os 
import pandas 
import numpy 

def read_dataset(path: str): 
    if not os.path.isfile(path): 
        raise FileNotFoundError(f"File not found {path}")
    if not path.endswith('csv'): 
        raise ValueError(f"Dataset must be .csv")
    dataset = pandas.read_csv(path)
    
    words = dataset.iloc[:, 0]
    probabilities = dataset.iloc[:, 1]
    return numpy.array(words), numpy.array(probabilities, dtype=numpy.float128)
