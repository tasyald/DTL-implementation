# Libraries
from math import log
from sklearn import datasets
import pandas as pd
import math

def missing_value(data, target):
    for i in len(data):
        for j in len(data[i]):
            if math.isnan(data[i][j]):
                t = target[i]
    




# Read iris dataset
data_iris = datasets.load_iris()
data_iris
print(data_iris['target'])

# Read play-tennis dataset
data_play_tennis = pd.read_csv('play-tennis.csv')
data_play_tennis = data_play_tennis.values.tolist()
data_play_tennis
