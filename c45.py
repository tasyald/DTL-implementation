# Libraries
from math import log
from sklearn import datasets
import pandas as pd
import math
import copy
def entropy_count (dictionary) :
    sumentropy = 0
    counttotal = 0
    for item in dictionary :
        counttotal = counttotal + dictionary[item]
    print(counttotal)
    for item in dictionary :
        probability = dictionary[item]/counttotal
        if not (probability ==0):
            sumentropy = sumentropy - (probability)*math.log2(probability)
    print(sumentropy)
    pass
	

def continouous_value (data, target) :
    tempdict = {}
    for cell in target : 
        tempdict[cell[0]] = 0
    # initdict = copy.deepcopy(tempdict)
    print(tempdict)
    for row in data:
        attrdict = {}
        # tempdict = copy.deepcopy(initdict)
        for i in range(0,len(row)-1): 
            if row[i] in attrdict:
                attrdict[row[i]][target[i][0]] = attrdict[row[i]][target[i][0]] + 1 
            else :
                attrdict[row[i]] = copy.deepcopy(tempdict)
                attrdict[row[i]][target[i][0]] = attrdict[row[i]][target[i][0]] + 1 
        print(attrdict)
        for attr in attrdict :
            entropy_count(attrdict[attr])
        

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

data_play_tennis_target = []
for x in data_play_tennis:
    data_play_tennis_target.append(x[-1:])
data_play_tennis_target

data_play_tennis_data = []
for x in data_play_tennis:
    data_play_tennis_data.append(x[:-1][1:])
data_play_tennis_data

data_play_tennis_data = list(zip(*data_play_tennis_data))
print(data_play_tennis_data[0])
continouous_value(data_play_tennis_data, data_play_tennis_target)
