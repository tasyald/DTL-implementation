# Libraries
from math import log
from sklearn import datasets
import pandas as pd
import math
from collections import Counter
import copy

def entropy_count (dictionary) :
    sumentropy = 0
    counttotal = 0
    for item in dictionary :
        counttotal = counttotal + dictionary[item]
    # print(counttotal)
    for item in dictionary :
        probability = dictionary[item]/counttotal
        if not (probability ==0):
            sumentropy = sumentropy - (probability)*math.log2(probability)
    print(sumentropy)
    pass
	
    # print(sumentropy)
    return sumentropy

def information_gain (dictionary) :
    
    # print(dictionary)
    # print("jancok")
    total_entropy_dict = {"total" : {}}
    
    for attr in dictionary :
        for target in dictionary[attr] : 
            if target in total_entropy_dict["total"] :
                total_entropy_dict["total"][target] = total_entropy_dict["total"][target]+ dictionary[attr][target]; 
            else :
                total_entropy_dict["total"][target] = dictionary[attr][target]; 
    total_dataset = 0
    for target in total_entropy_dict["total"]:
        total_dataset = total_dataset + total_entropy_dict["total"][target] 
    # print(total_entropy_dict)
    # print(total_dataset)
    entropy_total = entropy_count(total_entropy_dict["total"])
    print(entropy_total)
    temp_gain = 0
    for attr in dictionary :        
        count_total_attribute = 0
        entropy_attr = entropy_count(dictionary[attr])
        for target in dictionary[attr] : 
            count_total_attribute = count_total_attribute + dictionary[attr][target] 
        temp_gain = temp_gain - (count_total_attribute/total_dataset) * entropy_attr
    print("INI GAINNYA")
    print(entropy_total + temp_gain)
    
        # entropy_count(attrdict[attr])

def continouous_value (data, target) :
    tempdict = {}
    for cell in target : 
        tempdict[cell[0]] = 0
    # initdict = copy.deepcopy(tempdict)
    print(tempdict)
    for row in data:
        attrdict = {}
        # tempdict = copy.deepcopy(initdict)
        for i in range(0,len(row)): 
            if row[i] in attrdict:
                attrdict[row[i]][target[i][0]] = attrdict[row[i]][target[i][0]] + 1 
            else :
                attrdict[row[i]] = copy.deepcopy(tempdict)
                attrdict[row[i]][target[i][0]] = attrdict[row[i]][target[i][0]] + 1 
        print(attrdict)
        information_gain(attrdict)
        
        
def missing_value(data, target):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if type(data[i][j]) == float:
                if math.isnan(data[i][j]):
                    t = target[i]
                    values = []
                    for k in range(len(target)):
                        if target[k] == t and k != i:
                            values.append(data[k][j])
                    freq = Counter(values)
                    data[i][j] = freq.most_common()[0][0]

# Read iris dataset
data_iris = datasets.load_iris()
data_iris
data_iris['target']

# Read play-tennis dataset
data_play_tennis = pd.read_csv('play-tennis.csv')
data_play_tennis = data_play_tennis.values.tolist()
data_play_tennis

# play-tennis target
data_play_tennis_target = []
for x in data_play_tennis:
    data_play_tennis_target.append(x[-1:])
data_play_tennis_target

# play-tennis data
data_play_tennis_data = []
for x in data_play_tennis:
    data_play_tennis_data.append(x[:-1][1:])
data_play_tennis_data

missing_value(data_play_tennis_data, data_play_tennis_target)
data_play_tennis_data = list(zip(*data_play_tennis_data))
print(data_play_tennis_data[0])
continouous_value(data_play_tennis_data, data_play_tennis_target)
