# Libraries
from math import log
from sklearn import datasets
import pandas as pd
import math
from collections import Counter
import copy
from sklearn.model_selection import train_test_split
import numpy

def entropy_count (dictionary) :
    sumentropy = 0
    counttotal = 0
    for item in dictionary :
        counttotal = counttotal + dictionary[item]
    for item in dictionary :
        probability = dictionary[item]/counttotal
        if not (probability ==0):
            sumentropy = sumentropy - (probability)*math.log2(probability)
    return sumentropy

def information_gain (dictionary) :
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
    entropy_total = entropy_count(total_entropy_dict["total"])
    # print(entropy_total)
    temp_gain = 0
    for attr in dictionary :        
        count_total_attribute = 0
        entropy_attr = entropy_count(dictionary[attr])
        for target in dictionary[attr] : 
            count_total_attribute = count_total_attribute + dictionary[attr][target] 
        temp_gain = temp_gain - (count_total_attribute/total_dataset) * entropy_attr
    # print("INI GAINNYA")
    # print(entropy_total + temp_gain)
    return entropy_total + temp_gain
        # entropy_count(attrdict[attr])

def gain_ratio(dictionary):
    print('dict', dictionary)
    total_entropy_dict = {"total" : {}}
    for attr in dictionary:
        for target in dictionary[attr] : 
            if target in total_entropy_dict["total"] :
                total_entropy_dict["total"][target] = total_entropy_dict["total"][target]+ dictionary[attr][target]; 
            else :
                total_entropy_dict["total"][target] = dictionary[attr][target]; 
    total_dataset = 0
    for target in total_entropy_dict["total"]:
        total_dataset = total_dataset + total_entropy_dict["total"][target]
    entropy_total = entropy_count(total_entropy_dict["total"])
    split = 0
    temp_gain = 0
    for attr in dictionary :        
        count_total_attribute = 0
        entropy_attr = entropy_count(dictionary[attr])
        for target in dictionary[attr] : 
            count_total_attribute = count_total_attribute + dictionary[attr][target] 
        temp_gain = temp_gain - (count_total_attribute/total_dataset) * entropy_attr
        split = split - (count_total_attribute/total_dataset) * math.log2(count_total_attribute/total_dataset)
    gain = entropy_total + temp_gain
    print('SPLIT: ', split)
    print('INI GAIN RATIO: ', gain/split)
    

def continouous_value (data, target) :
    tempdict = {}
    for cell in target : 
        tempdict[str(cell[0])] = 0
    attrdict = {}
    for i in range(0,len(data)): 
        target[i].append(data[i])
    target.sort(key=itemgetter(0), reverse=False)
    target.sort(key=itemgetter(1), reverse=False)
    previous_value = target[0][0]
    best_gain = -999
    for j in range(1,len(target)) :
        if not(previous_value == target[j][0]):
            # print(previous_value)
            previous_value = target[j][0]
            attrdict = {"<"+str(target[j+1][1]):copy.deepcopy(tempdict),">="+str(target[j+1][1]):copy.deepcopy(tempdict)}
            for i in range(0,len(target)):
                if target[i][1]<target[j+1][1] :
                    attrdict["<"+str(target[j+1][1])][str(target[i][0])] = attrdict["<"+str(target[j+1][1])][str(target[i][0])] + 1 
                else:
                    attrdict[">="+str(target[j+1][1])][str(target[i][0])] = attrdict[">="+str(target[j+1][1])][str(target[i][0])] + 1 
                # if target[i][1] in attrdict :
                #     attrdict[target[i][1]][target[i][0]] = attrdict[target[i][1]][target[i][0]] + 1 
                # else :
                #     attrdict[target[i][1]] = copy.deepcopy(tempdict)
                #     attrdict[target[i][1]][target[i][0]] = attrdict[target[i][1]][target[i][0]] + 1 
            # print(attrdict)
            if best_gain < information_gain(attrdict) :
                best_gain = information_gain(attrdict)
                potential_split = (target[j][1]+target[j-1][1])/2
        else :
            pass
    # print("======================")
    # print(potential_split)
    # print(best_gain)
    return(potential_split,best_gain)
    target = copy.deepcopy(init_target)
        
        
        
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

def split_validation_data(data, target):
    data_target = copy.deepcopy(data)
    print(data_target)
    for i in range(len(data_target)):
        data_target[i].append(target[i][0])
    data_target = numpy.array(data_target)
    train_data ,test_data = train_test_split(data_target,test_size=0.2)
    print('Train')
    print(train_data)
    print('Test')
    print(test_data)

# def accuracy(data, target):
#     myC45(data, target)


def myC45(data, target):
    tempdict = {}
    missing_value(data, target)
    data = list(zip(*data))
    for cell in target: 
        tempdict[str(cell[0])] = 0
    for row in data:
        print(row)
        attrdict = {}
        for i in range(0,len(row)): 
            if row[i] in attrdict:
                attrdict[row[i]][target[i][0]] = attrdict[row[i]][target[i][0]] + 1 
            else :
                attrdict[row[i]] = copy.deepcopy(tempdict)
                attrdict[row[i]][target[i][0]] = attrdict[row[i]][target[i][0]] + 1
        



# Read iris dataset
data_iris = datasets.load_iris()
# print(data_iris)
data_iris_target = data_iris['target'].tolist()
data_iris_data = data_iris['data'].tolist()
print(data_iris_data)
for i in range(0,len(data_iris_target)) :
    data_iris_target[i] = [data_iris_target[i]]
print(data_iris_target)

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
data_play_tennis_data_tranpose = list(zip(*data_play_tennis_data))
# continouous_value(data_play_tennis_data, data_play_tennis_target)

split_validation_data(data_play_tennis_data, data_play_tennis_target)

from operator import itemgetter

print('ini:', data_play_tennis_data)
# data_iris_data = list(zip(*data_iris_data))
# data_play_tennis_data = list(zip(*data_play_tennis_data))
# print(continouous_value(data_play_tennis_data[0], data_play_tennis_target))

myC45(data_play_tennis_data, data_play_tennis_target)
