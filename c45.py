# Libraries
from math import log
from sklearn import datasets
import pandas as pd
import math
from statistics import mode
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

def continouous_value (data, target) :
    tempdict = {}
    for cell in target : 
        tempdict[str(cell[0])] = 0
    # initdict = copy.deepcopy(tempdict)
    # print(tempdict)
    # print(data)
    init_target = copy.deepcopy(target)
    # sortattr = row.
    attrdict = {}
    for i in range(0,len(data)): 
        target[i].append(data[i])
    target.sort(key=itemgetter(0), reverse=False)
    target.sort(key=itemgetter(1), reverse=False)
    # print(target)
    # tempdict = copy.deepcopy(initdict)
    previous_value = target[0][0]
    # for i in range(0,len(target)):
    #     if target[i][1] in attrdict :
    #         attrdict[target[i][1]][target[i][0]] = attrdict[target[i][1]][target[i][0]] + 1 
    #     else :
    #         attrdict[target[i][1]] = copy.deepcopy(tempdict)
    #         attrdict[target[i][1]][target[i][0]] = attrdict[target[i][1]][target[i][0]] + 1 
    # best_gain = information_gain(attrdict)
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
                # if target[i][1]<previous_value :
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
                    data[i][j] = mode(values)




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
from operator import itemgetter

print(data_play_tennis_data[0])
data_iris_data = list(zip(*data_iris_data))
data_play_tennis_data = list(zip(*data_play_tennis_data))
print(continouous_value(data_play_tennis_data[0], data_play_tennis_target))
