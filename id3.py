#Libraries
import math
from sklearn import datasets
import pandas as pd
import copy

def entropy_table(data, target):
    counttarget = {}
    for row in target:
        counttarget[row[0]] = 0
    for row in target:
        counttarget[row[0]] += 1
    print(counttarget)


def list_value(index, data):
    listval = set(data[index])
    countvalue = len(listval)
    return(listval)

def attribute_table(data, target):
    attrtarget = {}
    for row in target:
        attrtarget[row[0]] = 0

    attrdict = {}
    for row in data:
        attrins = {}
        for i in range(0, len(row)):
            if row[i] not in attrins:
                attrins[row[i]] = copy.deepcopy(attrtarget) 
            attrins[row[i]][target[i][0]] += 1
        attrdict[row] = attrins

    return attrdict

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

    return entropy_total + temp_gain

def remove_root(data, target, val):
    dataret, targetret = [], []

    for i,layer in enumerate(data):
        for element in layer:
            if element == val:
                dataret.append(layer)
                targetret.append(target[i])
                break

    return dataret, targetret

def id3(data, target, tree=None):
    if tree is None:
        tree = {}

    if entropy_count(entropy_table(data, target)) == 0:
        tree = {target[0]}
    else:
        gainParam = attribute_table(data, target)
        gainList = []
        for i in gainParam:
            res = information_gain(i)
            gainList.append(res)
        root = gainList.index(max(gainList))
        rootVal = attName[root]

        for i in list_value(root, data):
            dataret, targetret = remove_root(data, target, i)
            if entropy_count(entropy_table(data, target)) == 0:
                tree[rootVal][i] = targetret[0]
            else:
                tree[rootVal][i] = id3(dataret, targetret)

    return tree

# Read iris dataset
data_iris = datasets.load_iris()
data_iris
data_iris['target']

# Read play-tennis dataset
data_play_tennis = pd.read_csv('play-tennis.csv')
attName = data_play_tennis.columns.tolist()
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

data_play_tennis_data = list(zip(*data_play_tennis_data))
# print(data_play_tennis_data[0])
attribute_table(data_play_tennis_data, data_play_tennis_target)