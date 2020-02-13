# Libraries
from math import log
from sklearn import datasets
import pandas as pd
import math
from collections import Counter
import copy
from sklearn.model_selection import train_test_split
import numpy
from operator import itemgetter


def entropy_count (dictionary) :
    sumentropy = 0
    counttotal = 0
    for item in dictionary :
        counttotal = counttotal + dictionary[item]
    if not (counttotal == 0):
        for item in dictionary :
            probability = dictionary[item]/counttotal
            if not (probability ==0):
                sumentropy = sumentropy - (probability)*math.log2(probability)
        return sumentropy
    else :
        return 0 

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
    temp_gain = 0
    for attr in dictionary :        
        count_total_attribute = 0
        entropy_attr = entropy_count(dictionary[attr])
        for target in dictionary[attr] : 
            count_total_attribute = count_total_attribute + dictionary[attr][target] 
        temp_gain = temp_gain - (count_total_attribute/total_dataset) * entropy_attr
    return entropy_total + temp_gain

def gain_ratio(dictionary):
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
    try:
        ratio = gain/split
    except:
        ratio = 0
    return ratio
    

def continuous_value (data, target) :
    tempdict = {}
    init_target = copy.deepcopy(target)
    for cell in target : 
        tempdict[str(cell[0])] = 0
    attrdict = {}
    for i in range(0,len(data)): 
        target[i].append(data[i])
    target.sort(key=itemgetter(0), reverse=False)
    target.sort(key=itemgetter(1), reverse=False)
    previous_value = target[0][0]
    best_gain = -999
    print(target)
    for j in range(1,len(target)) :
        if not(previous_value == target[j][0]):
            previous_value = target[j][0]
            print("===========",j)
            if j+1 < len(target):
                attrdict = {"<"+str(target[j+1][1]):copy.deepcopy(tempdict),">="+str(target[j+1][1]):copy.deepcopy(tempdict)}
            else :
                attrdict = {"<"+str(target[j][1]):copy.deepcopy(tempdict),">="+str(target[j][1]):copy.deepcopy(tempdict)}
            for i in range(0,len(target)):
                if j+1 < len(target):
                    if target[i][1]<target[j+1][1] :
                        attrdict["<"+str(target[j+1][1])][str(target[i][0])] = attrdict["<"+str(target[j+1][1])][str(target[i][0])] + 1 
                    else:
                        attrdict[">="+str(target[j+1][1])][str(target[i][0])] = attrdict[">="+str(target[j+1][1])][str(target[i][0])] + 1
                else : 
                    if target[i][1]<target[j][1] :
                        attrdict["<"+str(target[j][1])][str(target[i][0])] = attrdict["<"+str(target[j][1])][str(target[i][0])] + 1 
                    else:
                        attrdict[">="+str(target[j][1])][str(target[i][0])] = attrdict[">="+str(target[j][1])][str(target[i][0])] + 1
            print(attrdict)
            if best_gain < information_gain(attrdict) :
                best_gain = information_gain(attrdict)
                potential_split = (target[j][1]+target[j-1][1])/2
        else :
            pass
    for i in range(0,len(data)): 
        target[i].pop()
    return (potential_split,best_gain)
        
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
    for i in range(len(data_target)):
        data_target[i].append(target[i][0])
    data_target = numpy.array(data_target)
    train_data ,test_data = train_test_split(data_target,test_size=0.2)

# def rule_set(tree_dictionary, rule) :
#     if not(type(tree_dictionary) is dict) :
#         return 
#     pass

def evaluate(tree_dictionary, row, original_target, header):
    if not(type(tree_dictionary) is dict) :
        # print (tree_dictionary)
        return original_target == tree_dictionary
    else :
        for item in tree_dictionary :
            if row[header.index(item.lower())] in tree_dictionary[item]:
                return evaluate(tree_dictionary[item][row[header.index(item.lower())]],row,original_target,header)
        return False
        
def accuracy(tree, data, target, header):
    count_true = 0
    for i in range(0,len(data)) : 
        if evaluate(tree,data[i], target[i], header) :
            count_true = count_true + 1
    print(count_true)
    return count_true/len(data)

def pruning(data, target, tree_dictionary_iterator, tree_dictionary_root, header) : 
    if not(type(tree_dictionary_iterator) is dict) :
        pass
    else :
        # prune_flag = True
        for child in tree_dictionary_iterator:
            if not(type(tree_dictionary_iterator[child]) is dict) :
                temp_tree = copy.deepcopy(tree_dictionary_root)
                temp_value = copy.deepcopy(tree_dictionary_iterator[child])
                print(child)
                del tree_dictionary_iterator[child]
                print(tree_dictionary_root)
                print(accuracy(tree_dictionary_root,data,target,header))
                if accuracy(tree_dictionary_root,data,target,header)<=accuracy(temp_tree,data,target,header):
                    tree_dictionary_iterator[child] = copy.deepcopy(temp_value)
            else :
                pruning(data,target,tree_dictionary_iterator[child],tree_dictionary_root, header)
                if len(tree_dictionary_iterator[child]) == 0 :
                    del tree_dictionary_iterator[child] 

def remove_root(data, target, val):
    dataret, targetret = [], []
    for i,layer in enumerate(data):
        for element in layer:
            if element == val:
                dataret.append(layer)
                targetret.append(target[i])
                break
    # print("DATARET:",dataret)
    # print("TARGETRET:", targetret)
    return dataret, targetret

def remove_root_continuous(data, target, val, column):
    dataret, targetret ,removedret, removedtargetret= [], [], [], []
    for i,layer in enumerate(data):
        if layer[column] < val:
            print("el", layer)
            dataret.append(layer)
            targetret.append(target[i])
        else :
            removedret.append(layer)
            removedtargetret.append(target[i])
    # print("DATARET:",dataret)
    # print("TARGETRET:", targetret)
    return dataret, targetret, removedret, removedtargetret

def entropy_table(data, target):
    counttarget = {}
    for row in target:
        counttarget[row[0]] = 0
    for row in target:
        counttarget[row[0]] += 1
    
    return counttarget

def list_value(index, data):
    listval = set()
    for i in data:
        listval.add(i[index])

    return listval

def attribute_table(data, target):
    dataTemp = copy.deepcopy(data)
    dataTrans = list(zip(*dataTemp))

    attrtarget = {}
    for row in target:
        attrtarget[row[0]] = 0

    attrdict = {}
    for row in dataTrans:
        attrins = {}
        for i in range(0, len(row)):
            if row[i] not in attrins:
                attrins[row[i]] = copy.deepcopy(attrtarget) 
            attrins[row[i]][target[i][0]] += 1
        attrdict[row] = attrins

    return attrdict

def myC45(data, target, tree=None):
    tempdict = {}
    missing_value(data, target)
    if tree is None:
        tree = {}

    # for d in range(len(data)):
    #     if type(data[d][0]) == float or type(data[d][0]) == int:
    #         potential_split, best_gain = continuous_value(data[d], target)
    #         dataTrans = list(zip(*data))
    #         print("PS", potential_split)
    #         cv = True
    #         if entropy_count(entropy_table(dataTrans, target)) == 0:
    #             tree = target[0]
    #         else:
    #             gainParam = attribute_table(dataTrans, target)
    #             gainList = []
    #             for i in gainParam:
    #                 res = gain_ratio(gainParam[i])
    #                 gainList.append(res)
    #             root = gainList.index(max(gainList))
    #             rootVal = attName[root] + " < " + str(potential_split)
    #             tree[rootVal] = {}
    #             for i in list_value(root, dataTrans):
    #                 dataret, targetret = remove_root(dataTrans, target, i, cv)
    #                 if entropy_count(entropy_table(dataret, targetret)) == 0:
    #                     tree[rootVal][i] = targetret[0]
    #                 else:
    #                     tree[rootVal][i] = myC45(dataret, targetret)
    #         return tree

    if entropy_count(entropy_table(data, target)) == 0:
        tree = target[0]
    else:
        gainParam = attribute_table(data, target)
        gainList = []
        splitList = []
        for i in gainParam:
            print("o",i[0])
            if type(i[0]) == float or type(i[0]) == int:
                print("==========i", i)
                potential_split, res = continuous_value(i, target)
                gainList.append(res)
                splitList.append(potential_split)
            else:
                splitList.append(-999)
                res = gain_ratio(gainParam[i])
                gainList.append(res)
        root = gainList.index(max(gainList))
        print("root", root)
        print("SL", splitList)
        print("GL", gainList)
        rootVal = attName[root]
        tree[rootVal] = {}
        if (splitList[root] == -999):
            for i in list_value(root, data):
                dataret, targetret = remove_root(data, target, i)
                if entropy_count(entropy_table(dataret, targetret)) == 0:
                    tree[rootVal][i] = targetret[0]
                else:
                    tree[rootVal][i] = myC45(dataret, targetret)
        else:
            dataret,targetret , removedret, removedtargetret= remove_root_continuous(data, target, splitList[root], root)
            # print("dataret =========", dataret)
            # print("removed =========", removedret)
            # print("entropy =========", entropy_count(entropy_table(dataret, targetret)))
            # print("entropyremoved =========", entropy_count(entropy_table(removedret, removedtargetret)))
            if entropy_count(entropy_table(dataret, targetret)) <= 0.5:
                if not(len(targetret)==0) :
                    tree[rootVal][i] = targetret[0]
            else:
                tree[rootVal][i] = myC45(dataret, targetret)
            if entropy_count(entropy_table(removedret, removedtargetret)) <= 0.5:
                if not(len(removedtargetret)==0) :
                    tree[rootVal][i] = removedtargetret[0]
            else:
                try:
                    tree[rootVal][i] = myC45(removedret, removedtargetret)
                except RecursionError:
                    pass
    return tree

def pretty(d, indent=0):
    for key, value in d.items():
        print('    ' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('    ' * (indent+1) + str(value))

    


# Read iris dataset
data_iris = datasets.load_iris()
data_iris_target = data_iris['target'].tolist()
data_iris_data = data_iris['data'].tolist()
for i in range(0,len(data_iris_target)) :
    data_iris_target[i] = [data_iris_target[i]]
attName = data_iris['feature_names']
# print(attName)

print(pretty(myC45(data_iris_data, data_iris_target)))


# Read play-tennis dataset
data_play_tennis = pd.read_csv('play-tennis.csv')
attName = data_play_tennis.columns.tolist()[1:5]
data_play_tennis = data_play_tennis.values.tolist()
# play-tennis target
data_play_tennis_target = []
for x in data_play_tennis:
    data_play_tennis_target.append(x[-1:])

# play-tennis data
data_play_tennis_data = []
for x in data_play_tennis:
    data_play_tennis_data.append(x[:-1][1:])

dummy_tree = {'Outlook': {'Sunny': {'humidity': { 'High': ['No']}}, 'Overcast': ['Yes'], 
'Rain': {'wind': {'Weak': ['Yes'], 'Strong': ['No']}}}}
# print (accuracy(data_play_tennis_data, data_play_tennis_target, data_play_tennis_header))
pruning(data_play_tennis_data, data_play_tennis_target,dummy_tree,dummy_tree, data_play_tennis_header )
print(dummy_tree)

missing_value(data_play_tennis_data, data_play_tennis_target)
print(pretty(myC45(data_play_tennis_data, data_play_tennis_target)))


split_validation_data(data_play_tennis_data, data_play_tennis_target)


