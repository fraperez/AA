from id3 import ID3, shanon_gain, gini_gain
from random_forest import RandomForest
import random
import pandas as pd
# import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from console import train_test_split, load_data
import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset")
# ap.add_argument("-p", "--predictor", type=str, default="decision_tree",
# 	help="predictor to use [decision_tree, random_forest]")
# ap.add_argument('-g', '--gain', type=str, default='shanon',
#     help='gain function to use [shanon, gini]')
# ap.add_argument('-ntree', '--num_tree', type=int, default=10,
#     help='number of trees to use')
# ap.add_argument('-ndata', '--n_data', type=float, default=0.8,
#     help='percentage of data to use in each tree [needed for random forest')
# ap.add_argument('-nattr', '--n_attr', required=True, type=int, default=3,
#     help='number of attr to analyze (depth of tree) default = 3')
# ap.add_argument('-t', '--test', required=True, type=float, default=0.3,
#     help='percentage of data to use as test')
# args = vars(ap.parse_args())


# build accuracy plot for each method


def claculate_acc(predictor, train_data, test_data):
    # now we test
    predictions_test = []
    # make predictions
    for index, case in test_data.iterrows():
        case = case.to_dict()
        predictions_test.append(predictor.predict(case))

    labels = [1, 0]
    conf_matrix = confusion_matrix(test_data.Survived.to_list(), predictions_test, labels=labels)
    accuracy_test = (conf_matrix[0,0] + conf_matrix[1, 1]) / float(len(test_data))

    # now for training

    predictions_train = []
    # make predictions
    for index, case in train_data.iterrows():
        case = case.to_dict()
        predictions_train.append(predictor.predict(case))

    conf_matrix = confusion_matrix(train_data.Survived.to_list(), predictions_train, labels=labels)
    accuracy_train = (conf_matrix[0,0] + conf_matrix[1, 1]) / float(len(train_data))

    return accuracy_train, accuracy_test



# data = load_data(args['dataset'], '\t', encoding='utf-8')
data = load_data('data/titanic.csv', '\t', encoding='utf-8')
# just fill nans
data = data.fillna(35)

# remember we need to categorize age
age_labels = ['child', 'teenage', 'young_adult', 'adult', 'old', 'retired']
categorical_age = pd.cut(data.Age, len(age_labels), labels=['child', 'teenage', 'young_adult', 'adult', 'old', 'retired'])
data.Age = categorical_age

# general settings
attrs = {
    'Age': age_labels,
    'Pclass': [1,2,3],
    'Sex': ['male', 'female']

}

goal_attr = {'Survived': [0,1]}

train_data, test_data = train_test_split(data, 0.3)


# #  single tree
# attrs_to_use = {}
# accuracies_train = []
# accuracies_test = []
# nodes = []
# for k, v in attrs.items():
#     gain_func = gini_gain
#     id3 = ID3(gain_func)
#     attrs_to_use[k] = v
#     tree = id3.train(train_data, goal_attr, attrs_to_use)
#     acc = claculate_acc(tree, train_data, test_data)
#     accuracies_train.append(acc[0])
#     accuracies_test.append(acc[1])
#     nodes.append(tree.get_tree_size())
# accuracies_test.reverse()
# accuracies_train.reverse()

# random forest
attrs_to_use = {}
accuracies_train = []
accuracies_test = []
nodes = []
for k, v in attrs.items():
    gain_func = shanon_gain
    forest = RandomForest(gain_func)
    attrs_to_use[k] = v
    forest.train(10, 0.8, len(attrs_to_use), train_data, goal_attr, attrs_to_use)
    acc = claculate_acc(forest, train_data, test_data)
    accuracies_train.append(acc[0])
    accuracies_test.append(acc[1])
    nodes.append(forest.get_tree_size())

# accuracies_test.reverse()
# accuracies_train.reverse()





plt.plot(nodes, accuracies_test, '-b', label='test')
plt.plot(nodes, accuracies_train, '-g', label='train')
plt.legend(loc="upper left")
plt.xlabel('Nodes')
plt.ylabel('Accuracy')
plt.show()
