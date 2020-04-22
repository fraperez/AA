from id3 import ID3, shanon_gain, gini_gain
from random_forest import RandomForest
import argparse
import random
import pandas as pd
# import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

'''
basic console to interact with the program
'''

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--predictor", type=str, default="decision_tree",
	help="predictor to use [decision_tree, random_forest]")
ap.add_argument('-g', '--gain', type=str, default='shanon',
    help='gain function to use [shanon, gini]')
ap.add_argument('-ntree', '--num_tree', type=int, default=10,
    help='number of trees to use')
ap.add_argument('-ndata', '--n_data', type=float, default=0.8,
    help='percentage of data to use in each tree [needed for random forest')
ap.add_argument('-nattr', '--n_attr', required=True, type=int, default=3,
    help='number of attr to analyze (depth of tree) default = 3')
ap.add_argument('-t', '--test', required=True, type=float, default=0.3,
    help='percentage of data to use as test')
# args = vars(ap.parse_args())


def load_data(path, separator=',', encoding='ISO-8859-1'):
    df = pd.read_csv(path, sep= separator, encoding=encoding)
    return df


def train_test_split(df, test_percentage):
    '''
    Split a dataframe in train and test given a percentage
    percentage should be between [0, 1]
    '''
    count = len(df)
    train_count = int(test_percentage * count)
    total_indexes = df.index.tolist()
    train_indexes = random.sample(total_indexes, k=train_count)
    train_df = df.query('index in {}'.format(train_indexes))
    test_df = df.drop(train_indexes)
    return train_df, test_df

if __name__ == '__main__':
    data = load_data(args['dataset'], '\t', encoding='utf-8')
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

    train_data, test_data = train_test_split(data, args['test'])


    # now we build each method
    if args['predictor'] == 'random_forest':
        predictior = RandomForest(shanon_gain)
        predictior.train(args['num_tree'], args['n_data'], args['n_attr'], train_data, goal_attr, attrs)
    else:
        gain_func = shanon_gain if args['gain'] == 'shanon' else gini_gain
        id3 = ID3(gain_func)
        predictior = id3.train(train_data, goal_attr, attrs)
    # now we test
    predictions = []
    # make predictions
    for index, case in test_data.iterrows():
        case = case.to_dict()
        predictions.append(predictior.predict(case))

    # build confusion matrix
    labels = [1, 0]
    conf_matrix = confusion_matrix(test_data.Survived.to_list(), predictions, labels=labels)
    df_cm = pd.DataFrame(conf_matrix, index=['survived', 'not survived'], columns=['survived', 'not survived'])
    sn.heatmap(df_cm, annot=True)

    accuracy = (conf_matrix[0,0] + conf_matrix[1, 1]) / float(len(test_data))
    print(f'[ACCURACY] {accuracy}  (max 1)')
    plt.show()
