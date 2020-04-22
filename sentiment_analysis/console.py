from knn import KNN
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to input dataset")
ap.add_argument('-k', '--neighbour', required=True, type=int, default=3,
                help='Number of neighbours to use (must be odd) [default = 3]')
ap.add_argument('-t', '--test', required=True, type=float, default=0.3,
                help='Percentage of data to use as test')
ap.add_argument('-w', '--weighted', type=str2bool, nargs='?', const=True, default=False,
                help='Indicates wheather or not knn use a weighted distance.')
args = vars(ap.parse_args())


def load_data(path, separator=',', encoding='ISO-8859-1'):
    df = pd.read_csv(path, sep=separator, encoding=encoding)
    return df


def train_test_split(df, train_percentage):
    '''
    Split a dataframe in train and test given a percentage
    percentage should be between [0, 1]
    '''
    count = len(df)
    train_count = int(train_percentage * count)
    total_indexes = df.index.tolist()
    train_indexes = random.sample(total_indexes, k=train_count)
    train_df = df.query('index in {}'.format(train_indexes))
    test_df = df.drop(train_indexes)
    return train_df, test_df


data = load_data(args['dataset'], ';')
# How many words avg does 1 star ratings have?
size_words = []
d = data[data['Star Rating'] == 1]
for idx, row in d.iterrows():
    size_words.append(row['wordcount'])
print('Average word count is => {}'.format(sum(size_words)/ float(len(d))))

data = data.fillna(method='bfill', axis=1)
# now we bring categories to numbers and work around the NaNs

data['textSentiment'] = data['textSentiment'].astype('category')
data['titleSentiment'] = data['titleSentiment'].astype('category')
# get categorical columns and make them numeric
cat_columns = data.select_dtypes(['category']).columns
data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)

train_set, test_set = train_test_split(data, 1 - args['test'])

knn = KNN(args['neighbour'], 'Star Rating', args['weighted'])
knn.train(train_set, ['sentimentValue', 'textSentiment',
                      'titleSentiment', 'wordcount'])


# now we test the results

predictions = []
expected = test_set['Star Rating'].to_list()

for index, row in test_set.iterrows():
    predictions.append(knn.predict(row.to_dict()))

# build confusion matrix
labels = set(expected)
labels = list(labels)
conf_matrix = confusion_matrix(
    np.array(expected), np.array(predictions), labels=labels)
df_cm = pd.DataFrame(conf_matrix, index=list(labels), columns=list(labels))
sn.heatmap(df_cm, annot=True)

plt.show()


# accuracy
acc = 0
for i in range(len(conf_matrix)):
    acc += conf_matrix[i, i]

acc /= len(test_set)

print(f'[ACCURACY] ====> {acc}')
