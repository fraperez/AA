from random import choices, sample
from id3 import ID3, shanon_gain, get_mode
import operator


class RandomForest(object):

    def __init__(self, gain_function):
        '''
        Define what gain function should be used.
        '''
        self.gain = gain_function

    def train(self, n_tree, n_data, n_attr, dataset, goal_attr, attrs):
        '''
        To train a random forest, we build each tree and the decide upond the most common answer.
        params:
        - n_tree: number of trees to build
        - n_data: percentage of data to input each tree to train.
        - dataset: datframe with all data.
        - n_attr: number of attributes to consider in each individual tree [1, n].
        - goal_attr: dict containing the name (key) and universe(value) of the output.
        - attrs: dict with name(key) and universe (value) of each attr expected in the dataset.

        '''
        self.forest = []

        # build each tree
        for i in range(n_tree):
            # get m data with replace
            mini_batch = self._train_split(dataset, n_data)
            #  now we generate tree
            id3 = ID3(self.gain)
            attrs_batch = sample(list(attrs.items()), k=n_attr)
            attrs_batch = dict(attrs_batch)
            tree = id3.train(mini_batch, goal_attr, attrs_batch)
            self.forest.append(tree)
        

    def _train_split(self, df, percentage):
        '''
        Sample dataframe with replacement
        Params:
        - df: dataframe
        - percentage: what percentage of the df should be sample [0, 1]
        '''
        count = len(df)
        train_count = int(percentage * count)
        total_indexes = df.index.tolist()
        train_indexes = choices(total_indexes, k=train_count)
        train_df = df.query('index in {}'.format(train_indexes))

        return train_df

    def predict(self, case):
        
        predictions = []

        for tree in self.forest:
            individual_prediciton = tree.predict(case)
            predictions.append(individual_prediciton)
        
        # get the most voted class
        return get_mode(predictions)
        
    