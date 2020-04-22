import math
import pandas as pd
from decision_tree import DecisionTreee, Node, Vertex
import operator
'''
This is an ID3 implmenetation.
'''


class ID3(object):

    def __init__(self, gain_function):
        self.gain = gain_function

    def train(self, train_data, objective_attr, attrs):
        '''
        build a decsion tree based on the gain function and the train data set
        input:
            - objective_attr: dict containing the attribute that we want to classify and the unierse.
            - train_data: training dtaset.
            - attrs: dict with other attributes which the tree can expect and their universe.

        output: a DecisionTree object
        '''
        objective_universe = list(objective_attr.values()).pop()
        objective_attr = list(objective_attr.keys()).pop()
        # sanity checks
        for value in objective_universe:

            if len(train_data[getattr(train_data, objective_attr) == value]) == len(train_data):
                return DecisionTreee(Node(value, objective_universe))

        if len(attrs) == 0:
            value = get_mode(getattr(train_data, objective_attr))
            return DecisionTreee(Node(value, objective_universe))

        # now, if we are here. There are multple attrs to evaluate.
        # we calculate the gain for each one and declare the highest as the root.

        tree_core = self._generate_tree(
            train_data, objective_attr, objective_universe, attrs, None)

        tree = DecisionTreee(tree_core)

        return tree

    def _generate_tree(self, train_data, objective_attr, objective_universe, attrs, parent):
        '''
        Generate a m-ary tree without header
        '''
        # sanity checks
        for value in objective_universe:

            if len(train_data[getattr(train_data, objective_attr) == value]) == len(train_data):
                return Node(value, objective_universe, parent)

        if len(attrs) == 0:
            value = get_mode(getattr(train_data, objective_attr))
            return Node(value, objective_attr, parent)

        # if we are here, then this is not a leaf node
        max_attr = self.get_max_gain_attr(
            train_data, objective_attr, objective_universe, attrs)
        values = attrs.pop(max_attr, None)
        node = Node(max_attr, attrs.get(max_attr), parent)

        # generate all children trees
        for value in values:
            to_node = self._generate_tree(train_data[getattr(
                train_data, max_attr) == value], objective_attr, objective_universe, attrs, node)
            vertex = Vertex(value, node, to_node)
            node.add_vertex(vertex)

        return node

    def get_max_gain_attr(self, train_data, objective_attr, objective_universe, attrs):
        max_attr = None
        gain = 0.0
        for attr, universe in attrs.items():
            new_gain = self.gain(train_data, objective_attr,
                                 objective_universe, attr, universe)
            if new_gain > gain or max_attr == None:
                gain = new_gain
                max_attr = attr

        return max_attr


def get_mode(array):
    if len(array) == 0:
        print('[WARNING] calculating mode of empty array')
        return None

    values = {}
    for value in array:
        values[value] = values.get(value, 0) + 1

    return max(values.items(), key=operator.itemgetter(1))[0]


def entropy(dataset, goal_attr, universe):
    '''
    Calculates the Shanon entropy for a given dataset taking as expected output the goal_attr of each individual.

    Entropy(S) = - (sum(P[goal_attr = i] * log2(P[goal_attr = i]) ))

    Paras:
        - dataset: pandas df of examples, each column should contain an attr
        - goal_attr : name of attr to classify in dataset objts.
        - universe: possible values of given goal_attr.
    '''
    if len(dataset) == 0:
        return 0

    shannon_entropy = 0
    for value in universe:
        frequency = len(
            dataset[getattr(dataset, goal_attr) == value]) / float(len(dataset))
        if frequency > 0:
            shannon_entropy -= frequency * math.log2(frequency)

    return shannon_entropy


def shanon_gain(cls, goal_attr, goal_universe, attr, attr_universe):
    '''
    Shanons defintion of gain is based on the shanon's entropy formula:
    S = dataset
    A = attribute
    Gain(S, A): H(S) - sum( |Sv|/ |S| * H(Sv) ) [sum over the universe of A]

    Params:
    - cls: the data frame where columns represent attrs.
    - goal_attr: the goal_attr for the tree
    - goal_universe: possible classes for goal attr
    - attr: the specific attr to analyze
    - attr_univers: the list of possible values of attr
    '''

    gain = entropy(cls, goal_attr, goal_universe)

    for value in attr_universe:
        frequency = len(cls[getattr(cls, attr) == value]) / float(len(cls))
        etpy = entropy(cls[getattr(cls, attr) == value],
                       goal_attr, goal_universe)
        gain -= frequency * etpy

    return gain


def gini_gain(cls, goal_attr, goal_universe, attr, attr_universe):
    '''
    Gini gain is based on the gini index
    Params:
    - cls: the data frame where columns represent attrs.
    - goal_attr: the goal_attr for the tree
    - goal_universe: possible classes for goal attr
    - attr: the specific attr to analyze
    - attr_univers: the list of possible values of attr

    Obs:
    As the gini gain works inverse to the entropy gain function. We deliverately invert the gini gain
    '''

    gain = 0

    for value in attr_universe:
        frequency = len(cls[getattr(cls, attr) == value]) / float(len(cls))
        gini = gini_index(cls[getattr(cls, attr) == value],
                       goal_attr, goal_universe)
        gain += frequency * gini

    return 1.0/gain if gain > 0 else 0.0



def gini_index(dataset, goal_attr, universe):
    '''
    Calculate gini index for a given attr
    Params:
    - dataset : dataframe with all columns and examples to take frequency.
    - goal_attr: the column to base calcualtions on
    '''

    if len(dataset) == 0:
        return 1

    gini = 1

    for value in universe:
        frequency = len(
            dataset[getattr(dataset, goal_attr) == value]) / float(len(dataset))

        gini -= frequency ** 2

    return gini


