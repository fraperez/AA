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

    def train(self, train_data, objective_attr, objective_universe, attrs):
        '''
        build a decsion tree based on the gain function and the train data set
        input:
            - objective_attr: the attribute that we want to classify.
            - train_data: training dtaset.
            - attrs: dict with other attributes which the tree can expect and their universe.
        
        output: a DecisionTree object
        '''

        # sanity checks
        for value in objective_universe:
        
            if len(train_data[getattr(train_data, objective_attr) == value]) == len(train_data):
                return DecisionTreee(Node(value))


        if len(attrs) == 0:
            value = get_mode(getattr(train_data, objective_attr))
            return DecisionTreee(Node(value))

        # now, if we are here. There are multple attrs to evaluate.
        # we calculate the gain for each one and declare the highest as the root.

        tree_core = self._generate_tree(train_data, objective_attr, objective_universe, attrs, None)

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
        max_attr = self.get_max_gain_attr(train_data, objective_attr, objective_universe, attrs)
        values = attrs.pop(max_attr, None)
        node = Node(max_attr, attrs.get(max_attr), parent)

        # generate all children trees
        for value in values:
            to_node = self._generate_tree(train_data[getattr(train_data, max_attr) == value], objective_attr, objective_universe, attrs, node)
            vertex = Vertex(value, node, to_node)
            node.add_vertex(vertex)

        return node
        




        

    def get_max_gain_attr(self, train_data, objective_attr, objective_universe, attrs):
        max_attr = None
        gain = 0.0
        for attr, universe in attrs.items():
            new_gain = self.gain(train_data, objective_attr, objective_universe, attr, universe)
            if new_gain > gain:
                gain = new_gain
                max_attr = attr
        return max_attr


def get_mode(array):
    if len(array) == 0:
        print('[WARNING] calculating mode of empty array')
        return None

    values = {}
    for value in array:
        values[value] = values.get(value, 0) +  1

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
        frequency = len(dataset[getattr(dataset, goal_attr) == value]) / float(len(dataset))
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

    for value in  attr_universe:
        frequency = len(cls[getattr(cls, attr) == value]) / float(len(cls))
        etpy  = entropy(cls[getattr(cls, attr) == value], goal_attr, goal_universe)
        gain -= frequency * etpy


    return gain




################################ TESTS ######################################################

def test_entropy():
    attr = 'play'
    df = {
            attr: [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
        }

    df = pd.DataFrame(df)
    result = entropy(df, attr, [0,1])
    expected = (- (9.0/14) * math.log2(9/14.0) - (5/14.0) * math.log2(5/14.0))
    assert result == expected


def test_shanon_gain():
    goal_attr = 'play'
    attr = 'wind'
    attr_universe = ['strong', 'weak']
    
    df = {
            goal_attr: [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
            attr : ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'strong']
        }
    df = pd.DataFrame(df)
    result = shanon_gain(df, goal_attr, [0, 1], attr, attr_universe)
    expected = (- (9.0/14) * math.log2(9/14.0) - (5/14.0) * math.log2(5/14.0))
    expected += -(8/14.0) * 0.8112781244591328 - (6/14.0) * 1
    assert result == expected



def test_id3():
    goal_attr = 'play'
    attr = 'wind'
    attr_universe = ['strong', 'weak']

    attr_2 = 'wheather'
    attr_2_univserse = ['sunny', 'cloudy', 'rainny']

    attr_3 = 'temperature'
    attr_3_universe = ['cold', 'norm', 'hot']

    attr_4 = 'humidity'
    attr_4_universe = ['norm', 'high']
    

    df = {
            goal_attr: [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
            attr : ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak'],
            attr_2 : ['sunny', 'sunny', 'cloudy', 'rainny', 'rainny', 'rainny', 'cloudy', 'sunny', 'sunny', 'rainny', 'sunny', 'cloudy', 'cloudy'],
            attr_3: ['hot', 'hot', 'hot', 'norm', 'cold', 'cold', 'cold', 'norm', 'cold', 'norm', 'norm', 'norm', 'hot'],
            attr_4: ['high', 'high', 'high', 'high', 'norm', 'norm', 'norm', 'high', 'norm', 'norm', 'norm', 'high', 'norm']

        }

    df = pd.DataFrame(df)
    id3 = ID3(shanon_gain)
    tree = id3.train(df, goal_attr, [0, 1], {attr: attr_universe, attr_2: attr_2_univserse, attr_3: attr_3_universe, attr_4: attr_4_universe})
    
    case = {
        attr : 'strong', 
        attr_2: 'rainny',
        attr_3: 'norm',
        attr_4: 'high'
    }

    result = tree.predict(case)
    expected = 0

    assert result == expected


test_entropy()

test_shanon_gain()

test_id3()