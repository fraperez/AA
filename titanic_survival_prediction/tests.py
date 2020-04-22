import math
import pandas as pd
from decision_tree import DecisionTreee, Node, Vertex
import operator
from id3 import ID3, shanon_gain, entropy, gini_index, gini_gain
from random_forest import RandomForest

################################ TESTS ######################################################


def test_entropy():
    attr = 'play'
    df = {
        attr: [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    }

    df = pd.DataFrame(df)
    result = entropy(df, attr, [0, 1])
    expected = (- (9.0/14) * math.log2(9/14.0) - (5/14.0) * math.log2(5/14.0))
    assert result == expected


def test_shanon_gain():
    goal_attr = 'play'
    attr = 'wind'
    attr_universe = ['strong', 'weak']

    df = {
        goal_attr: [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
        attr: ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong',
               'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'strong']
    }
    df = pd.DataFrame(df)
    result = shanon_gain(df, goal_attr, [0, 1], attr, attr_universe)
    expected = (- (9.0/14) * math.log2(9/14.0) - (5/14.0) * math.log2(5/14.0))
    expected += -(8/14.0) * 0.8112781244591328 - (6/14.0) * 1
    assert result == expected

def test_gini_index():
    df = {
        'class': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
        'attr1': [0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
        'attr2': [33, 54, 56, 42, 50, 55, 31, -4, 77, 49]
    }
    df = pd.DataFrame(df)
    result = gini_gain(df, 'class', ['A', 'B'], 'attr1', [0, 1])
    expected = 0.4166666666666667

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
        attr: ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak'],
        attr_2: ['sunny', 'sunny', 'cloudy', 'rainny', 'rainny', 'rainny', 'cloudy', 'sunny', 'sunny', 'rainny', 'sunny', 'cloudy', 'cloudy'],
        attr_3: ['hot', 'hot', 'hot', 'norm', 'cold', 'cold', 'cold', 'norm', 'cold', 'norm', 'norm', 'norm', 'hot'],
        attr_4: ['high', 'high', 'high', 'high', 'norm', 'norm',
                 'norm', 'high', 'norm', 'norm', 'norm', 'high', 'norm']

    }

    df = pd.DataFrame(df)
    id3 = ID3(shanon_gain)
    tree = id3.train(df, {goal_attr: [0, 1]}, {
                     attr: attr_universe, attr_2: attr_2_univserse, attr_3: attr_3_universe, attr_4: attr_4_universe})

    case = {
        attr: 'strong',
        attr_2: 'rainny',
        attr_3: 'norm',
        attr_4: 'high'
    }

    result = tree.predict(case)
    expected = 0

    assert result == expected


def test_random_forest():
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
        attr: ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak'],
        attr_2: ['sunny', 'sunny', 'cloudy', 'rainny', 'rainny', 'rainny', 'cloudy', 'sunny', 'sunny', 'rainny', 'sunny', 'cloudy', 'cloudy'],
        attr_3: ['hot', 'hot', 'hot', 'norm', 'cold', 'cold', 'cold', 'norm', 'cold', 'norm', 'norm', 'norm', 'hot'],
        attr_4: ['high', 'high', 'high', 'high', 'norm', 'norm',
                 'norm', 'high', 'norm', 'norm', 'norm', 'high', 'norm']

    }

    df = pd.DataFrame(df)
    forest = RandomForest(shanon_gain)
    forest.train(10, 1, 4, df, {goal_attr: [0, 1]}, {
                 attr: attr_universe, attr_2: attr_2_univserse, attr_3: attr_3_universe, attr_4: attr_4_universe})

    case = {
        attr: 'strong',
        attr_2: 'rainny',
        attr_3: 'norm',
        attr_4: 'high'
    }

    result = forest.predict(case)
    expected = 0

    assert result == expected


test_entropy()

test_shanon_gain()

test_id3()

test_random_forest()

test_gini_index()
