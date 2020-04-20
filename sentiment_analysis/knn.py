from heapq import heappush, heappop
import math
import operator

'''
KNN implementation using pandas and numpy for matrix representation
'''


class KNN(object):
    def __init__(self, k_neighbours, goal_attr, weighted=False):
        '''
        Params:
        - k_neighbours: expected clases to be output, can't be even amount
        - goal_attr: column to find label
        '''
        if k_neighbours % 2 == 0:
            print('[ERROR] trying to use an even amount of neighbours')
            raise IndexError
        self.k_neighbours = k_neighbours
        self.goal_attr = goal_attr
        self.weighted = weighted

    def train(self, trainset, attrs):
        '''
        train the classificator using the dataset given.

        Params:
        - trainset: DataFrame containing a example per row
        - goal_attr:  the desire output column
        - attrs: list containing all analyzable columns
        '''
        self.attrs = attrs
        self.samples = []

        # classify each example based on the goal attr
        for index, row in trainset.iterrows():
            self.samples.append((row.to_dict(), self._label(row)))

    def _label(self, case):
        '''
        Define corresponding label for a given example
        '''
        return int(getattr(case, self.goal_attr))

    def predict(self, case):
        '''
        Classif a new example by k_nearest
        Params:
        - case: dict containing each attr value
        '''

        distances_heap = []
        for sample in self.samples:
            heappush(distances_heap, (self.distance_to(
                sample[0], case), sample[1]))

        # now we get the k first neighbours
        neighbour_labels = {}
        for i in range(self.k_neighbours):
            # we get the lable for the neighbour
            label = heappop(distances_heap)
            # if the case equals a known result, we cosnider it belongs to that label
            if label[0] == 0:
                return label[1]

            weight = 1/label[0] if self.weighted else 1

            neighbour_labels[label[1]] = neighbour_labels.get(
                label, 0) + weight

        # return the most frequent one

        return max(neighbour_labels.items(), key=operator.itemgetter(1))[0]

    def distance_to(self, from_vec, to_vec):
        '''
        Calculate euclidean distance between from_vec and to_vec
        we use attrs defined during training
        '''

        distance = 0.0

        for col in self.attrs:
            distance += (from_vec.get(col, 0.0) - to_vec.get(col, 0.0)) ** 2

        return math.sqrt(distance)

