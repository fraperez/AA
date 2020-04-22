
'''
DecisionTree represent an m-ary tree created by ID3
'''


class DecisionTreee(object):
    def __init__(self, root_node=None):
        self.root = root_node


    def get_tree_size(self):
        '''
        Returns the amount of nodes used.
        '''
        return self._tree_size(self.root)

    def _tree_size(self, current):
        if current == None:
            return 0

        nodes = 1
        for child in current.get_children():
            nodes += self._tree_size(child.get_to())

        return nodes

    def predict(self, case):
        '''
        Predict a value for the given case
        Arguments:
        - case: a dict with all attrs and the fiven value
        '''
        return self._predict(case, self.root)

    def _predict(self, case, node):
        if node.is_leaf():
            return node.get_value()

        next_node = node.get_children()
        for vertex in next_node:
            if vertex.get_value() == case.get(node.get_value()):
                next_node = vertex.get_to()
                return self._predict(case, next_node)


class Node(object):
    def __init__(self, value, universe, parent=None):
        '''
        A node must have a given value which is contained in the universe (possible values)
        and have a parent node (root node have parent  = None).
        '''
        self.value = value
        self.universe = universe
        self.parent = parent
        self.children_vertexes = []

    def add_vertex(self, child_vertex):
        self.children_vertexes.append(child_vertex)

    def get_children(self):
        return self.children_vertexes

    def get_parent(self):
        return self.parent

    def get_value(self):
        return self.value

    def get_universe(self):
        return self.universe

    def is_leaf(self):
        return len(self.children_vertexes) == 0


class Vertex(object):
    def __init__(self, value, from_node, to_node):
        self.value = value
        self.from_node = from_node
        self.to_node = to_node

    def get_value(self):
        return self.value

    def get_from(self):
        return self.from_node

    def get_to(self):
        return self.to_node
