import pandas as pd
import numpy as np


class DecisionTreeID3:
    def __init__(self):
        self.tree = {}
        self.gains = []
        self.classes = {}
        self.targets = []

    def _entropy(self, p):
        if p != 0:
            return -p * np.log2(p)
        else:
            return 0

    def _getSubTable(self, df, node, value):
        return df[df[node] == value].reset_index(drop=True)

    def _info_gain(self, feature):
        gain = 0
        nData = len(feature)

        values = {}
        for v in feature:  # Find unique values the feature can take and num occurrences
            if v in values:
                values[v] += 1
            else:
                values[v] = 1

        count = dict.fromkeys(self.classes, 0)
        for d in range(nData):
            count[self.targets[d]] += 1

        numValues = len(values)
        entropy = np.zeros(numValues + 1)  # Extra row for the total entropy
        valueIndex = 0

        for value in values:  # Loop through each value feature can take
            classCount = dict.fromkeys(self.classes,
                                       0)  # Define Dict to hold number of value instances associated with each class
            totalClassCount = 0
            for d in range(nData):  # Loop through all data
                if feature[d] == value:  # If found the current value, update the class dict
                    classCount[self.targets[d]] += 1
                    totalClassCount += 1

            for c in classCount:
                entropy[valueIndex] += self._entropy(classCount.get(c) / totalClassCount)
                if valueIndex == numValues - 1:
                    entropy[numValues] += self._entropy(count.get(c) / nData)

            gain += (values.get(value) / nData) * entropy[valueIndex]

            valueIndex += 1
        gain = entropy[numValues] - gain
        # print("Information gain for feature: {:.3f}".format(gain))
        self.gains.append(gain)

    def fit(self, inputs, targets):
        self.gains = []
        featureNames = [*inputs.keys()]
        nData = len(inputs.get(featureNames[0]))
        nFeatures = len(featureNames)

        self.targets = targets
        for t in targets:
            if t not in self.classes:
                self.classes[t] = 1
            else:
                self.classes[t] += 1

        for i in range(nFeatures):
            self._info_gain(inputs[featureNames[i]])
        bestFeature = np.argmax(self.gains)
        bestName = featureNames[bestFeature]
        # print(inputs[featureNames[bestFeature]])

        tree = {bestName: {}}
        values = set(inputs[bestName])
        df = inputs
        df['T'] = targets

        for value in values:
            subTable = self._getSubTable(df, bestName, value)
            clValue, counts = np.unique(subTable['T'], return_counts=True)
            newTargets = subTable['T']
            newInputs = subTable[featureNames]

            if len(counts) == 1:
                tree[bestName][value] = clValue[0]
            else:
                tree[bestName][value] = self.fit(newInputs, newTargets)

        # for g in self.gains:
        #     print("Information gain for feature: {:.3f}".format(g))
        self.tree = tree
        return tree

    def _findPath(self, tree, inputs):
        if isinstance(tree, str):
            return tree
        # Otherwise, we are not yet on a leaf node.
        # Call predict method recursively until we get to a leaf node.
        else:
            for key in tree.keys():
                if key in [*inputs.keys()]:
                    for value in tree.get(key):
                        if value == inputs.get(key):
                            return self._findPath(tree.get(key), inputs)
                elif key in [*inputs.values()]:
                    return self._findPath(tree.get(key), inputs)
            # Otherwise, return the most common class
            # return the mode label of examples with other attribute values for the current attribute

        return max(self.classes, key=self.classes.get)

    def predict(self, inputs):
        print("Predicting using decision tree:\n", self.tree)
        result = self._findPath(self.tree, inputs)

        return self.tree, result

