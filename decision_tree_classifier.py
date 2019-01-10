from sklearn import tree
import hw3_utils


class DecTreeClassifier(hw3_utils.abstract_classifier):

    def __init__(self, decision_tree):
        self.decision_tree = decision_tree

    def classify(self, features):
        return self.decision_tree.predict([features])[0]


class DecTreeClassifierFactory(hw3_utils.abstract_classifier_factory):

    def train(self, data, labels):
        decision_tree = tree.DecisionTreeClassifier(criterion="entropy")
        decision_tree = decision_tree.fit(data, labels)
        return DecTreeClassifier(decision_tree)
