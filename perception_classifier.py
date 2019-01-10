import hw3_utils
from sklearn.linear_model import Perceptron


class PercClassifier(hw3_utils.abstract_classifier):

    def __init__(self, perception):
        self.perception = perception

    def classify(self, features):
        return self.perception.predict([features])[0]


class PercClassifierFactory(hw3_utils.abstract_classifier_factory):

    def train(self, data, labels):
        clf = Perceptron()
        clf = clf.fit(data, labels)
        return PercClassifier(clf)
