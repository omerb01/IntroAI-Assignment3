import hw3_utils
import utils


class knn_classifier(hw3_utils.abstract_classifier):

    def __init__(self, k, data, labels):
        self.k = k
        self.data = data
        self.labels = labels

    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of feature to classify
        :return: a tagging of the given features (1 or 0)
        '''

        labels_copy = self.labels.copy()
        distance_list = [utils.euclidian_distance(features, obj) for obj in self.data]

        counter = 0
        for i in range(self.k):
            # get the min value index and check if it's classified
            index = distance_list.index(min(distance_list))
            if labels_copy[index] == True:
                counter += 1
            distance_list.pop(index)
            labels_copy.pop(index)
        if float(counter / self.k) > 0.5:
            return 1  # there are more sick people (we are closer to more than half True objs in classify matrix)
        return 0


class knn_factory(hw3_utils.abstract_classifier_factory):

    def __init__(self, k):
        self.k = k

    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents  the labels that the classifier will be trained with
        :return: abstruct_classifier object
        '''

        return knn_classifier(self.k, data, labels)
