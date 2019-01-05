data_path = r'data/Data.pickle'
import pickle
import numpy as np


def load_data(path=r'data/Data.pickle'):
    '''
    return the dataset that will be used in HW 3
    prameters:
    :param path: the path of the csv data file (default value is data/ecg_examples.data)

    :returns: a tuple train_features, train_labels ,test_features
    features - a numpy matrix where  the ith raw is the feature vector of patient i.
    '''
    with open(path, 'rb') as f:
        train_features, train_labels, test_features = pickle.load(f)
    return train_features, train_labels, test_features


def write_prediction(pred, path='results.data'):
    '''
    write the prediction of the test set into a file for submission
    prameters:
    :param pred: - a list of result the ith entry represent the ith subject (as integers of 1 or 0, where 1 is a healthy patient and 0 otherwise)
    :param path: - the path of the csv data file will be saved to(default value is res.data)
    :return: None
    '''
    output = []
    for l in pred:
        output.append(l)
    with open(path, 'w') as f:
        f.write(', '.join([str(x) for x in output]) + '\n')


class abstract_classifier_factory:
    '''
    an abstruct class for classifier factory
    '''

    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents  the labels that the classifier will be trained with
        :return: abstruct_classifier object
        '''
        raise Exception('Not implemented')


class abstract_classifier:
    '''
        an abstruct class for classifier
    '''

    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of feature to classify
        :return: a tagging of the given features (1 or 0)
        '''
        raise Exception('Not implemented')


def euclidian_distance(object1, object2):
    result = map(lambda x, y: (x - y) ** 2, object1, object2)
    return sum(result) ** 0.5


def split_crosscheck_groups(dataset, num_folds):
    # extract data and labels from data.
    data = dataset[0]
    labels = dataset[1]

    true_list = []
    false_list = []
    folds = []

    data_size = len(data)

    for i in range(data_size):
        if labels[i] == True:
            true_list.append(data[i])
        else:
            false_list.append(data[i])

    true_list_size = len(true_list)
    false_list_size = len(false_list)

    # shuffle the true and false lists.
    np.random.shuffle(true_list)
    np.random.shuffle(false_list)

    '''
    there will be cast to int which will keep equal amount of true and false classifiers 
    in each fold and will not treat extra classifiers to keep the ratio equal
    '''

    num_of_true_in_fold = int(true_list_size / num_folds)
    num_of_false_in_fold = int(false_list_size / num_folds)

    for i in range(num_folds):
        fold = []
        for _ in range(num_of_true_in_fold):
            fold.append((true_list.pop(0), True))  # add the relative number of true in each folder
        for _ in range(num_of_false_in_fold):
            fold.append((false_list.pop(0), False))  # add the relative number of false in each folder
        np.random.shuffle(fold)
        folds.append(fold)

    for i in range(num_folds):
        file_name = 'data/ecg_fold_' + str(i + 1) + '.pickle'
        with open(file_name, 'wb') as fold_data_file:
            # break the tuples into the desired matrixs
            matrix1 = np.array([obj[0] for obj in folds[i]])
            matrix2 = [obj[1] for obj in folds[i]]
            matrix3 = np.array([[]])
            p = pickle.dumps((matrix1, matrix2, matrix3))
            fold_data_file.write(p)


def evaluate(classifier_factory, k):
    # load all folds
    folds = [load_data('data/ecg_fold_' + str(i + 1) + '.pickle') for i in range(k)]

    accuracies = []
    errors = []
    for i in range(k):
        # choose 1 group to be test group all others will be train groups
        test_data = folds[i][0]
        test_labels = folds[i][1]

        train_folds = [folds[j][0] for j in range(k) if j != i]
        train_data = []
        for train_fold in train_folds:
            for features in train_fold:
                train_data.append(features)
        train_data = np.array(train_data)  # converstion to np array
        train_labels = []
        for j in range(k):
            if j != i:
                for train_label in folds[j][1]:
                    train_labels.append(train_label)

        # run groups with classifier
        classifier = classifier_factory.train(train_data, train_labels)
        res_list = [classifier.classify(features) for features in test_data]

        '''
        classify each classify result when True means subject is actually sick
        and res_list = 0 means classified as sick 
        '''
        test_false_positive = 0
        test_false_negative = 0
        test_true_positive = 0
        test_true_negative = 0
        N = len(res_list)
        for j in range(N):
            if res_list[j] == 0 and test_labels[j] == True:
                test_true_positive += 1
            elif res_list[j] == 0 and test_labels[j] == False:
                test_false_positive += 1
            elif res_list[j] == 1 and test_labels[j] == True:
                test_false_negative += 1
            elif res_list[j] == 1 and test_labels[j] == False:
                test_true_negative += 1

        accuracies.append((test_true_positive + test_true_negative) / N)
        errors.append((test_false_positive + test_false_negative) / N)

    return np.average(accuracies), np.average(errors)
