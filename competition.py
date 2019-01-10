import classifier
from hw3_utils import load_data
import numpy as np


def evaluate_comp(classifier_factory, k, feature_index_to_ignore):
    # load all folds
    folds = [load_data('ecg_fold_' + str(i + 1) + '.pickle') for i in range(k)]
    folds = [(np.delete(data, feature_index_to_ignore, 1), labels, test) for data, labels, test in folds]

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
            if res_list[j] == 1 and test_labels[j] == True:
                test_true_positive += 1
            elif res_list[j] == 1 and test_labels[j] == False:
                test_false_positive += 1
            elif res_list[j] == 0 and test_labels[j] == True:
                test_false_negative += 1
            elif res_list[j] == 0 and test_labels[j] == False:
                test_true_negative += 1

        accuracies.append((test_true_positive + test_true_negative) / N)
        errors.append((test_false_positive + test_false_negative) / N)

    return np.average(accuracies), np.average(errors)


if __name__ == '__main__':
    data = load_data()
    for feature_index_to_ignore in range(len(data[0][0])):
        knn_fac = classifier.knn_factory(1)
        accuracy, error = evaluate_comp(knn_fac, 2, feature_index_to_ignore)
        line = str(accuracy) + ',' + str(error)
        print(line)
