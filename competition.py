import hw3_utils
import classifier
import numpy as np


def evaluate_comp(classifier_factory, folds, feature_index_to_ignore):
    k = len(folds)
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

    return feature_index_to_ignore, np.average(accuracies), np.average(errors)


def run_research():
    # a research to exam which feature is best to ignore for highest accuracy

    knn_fac = classifier.knn_factory(1)
    folds = [hw3_utils.load_data('ecg_fold_' + str(i + 1) + '.pickle') for i in range(2)]
    features_num = len(folds[0][0][0])

    max_accuracy_feature = None
    for run_num in range(1, 8):

        if max_accuracy_feature is not None:
            folds = [(np.delete(data, max_accuracy_feature, 1), labels, test) for data, labels, test in folds]
            features_num = len(folds[0][0][0])

        results = [evaluate_comp(knn_fac, folds, feature) for feature in range(features_num)]
        max_accuracy_feature = max(results, key=lambda item: item[1])[0]

        with open('my_experiments' + str(run_num) + '.csv', 'w+') as result_file:
            for feature, accuracy, error in results:
                line = str(feature) + ',' + str(accuracy) + ',' + str(error) + '\n'
                result_file.write(line)


def run_my_classify():
    # predicts the test data with specific features

    data, labels, tests = hw3_utils.load_data()

    # features list to ignore that came from the research before
    features_to_ignore = [90, 23, 90, 103, 36]

    for feature in features_to_ignore:
        data = np.delete(data, feature, 1)
        tests = np.delete(tests, feature, 1)

    clf = classifier.knn_factory(1).train(data, labels)
    results = [clf.classify(test) for test in tests]

    hw3_utils.write_prediction(results)


if __name__ == '__main__':
    run_my_classify()
