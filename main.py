import utils
import classifier
import decision_tree_classifier


def run_knn():
    ks = [1, 3, 5, 7, 13]

    with open('experiments6.csv', 'w') as _:
        pass

    with open('experiments6.csv', 'w+') as result_file:
        for k in ks:
            knn_fac = classifier.knn_factory(k)
            accuracy, error = utils.evaluate(knn_fac, 2)
            line = str(k) + ',' + str(accuracy) + ',' + str(error) + '\n'
            result_file.write(line)
            print("finished: ", k)

    print("FINISHED")


def run_decision_tree():
    decision_tree_factory = decision_tree_classifier.DecTreeClassifierFactory()
    accuracy, error = utils.evaluate(decision_tree_factory, 2)
    line = str(accuracy) + ',' + str(error)
    print(line)


if __name__ == '__main__':
    run_decision_tree()
