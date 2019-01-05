import hw3_utils
import classifier

data = hw3_utils.load_data()
ks = [1, 3, 5, 7, 13]

with open('experiments6.csv', 'w') as _:
    pass

with open('experiments6.csv', 'w+') as result_file:
    for k in ks:
        knn_fac = classifier.knn_factory(k)
        accuracy, error = hw3_utils.evaluate(knn_fac, 2)
        line = str(k) + ',' + str(accuracy) + ',' + str(error) + '\n'
        result_file.write(line)
        print("finished: ", k)

print("FINISHED")