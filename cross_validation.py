import numpy as np
import data_preprocessing as dp
from rbf_kernel_svm import rbf_kernel_svm
def cross_validation(train_data, A, C):
    """_Using the cross_validation method to find the best cost and gamma in rbf kernel_

    Args:
        train_data (_type_): _description_
        A (_type_): _description_
        C (_type_): _description_

    Returns:
        _type_: _description_
    """
    # randomly choose 50% of the training set as the cross validation set
    temp = train_data
    np.random.shuffle(temp)
    cross_validation_set = temp[:int(len(temp)/2)]
    #devide the cross validation set into 5 equal-sized subset
    n = len(cross_validation_set)
    bag = []
    chunk = n/5
    for i in range(5):
        start = i * chunk
        end = chunk * (i + 1)
        if i == 4: 
            end = n
        train = []
        test = []
        
        for j in range(n):
            if j >= start and j < end:
                train.append(cross_validation_set[j])
            else:
                test.append(cross_validation_set[j])
        bag.append((train, test))
    # train each subset
    best = (0, 0, 0)
    for a in A:
        for c in C:
            curAccuracy = []
            for subSet in bag:
                 x_train, y_train = dp.get_x_y(subSet[0])
                 x_test, y_test = dp.get_x_y(subSet[1])
                 model = rbf_kernel_svm()
                 m = model.train(x_train, y_train, c, a)
                 res = model.predict(x_test, y_test, m)
                 curAccuracy.append(res[1][0])
            # The cross-validation accuracy is the average accuracy over the 5 validation subsets
            accuracy = np.average(curAccuracy)
            if accuracy > best[2]:
                best = (a, c, accuracy)
    return best[0], best[1]
            