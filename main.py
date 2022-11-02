import data_preprocessing as dp
from libsvm.svmutil import *
import matplotlib.pyplot as plt
from linear_kernel_svm import linear_kernel_svm
from rbf_kernel_svm import rbf_kernel_svm
import cross_validation as cv
def main():
    # get preprocessed data
    train_data = dp.get_data('ncRNA_s.train.txt')
    x_train, y_train = dp.get_x_y(train_data)
    test_data = dp.get_data('ncRNA_s.test.txt')
    x_test, y_test = dp.get_x_y(test_data)
    # vary C from [2^-4, 2^-3, 2^-2 ..., 2^7, 2^8]
    C = []
    accuracy =[]
    model = linear_kernel_svm()
    for i in range(-4, 9):
        C.append(pow(2, i))
    for c in C:
        m = model.train(x_train, y_train, c)
        res = model.predict(x_test, y_test, m)
        accuracy.append(res[1][0])
    # plot the accuracy graph
    plt.plot(C, accuracy, color='tab:blue', marker='o')
    plt.plot(C, accuracy, 'g', label = 'Accuracy')
    plt.title('Classification accuracy with respect to different C')
    plt.xlabel('C-cost')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.plot()
    plt.show()
    # Part 2
    # 2.1 using cross validation to choose the best alpha and C
    A = C
    a, c = cv.cross_validation(train_data, A, C)
    print(f'the best a and c are {a}, {c}')
    # 2.2 train and find accuracy
    model = rbf_kernel_svm()
    m = model.train(x_train, y_train, c, a)
    res = model.predict(x_test, y_test, m)
    print(f'the RBF kernel SVM is {res[1][0]}')
if __name__ == '__main__':
    main()
    
        
        
        
    
    