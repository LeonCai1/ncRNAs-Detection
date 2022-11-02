from libsvm.svmutil import *
class rbf_kernel_svm():
    def __init__(self) -> None:
        pass
    
    def train(self, x_train, y_train, c, a):
        prob = svm_problem(y_train, x_train)
        param = svm_parameter(f'-c {c} -g {a}')
        m = svm_train(prob, param)
        return m
    
    def predict(self, x_test, y_test, model):
        return svm_predict(y_test, x_test, model)
        