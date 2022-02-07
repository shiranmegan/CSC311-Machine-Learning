from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.4,
        "weight_regularization": 0.,
        "num_iterations": 1000
    }
    weights = np.random.rand(M+1, 1)
    # weights = np.zeros((M + 1, 1)) # fixed weights of all zeros
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    ce_train = []
    ce_valid = []
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        weights = weights - hyperparameters["learning_rate"] * df

        # record cross entropy
        ce, frac_correct = evaluate(train_targets, y)
        ce_train.append(ce)

        valid_prediction = logistic_predict(weights, valid_inputs)
        ce, frac_correct_valid = evaluate(valid_targets, valid_prediction)
        ce_valid.append(ce)
    # final entropy
    print("this is final entropy of training: ", ce_train[-1])
    print("this is final entropy of valid: ", ce_valid[-1])
    # final classification rate
    print("this is final classification rate of train: ", frac_correct)
    print("this is final classification rate of valid: ", frac_correct_valid)

    test_inputs, test_targets = load_test()
    test_prediction = logistic_predict(weights, test_inputs)
    ce, frac_correct = evaluate(test_targets, test_prediction)
    print("this is final entropy of test: ", ce)
    print("this is final classification rate of test: ", frac_correct)
    plt.plot(list(range(hyperparameters["num_iterations"])), ce_train)
    plt.plot(list(range(hyperparameters["num_iterations"])), ce_valid)
    # plt.title("mnist train dataset")
    plt.title("mnist train dataset small")
    plt.xlabel("number of iterations")
    plt.ylabel("cross entropy value")
    plt.text(300, 0.05, "training")
    plt.text(300, 0.75, "validation")

    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
