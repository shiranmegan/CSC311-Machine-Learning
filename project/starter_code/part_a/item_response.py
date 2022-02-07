from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    user_id, question_id, is_correct = data["user_id"], data["question_id"], data["is_correct"]
    for k in range(len(user_id)):
        theta_i = theta[user_id[k]]
        beta_j = beta[question_id[k]]
        log_lklihood += is_correct[k]*(theta_i-beta_j)+beta_j-np.logaddexp(beta_j, theta_i)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id, question_id, is_correct = data["user_id"], data["question_id"], data["is_correct"]
    n_theta = np.zeros(theta.shape)
    n_beta = np.zeros(beta.shape)
    for k in range(len(user_id)):
        n_theta[user_id[k]] += sigmoid(theta[user_id[k]]-beta[question_id[k]]) - is_correct[k]
        n_beta[question_id[k]] += is_correct[k] - sigmoid(theta[user_id[k]]-beta[question_id[k]])
    theta = theta - lr * n_theta
    beta = beta - lr * n_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(542)
    beta = np.random.rand(1774)

    val_acc_lst = []
    train_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        train_acc_lst.append(evaluate(data=data, theta=theta, beta=beta))
        train_lld_lst.append(-neg_lld_train)
        val_lld_lst.append(-neg_lld_val)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_acc_lst, train_lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    irt_0 = irt(train_data, val_data, 0.01, 50)
    # print("For learning rate = 0.01, iteration = 50, the validation accuracy is {}".format(irt_0[2]))
    plt.plot(list(range(50)), irt_0[4], label="train log_likelihhood")
    plt.plot(list(range(50)), irt_0[5], label="validation log_likelihhood")
    plt.xlabel("iteration")
    plt.ylabel("log_likelihood")
    plt.legend()
    plt.show()

    # (c)
    theta_last = irt_0[0]
    beta_last = irt_0[1]
    val_acc_last = evaluate(val_data, theta_last, beta_last)
    test_acc_last = evaluate(test_data, theta_last, beta_last)
    print("For learning rate = 0.01, iteration = 50")
    print("the final validation accuracy is {}".format(val_acc_last))
    print("the final test accuracy is {}".format(test_acc_last))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    theta = np.sort(theta_last)
    for i in range(3):
        question_i = beta_last[i]
        c_ij = sigmoid(theta - question_i)
        plt.plot(theta, c_ij, label="Question "+str(i+1))
    plt.ylabel("Probability of Correct Response vs. Theta")
    plt.xlabel("Theta")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
