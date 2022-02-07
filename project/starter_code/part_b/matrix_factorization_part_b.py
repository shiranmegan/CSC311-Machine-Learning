from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import datetime
import numpy as np


def add_weight(data, q_num):
    q_num += 1
    question_id = data["question_id"]
    is_correct = data["is_correct"]
    correct = [0] * q_num
    total = [0] * q_num
    for i in range(len(question_id)):
        total[question_id[i]] += 1
        if is_correct[i] == 1:
            correct[question_id[i]] += 1
    result = [0.0] * q_num
    for i in range(q_num):
        if total[i] != 0:
            result[i] = correct[i] / total[i]
    weights = [1 - 0.4 * np.abs(0.6 - x) for x in result]
    return weights


def squared_error_loss_baseline(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def squared_error_loss(data, u, z, lamb):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :param lamb: float
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += ((data["is_correct"][i]
                  - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
                 + np.prod(lamb * np.linalg.norm(u[data["user_id"][i]], axis=0), dtype=np.int64) ** 2
                 + np.prod(lamb * np.linalg.norm(z[q], 2), dtype=np.int64) ** 2)
    return 0.5 * loss


def update_u_z_baseline(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    u[n] = u[n] + lr * (c - np.dot(np.transpose(u[n]), z[q])) * z[q]
    z[q] = z[q] + lr * (c - np.dot(np.transpose(u[n]), z[q])) * u[n]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def update_u_z(train_data, lr, u, z, lamb, weights):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :param lamb: float
    :param weights: vector
    :return: (u, z)
    """
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    c = c * weights[q]
    u_r = np.linalg.norm(u[n], 2)
    z_r = np.linalg.norm(z[q], 2)

    # update u and z
    u[n] = u[n] + lr * (c - np.dot(np.transpose(u[n]), z[q])) * z[q] \
                        + np.prod((lamb, u_r), dtype=np.int64)
    z[q] = z[q] + lr * (c - np.dot(np.transpose(u[n]), z[q])) * u[n] \
                        + np.prod((lamb, z_r), dtype=np.int64)
    return u, z


def als_baseline(train_data, k, lr, num_iteration, val_data):
    """ Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param val_data: vector
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    loss = [0] * 50
    loss_val = [0] * 50
    for i in range(num_iteration):
        u, z = update_u_z_baseline(train_data, lr, u, z)
        if i % 1000 == 0:
            loss[i // 1000] = squared_error_loss_baseline(train_data, u, z)
            loss_val[i // 1000] = squared_error_loss_baseline(val_data, u, z)
    mat = np.dot(u, np.transpose(z))
    return mat, loss, loss_val


def als(train_data, k, lr, num_iteration, val_data, lamb, weights):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param val_data: validation data (for calculating loss)
    :param lamb: float
    :param weights: vector
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    loss = [0] * 50
    loss_val = [0] * 50
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z, lamb, weights)

        # calculation purpose
        if i % 1000 == 0:
            loss[i // 1000] = squared_error_loss(train_data, u, z, lamb)
            loss_val[i // 1000] = squared_error_loss(val_data, u, z, lamb)
    mat = np.dot(u, np.transpose(z))
    return mat, loss, loss_val


def main():
    start = datetime.datetime.now()
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    weights = add_weight(train_data, 1773)  # calculate weight
    k = [5, 10, 25, 50, 100]
    # k_star = 50
    # change hyperparameters here
    # accuracy = [0] * 5
    # accuracy_baseline = [0] * 5
    # for i in range(5):
    k_star = 50
    lr = 0.1
    lamb = 0.02
    num_iterations = 500
    # for i in k:
    #     print(i, sparse_matrix_evaluate(val_data, als(train_data, i, lr, num_iterations)[0]))

    x_axis = [0] * 50
    for j in range(50):
        x_axis[j] = j * 1000

    mat, loss_train, loss_val = als(train_data, k_star, lr, num_iterations, val_data, lamb, weights)
    # mat_baseline, loss_train_baseline, loss_val_baseline = als_baseline(train_data, k_star, lr, num_iterations,
    #                                                                     val_data)
    # draw graphs
    # plt.plot(x_axis, loss_train, label="training on modified ALS")
    # plt.plot(x_axis, loss_val, label="validation on modified ALS")
    # plt.plot(x_axis, loss_train_baseline, label="training on base ALS")
    # plt.plot(x_axis, loss_val_baseline, label="validation on base ALS")
    # plt.xlabel("number of iterations")
    # plt.ylabel("Squared Error Loss")
    # plt.legend()
    # plt.show()

    print("validation set accuracy: ", sparse_matrix_evaluate(val_data, mat))
    print("test set accuracy: ", sparse_matrix_evaluate(test_data, mat))
    # accuracy[i] = sparse_matrix_evaluate(test_data, mat)
    # accuracy_baseline[i] = sparse_matrix_evaluate(test_data, mat_baseline)
    # print(accuracy)
    # print(accuracy_baseline)
    # plt.plot(k, accuracy, label="modified ALS")
    # plt.plot(k, accuracy_baseline, label="baseline ALS")
    # plt.xlabel("k value")
    # plt.ylabel("accuracy")
    # plt.legend()
    # plt.show()

    # end = datetime.datetime.now()
    # print("The new version(part b) runtime:")
    # print(end - start)


if __name__ == "__main__":
    main()
