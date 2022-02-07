from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    knn = KNNImputer(n_neighbors=k)
    mat = knn.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("For k = {}".format(k))
    print("validation accuracy imputed by item is: ")
    print(acc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    acc_by_user = []
    acc_by_item = []
    k_sets = [1, 6, 11, 16, 21, 26]
    for k in k_sets:
        acc_by_user.append(knn_impute_by_user(sparse_matrix, val_data, k))
    for k in k_sets:
        acc_by_item.append(knn_impute_by_item(sparse_matrix, val_data, k))
    plt.plot(k_sets, acc_by_user)
    plt.xlabel("k value")
    plt.ylabel("accuracy validation")
    plt.title("Validation accuracy for K imputed by user")
    plt.show()
    k_user_max = k_sets[acc_by_user.index(max(acc_by_user))]
    print("For user part we choose k = {}".format(k_user_max))
    print("The final test accuracy is {}".format(knn_impute_by_user(sparse_matrix, test_data, k_user_max)))

    plt.plot(k_sets, acc_by_item)
    plt.xlabel("k value")
    plt.ylabel("accuracy validation")
    plt.title("Validation accuracy for K imputed by item")
    plt.show()
    k_item_max = k_sets[acc_by_item.index(max(acc_by_item))]
    print("For item part we choose k = {}".format(k_item_max))
    print("The final test accuracy is {}".format(knn_impute_by_user(sparse_matrix, test_data, k_item_max)))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
