# TODO: complete this file.
from matrix_factorization import *
from random import randrange
import random
from utils import *
import numpy as np


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    n = len(train_data["is_correct"])
    indices_1 = random.choices(range(n), k=n)
    indices_2 = random.choices(range(n), k=n)
    indices_3 = random.choices(range(n), k=n)
    user_id = train_data['user_id']
    question_id = train_data['question_id']
    is_correct = train_data['is_correct']
    train_1 = {
        'user_id': [user_id[i] for i in indices_1],
        'question_id': [question_id[i] for i in indices_1],
        'is_correct': [is_correct[i] for i in indices_1]
    }
    train_2 = {
        'user_id': [user_id[i] for i in indices_2],
        'question_id': [question_id[i] for i in indices_2],
        'is_correct': [is_correct[i] for i in indices_2]
    }
    train_3 = {
        'user_id': [user_id[i] for i in indices_3],
        'question_id': [question_id[i] for i in indices_3],
        'is_correct': [is_correct[i] for i in indices_3]
    }
    als_1 = sparse_matrix_evaluate(test_data, als(train_1, 50, 0.1, 50000, val_data)[0])
    als_2 = sparse_matrix_evaluate(test_data, als(train_2, 50, 0.1, 50000, val_data)[0])
    als_3 = sparse_matrix_evaluate(test_data, als(train_3, 50, 0.1, 50000, val_data)[0])
    als_avg = (als_1 + als_2 + als_3) / 3
    print("test data accuracy: ", als_avg)


if __name__ == "__main__":
    main()