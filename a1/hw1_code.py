import graphviz
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def load_data(file1, file2):
    f1 = open(file1, 'r')
    f2 = open(file2, 'r')
    dataset = []
    for line in f1.readlines():
        dataset.append(line.strip())  # real
    label = [0] * len(dataset)
    for line in f2.readlines():
        dataset.append(line.strip())  # fake
    f1.close()
    f2.close()
    label += [1] * (len(dataset)-len(label))
    vectorizer = CountVectorizer()
    news = vectorizer.fit_transform(dataset)
    features = vectorizer.get_feature_names()
    n_train, n_rest, label_train, label_rest = train_test_split(news, label, train_size=0.7, random_state=45)
    n_validate, n_test, label_validate, label_test = train_test_split(n_rest, label_rest, test_size=0.5,
                                                                      random_state=45)
    return n_train, n_validate, n_test, label_train, label_validate, label_test, features


def select_model(n_train, n_validate, n_test, label_train, label_validate, label_test, criterion, max_depth):
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    clf = clf.fit(n_train, label_train)
    res = clf.predict(n_validate)
    correct = 0
    for i in range(len(label_validate)):
        if label_validate[i] == res[i]:
            correct += 1
    return correct/len(label_validate)


def compute_information_gain(decision_tree):
    entropy_parent = decision_tree.tree_.impurity[0]
    count = decision_tree.tree_.n_node_samples[0]
    left = decision_tree.tree_.children_left[0]
    entropy_left = decision_tree.tree_.impurity[left]
    left_count = decision_tree.tree_.n_node_samples[left]
    right = decision_tree.tree_.children_right[0]
    entropy_right = decision_tree.tree_.impurity[right]
    right_count = decision_tree.tree_.n_node_samples[right]
    info_gain = entropy_parent - left_count/count * entropy_left - right_count/count * entropy_right
    return info_gain


if __name__ == '__main__':
    # part a
    x_train, x_validate, x_test, y_train, y_validate, y_test, feature_names = load_data('clean_real.txt',
                                                                                        'clean_fake.txt')
    # part b
    depths = [2, 8, 15, 25, 40, 55]
    criteria = ['gini', 'entropy']
    accuracy = []
    for j in criteria:
        for k in depths:
            accuracy.append(select_model(x_train, x_validate, x_test, y_train, y_validate, y_test, j, k))
    print(accuracy)

    # part c
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=40)
    clf = clf.fit(x_train, y_train)
    plt.figure(figsize=(10, 6))
    tree.plot_tree(clf, max_depth=2, feature_names=feature_names, class_names=['real', 'fake'], fontsize=12,
                   filled=True)
    plt.show()  # generate the graph

    # part d
    print(compute_information_gain(clf))  # information of topmost split

    # information gain with sklearn function (mutual information)
    # res = dict(zip(feature_names, mutual_info_classif(x_train, y_train, discrete_features=True)))
    # print(res['trump'])
    # print(res['donald'])
