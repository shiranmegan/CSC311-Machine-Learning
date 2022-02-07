'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        temp = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = np.mean(temp)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        temp = data.get_digits_by_label(train_data, train_labels, i)
        x_mu = temp - means[i]
        covariances[i] = np.dot(x_mu, np.transpose(x_mu)) + 0.01*np.identity(64)
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    n = len(digits)
    log_like = np.zeros((n, 10))
    for i in range(n):
        for j in range(10):
            x_mu = digits[i] - means[j]
            temp = np.dot(np.dot(np.transpose(x_mu), np.linalg.inv(covariances[j])), x_mu)
            log_like[i][j] = -32*np.log(2*np.pi) - 1/2*np.log(np.linalg.det(covariances[j])) - 1/2*temp

    return log_like

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    n = len(digits)
    cond_like = np.zeros((n, 10))
    log_like = generative_likelihood(digits, means, covariances)
    for i in range(n):
        cond_like[i] = log_like[i] - np.logaddexp(log_like[i])
    return cond_like

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    return None

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation

if __name__ == '__main__':
    main()
