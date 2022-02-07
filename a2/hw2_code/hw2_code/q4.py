# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)
from sklearn.model_selection import train_test_split

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    diagol = []
    exps = []
    print(np.transpose(test_datum).shape)
    distances = l2(np.transpose(test_datum), x_train)
    print(distances[0])

    n = len(y_train) # number of training examples
    for i in range(n):
        exps.append((np.exp(- l2(np.transpose(test_datum), np.matrix(x_train[i])))) / (2 * tau ** 2))
    for i in range(n):
        diagol.append((exps[i]) / sum(exps))
    A = np.diag(diagol)
    w_prime = np.linalg.solve(np.linalg.multi_dot([np.transpose(x_train), A, x_train])+lam * np.identity(1),
                              np.linalg.multi_dot([np.transpose(x_train), A, y_train]))
    y_hat = np.dot(np.transpose(test_datum), w_prime)
    return y_hat
    ## TODO




def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    ## TODO
    x_train, x_valid, y_train, y_label = train_test_split(x, y, train_size=1-val_frac, random_state=45)
    prediction = []
    losses = []
    for tau in taus:
        for x in x_valid:
            prediction.append(LRLS(np.transpose(np.matrix(x)), x_train, y_train, tau))
        loss = 0
        for i in range(len(y_train)):
            loss += np.square(y_train[i]-prediction[i])
        losses.append(loss / len(y_train))
    plt.title("average loss of validation set")
    plt.xlabel("tau")
    plt.ylabel("loss")
    plt.plot(tau, losses)
    plt.show()
    # return None
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish

    taus = np.logspace(1.0,3,200)
    run_validation(x,y,taus,val_frac=0.3)
    # train_losses, test_losses =
    #plt.semilogx(train_losses)
    #plt.semilogx(test_losses)

