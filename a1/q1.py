import numpy as np
import matplotlib.pyplot as plt

l1 = [0] * 4950
means = [0] * 11
stds = [0] * 11

for k in range(0, 11):
    a = np.random.rand(100, 2 ** k)
    n = 0
    for i in range(0, 100):
        for j in range(i + 1, 100):
            l1[n] = np.sum(a[i] - a[j]) ** 2
            n += 1
    means[k] = np.mean(l1)
    stds[k] = np.std(l1)

if __name__ == '__main__':
    print(means)
    print(stds)
    # plt.plot([2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10], means,'ro')
    # plt.xlabel('dimension')
    # plt.ylabel('mean')
    # plt.show()
    plt.plot([2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10], stds)
    plt.xlabel('dimension')
    plt.ylabel('standard deviation')
    plt.show()
