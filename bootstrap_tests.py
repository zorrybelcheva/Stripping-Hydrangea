import numpy as np
import matplotlib.pyplot as plt
import time


def bootstrap(sample):
    size = len(sample)
    boot = np.zeros(size)
    for i in range(size):
        rand = np.random.randint(0, size-1)
        boot[i] = sample[rand]
    return boot


start = time.time()

sample = np.arange(215)
a = []
u = []

for i in range(10000):
    b = bootstrap(sample)
    # print(b)
    a.append(np.average(b))
    u.append(len(np.unique(b)))
    # print(np.average(b), len(np.unique(b)))
    # print('\n')

end = time.time()
print('Elapsed: ', end-start)

plt.figure(1)
plt.hist(a, bins=50)
plt.axvline(214/2, c='k')
plt.title('average')

plt.figure(2)
plt.hist(u, bins=50)
plt.title('unique')
