import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt


def random_range(length, num):
    x = np.arange(-length/2, length/2)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, scale=length/5) - ss.norm.cdf(xL, scale=length/5)
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
    nums = np.random.choice(x, size=num, p=prob)
    return nums

nums = random_range(1600, 10000)
print(nums.min())
print(nums.max())

print(random_range(200, 100))
plt.hist(nums, bins=200)
plt.show()
