import matplotlib.pyplot as plt
import numpy as np

data1 = np.array([(x + 2, y + 1) for x, y in np.random.normal(0, 1, (200, 2))
                  if x ** 2 + y ** 2 <= 1])
data2 = np.array([(x, y) for x, y in np.random.normal(0, 2, (500, 2))
                  if (x ** 2) + 4 * (y ** 2) <= 4])

x1, y1 = data1[:, 0], data1[:, 1]
x2, y2 = data2[:, 0], data2[:, 1]

mu1 = np.mean(data1, axis=0)
mu2 = np.mean(data2, axis=0)

siga1 = np.mat(data1 - mu1).T * np.mat(data1 - mu1)
siga2 = np.mat(data2 - mu2).T * np.mat(data2 - mu2)

s_w = siga1 + siga2
w = (mu1 - mu2).T * np.linalg.inv(s_w)
w = np.array(w)
k = w[0][1] / w[0][0]

xs = np.linspace(-3, 3, 10)
ys = xs * k

np.set_printoptions(precision=1)
modu = np.sqrt(np.sum(np.square(w)))
trans1 = data1.dot(w[0]) / modu
trans2 = data2.dot(w[0]) / modu

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(xs, ys, c='red')
plt.scatter(x1, y1, c='orange')
plt.scatter(x2, y2, c='blue')
plt.xlim(-3, 4)
plt.ylim(-2, 3)
plt.grid()

plt.subplot(122)
plt.hist(trans1, alpha=0.8, color='orange')
plt.hist(trans2, alpha=0.8, color='blue')

plt.show()
