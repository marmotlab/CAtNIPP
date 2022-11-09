import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import product


class Gaussian2D():
    def __init__(self):
        self.rangeDistribs = (8,12)
        self.mean = []
        self.sigma_x = []
        self.sigma_y = []
        self.cov = []
        self.max_value = None
        self.createProbability()

    def addGaussian(self):

        gaussian_mean = np.random.rand(2)
        gaussian_var = np.zeros((2, 2))
        gaussian_var[([0, 1], [0, 1])] = np.random.uniform(0.00005,0.0002,2)
        SigmaX = np.sqrt(gaussian_var[0][0])
        SigmaY = np.sqrt(gaussian_var[1][1])
        Covariance = gaussian_var[0][1]
        self.mean.append(gaussian_mean)
        self.sigma_x.append(SigmaX)
        self.sigma_y.append(SigmaY)
        self.cov.append(Covariance)

    def createProbability(self):
        numDistribs = np.random.randint(self.rangeDistribs[0], self.rangeDistribs[1] + 1)
        # print(numDistribs)
        for _ in range(numDistribs):
            self.addGaussian()

    def distribution_function(self, X):
        y = np.zeros(X.shape[0])
        row_mat, col_mat = X[:,0], X[:,1]
        for gaussian_mean, SigmaX1, SigmaX2, Covariance in zip(self.mean, self.sigma_x, self.sigma_y, self.cov):
            # gaussian_mean /= 64
            # SigmaX1 /= 1024
            # SigmaX2 /= 1024
            # Covariance /= 64
            r = Covariance / (SigmaX1 * SigmaX2)
            coefficients = 1 / (2 * math.pi * SigmaX1 * SigmaX2 * np.sqrt(1 - math.pow(r, 2)))
            p1 = -1 / (2 * (1 - math.pow(r, 2)))
            px = np.power(row_mat - gaussian_mean[0], 2) / SigmaX1
            py = np.power(col_mat - gaussian_mean[1], 2) / SigmaX2
            pxy = 2 * r * (row_mat - gaussian_mean[0]) * (col_mat - gaussian_mean[1]) / (SigmaX1 * SigmaX2)
            distribution_matrix = coefficients * np.exp(p1 * (px - pxy + py))
            y += distribution_matrix
        # y /= np.max(y)
        # print(y)
        if self.max_value is None:
            # print(y.shape)
            #assert y.shape == (2500,)
            self.max_value = np.max(y)
            y /= self.max_value
        else:
            y /= self.max_value
        return y



    @staticmethod
    def plot(img):
        plt.imshow(img)
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    example = Gaussian2D()
    print(len(example.mean))
    x1 = np.linspace(0,1)
    x2 = np.linspace(0,1)
    x1x2 = np.array(list(product(x1, x2)))
    y = example.distribution_function(x1x2)
    example.plot(y.reshape(50,50))
