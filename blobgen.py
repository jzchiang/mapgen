from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt

class Blob(NamedTuple):
    mu: np.array
    cov: np.array
    mag: float

    def __call__(self, x: float, y: float):
        dr = self.mu - np.array([x, y])
        return self.mag * np.exp(-0.5 * dr @ self.cov @ dr)

class BlobMap:
    def __init__(self, N: int, width: float, height: float):

        mux = np.random.uniform(-width, width, N)
        muy = np.random.uniform(-height, height, N)
        covi = 0.2*np.identity(2)
        mag = np.random.normal(size=N) # customizable
        self.blobs: list[Blob] = [Blob(np.array([muxi, muyi]), covi, magi) for muxi, muyi, magi in zip(mux, muy, mag)]
    
    def __call__(self, x: float, y: float):
        tmp = np.array([b(x, y) for b in self.blobs])
        return np.sum(tmp)

if __name__ == '__main__':

    width = 4
    height = 4

    blob_map = BlobMap(20, width, height)

    x = np.linspace(-width, width, 21)
    y = np.linspace(-height, height, 21)

    X, Y = np.meshgrid(x, y)

    Z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i][j] = blob_map(x[i], y[j])

    Z = Z - np.mean(Z)

    plt.scatter([blob.mu[0] for blob in blob_map.blobs], [blob.mu[1] for blob in blob_map.blobs], [30*np.exp(blob.mag) for blob in blob_map.blobs])
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(X, Y, Z)
    plt.show()