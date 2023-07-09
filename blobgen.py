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
    def __init__(self, nblobs: int, width: float, height: float):
        self.nblobs = nblobs
        self.width = width
        self.height = height

        mux = np.random.uniform(-width, width, nblobs)
        muy = np.random.uniform(-height, height, nblobs)
        covi = 0.2 * np.identity(2)
        mag = np.random.normal(size=nblobs)  # customizable
        self.blobs: list[Blob] = [
            Blob(np.array([muxi, muyi]), covi, magi)
            for muxi, muyi, magi in zip(mux, muy, mag)
        ]

    def __call__(self, x: float, y: float):
        tmp = np.array([b(x, y) for b in self.blobs])
        return np.sum(tmp)

    def mapgen(self):
        x = np.linspace(-self.width, self.width, 21)
        y = np.linspace(-self.height, self.height, 21)

        X, Y = np.meshgrid(x, y)

        Z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i][j] = self(x[i], y[j])

        Z = Z - np.mean(Z)

        return X, Y, Z


def plot_blob_origins(blob_map):
    plt.scatter(
        [blob.mu[0] for blob in blob_map.blobs],
        [blob.mu[1] for blob in blob_map.blobs],
        [30 * np.exp(blob.mag) for blob in blob_map.blobs],
    )


def plot_surf(X, Y, Z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z)


def print_map_to_file(f, X, Y, Z):
    print(*X.shape, file=f)
    for x, y, z in zip(X.flat, Y.flat, Z.flat):
        print(x, y, z, file=f)


if __name__ == "__main__":
    width = 4
    height = 4
    nblobs = 20

    blob_map = BlobMap(nblobs, width, height)

    X, Y, Z = blob_map.mapgen()

    with open("map.dat", "w") as f:
        print_map_to_file(f, X, Y, Z)

    plot_blob_origins(blob_map)
    plot_surf(X, Y, Z)
    plt.show()
