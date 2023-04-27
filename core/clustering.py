import kmedoids
import numpy
from sklearn.datasets import fetch_openml
from sklearn.metrics.pairwise import euclidean_distances
import time

X, _ = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X[:10000]

print(X.shape)

diss = euclidean_distances(X)
start = time.time()
fp = kmedoids.fasterpam(diss, 100)
print("FasterPAM took: %.2f ms" % ((time.time() - start)*1000))
print("Loss with FasterPAM:", fp.loss)
print("labels:", fp.labels, fp.labels.shape)