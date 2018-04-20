#!/usr/bin/env python
import numpy as np
from kmodes.kprototypes import KPrototypes
import pandas as pd

# stocks with their market caps, sectors and countries
syms = np.genfromtxt('stocks.csv', dtype=str, delimiter=',')[:, 0]
X = np.genfromtxt('stocks.csv', dtype=object, delimiter=',')[:, 1:]

X[:, 0] = X[:, 0].astype(float)
kproto = KPrototypes(n_clusters=3, init='Cao', verbose=8)
clusters = kproto.fit_predict(X, categorical=[1, 2]) #TC: define categorical variables here

# Print cluster centroids of the trained model.
print("\nCluster centroid")
print(kproto.cluster_centroids_)
# Print training statistics
print("\nCost")
print(kproto.cost_)
print("\nNumber of iterations")
print(kproto.n_iter_)

"""for s, c in zip(syms, clusters):
    print("Symbol: {}, cluster:{}".format(s, c))"""

print("\nClustering result")
df = pd.DataFrame(zip(syms, clusters))
df.columns = ["Symbol", "Cluster"]
print(df)


