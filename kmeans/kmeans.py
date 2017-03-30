
# coding: utf-8


import random 
import numpy as np
import pandas as pd
import sys
from collections import Counter
from sklearn import preprocessing
from collections import defaultdict

# return X: list of list
def get_data(data):
    X = pd.read_csv(data, sep=",", header=None)
    len_col = len(X.columns)
    # maps class labels from chars to 1, 2, 3, ...
    mapping = {'a':1}
    l ='a'
    for i in range(25):
        l = chr(ord(l)+1) 
        mapping.update({l: i+2})
    X_class_col = X[len_col-1].map(mapping)
    maxmin_scalar = preprocessing.MinMaxScaler().fit_transform(X.ix[:,:len_col-2])
    X = pd.DataFrame(maxmin_scalar)
    X = pd.concat([X, X_class_col], axis=1).astype(float)
    print('The data FORMAT is shown as below\n')
    print(X.head())
    X = X.values.tolist()
    return X


def is_converged(centroids, old_centroids):
    return set([tuple(a) for a in centroids]) == set([tuple(b) for b in old_centroids])

# return int: euclidean distance
def get_distance(x, c):
    return np.linalg.norm(np.array(x)-np.array(c))

# return: dictionary of lists 
def get_clusters(X, centroids):
    clusters = defaultdict(list)
    for x in X:
        # cluster is a num to indicate the # of centroids
        cluster = np.argsort([get_distance(x[:-1], c[:-1]) for c in centroids])[0]
        clusters[cluster].append(x)
    return clusters

# return: list of np.lists 
def get_centeroids(old_centroids, clusters):
    new_centroids = []
    keys = sorted(clusters.keys())
    for k in keys:
        if clusters[k]:
            new_centroid = np.mean(clusters[k], axis=0)
            # the class label is determined by majortity vote inside the cluster
            new_centroid[len(clusters[0][0])-1] = Counter([clusters[k][i][-1] for i in range(len(clusters[k]))]).most_common(1)[0][0]
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(old_centroids[k])
    return new_centroids

# return: tuple (centroids, clusters, iteration)
def find_centers(X, K):
    old_centroids = random.sample(X, K)
    centroids = random.sample(X, K)
    iteration = 0
    while not is_converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = get_clusters(X, centroids)
        centroids = get_centeroids(old_centroids, clusters)
        iteration += 1
    return (centroids, clusters, iteration)

# return purity score 
def get_purity(clusters, centroids, num_instances):
    counts = 0
    for k in clusters.keys():
        labels = np.array(clusters[k])[:, -1]
        counts += Counter(labels).most_common(1)[0][1]
    return float(counts)/num_instances



# data = 'wine_data.csv'
# data = 'glass.csv'
data = sys.argv[1]
k = int(sys.argv[2])
output = sys.argv[3]
X = get_data(data)
num_instances = len(X)
centroids, clusters , iteration= find_centers(X, k)
# store the best records in 5 iterations 
best_score = 0
best_centroids = []
best_clusters =[]
best_iteratoin = 0
for i in range(5):
    centroids, clusters , iteration= find_centers(X, k)
    purity = get_purity(clusters, centroids, num_instances)
    if purity > best_score:
        best_centroids = centroids
        best_clusters = clusters
        best_score = purity
        best_iteratoin = iteration

# mapping the class label back to 'a','b','c'...
mapping = {1:'a'}
label = 'a'
for i in range(25):
    label = chr(ord(label)+1)
    mapping.update({i+2: label})
centroids = []
for c in best_centroids:
    c = c.tolist()
    c[-1] = mapping[c[-1]]
    centroids.append(c)
best_centroids = centroids
print('The best purity score is %f' % best_score)
print('It takes %d number of iterations' % best_iteratoin)
with open(output, 'w') as out:
    for k in best_clusters.keys():
        out.write('The %d centroid is \n%s\n\n' % (k, best_centroids[k]))
        out.write('It has following points: \n')
        for pt in clusters[k]:
            out.write('%s\n' % pt)
        out.write('\n\n\n\n')






