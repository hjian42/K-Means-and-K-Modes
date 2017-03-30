import random 
import numpy as np
from collections import defaultdict, Counter, deque
N = 200
k = 5
data = 'mushroom.training'
def get_data(data):
    with open(data, 'r') as f:
        lines = f.readlines()
        instances = []
        # labels = []
        for line in lines:
            line = line.strip().split('\t')
            # label = line[0]
            # attributes = line[1:]
            # shift the class attribute to the last column
            line = deque(line)
            line.rotate(-1)
            line = list(line)
            instances.append(line)
            # labels.append(label)
    return instances

def is_converged(centroids, old_centroids):
    return set([tuple(a) for a in centroids]) == set([tuple(b) for b in old_centroids])

def get_distance(x, c):
    return np.sum(np.array(x) != np.array(c), axis = 0)

def get_clusters(X, centroids):
    clusters = defaultdict(list)
    for x in X:
        # cluster is a num to indicate the # of centroids
        cluster = np.argsort([get_distance(x[:-1], c[:-1]) for c in centroids])[0]
        clusters[cluster].append(x)
    return clusters

def get_centeroids(old_centroids, clusters):
    new_centroids = []
    keys = sorted(clusters.keys())
    for k in keys:
        points = np.array(clusters[k])
        mode = [Counter(points[:, i]).most_common(1)[0][0] for i in range(len(old_centroids[0])-1)]
        mode.append('PAD')
        new_centroids.append(mode)
    return new_centroids

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

def get_purity(clusters, centroids, num_instances):
    counts = 0
    for k in clusters.keys():
        labels = np.array(clusters[k])[:, -1]
        counts += Counter(labels).most_common(1)[0][1]
    return float(counts)/num_instances

# X = [x1, x2, ..., label]
X = get_data(data)
num_instances = len(X)
# print X[0]
# print labels[0]
centroids, clusters, iteration= find_centers(X, 2)
purity = get_purity(clusters, centroids, num_instances)
# print centroids, iteration
for k in clusters.keys():
        points = np.array(clusters[k])
        class_attr = Counter(points[:, -1]).most_common(1)
        print class_attr
print '\n'
print('The purity for the task is %f' % purity)













