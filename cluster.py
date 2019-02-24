import numpy as np
import matplotlib.pyplot as plt
import random

N = 5  #number of clusters

def clust():
    """clust creates a 300x2 matrix centered at x0,y0 with stdev 1"""
    x0 = random.random()*10
    y0 = random.random()*10     #x0,y0 is the center of the cluster
    return np.random.normal((x0,y0), 1, (300,2))

def numclust(k):
    """numclust creates k number of clusters"""
    return np.concatenate([clust() for i in range(k)])


data = numclust(N)  #we are making k clusters
rows = random.sample(range(data.shape[0]), N)
centroids = data[rows]  #centroids is random sample of rows


def plotting(data, centroids, clusters):
    """plots all of the data and the k centroids"""
    clrs = ("r", "m", "black", "b", "c", "y", "g")
    plt.scatter(data[:,0], data[:,1], c = [clrs[c] for c in clusters], s=5)   #this line plots all the data
    #plt.scatter(centroids[:,0], centroids[:,1], s=500, c='clrs', marker="*", edgecolors='b')  #this plots 3 centroids
    plt.scatter(centroids[:,0], centroids[:,1], s=200)
    plt.show()


def distances(data, centroids):
    """next step is to find distance from every point to every vector (300x3 matrix)"""
    sum = [np.sum(((data-centroid)**2), axis=1).reshape((-1,1)) for centroid in centroids]
    return np.concatenate(sum, axis=1)

def smdist(distances):
    """finds smallest distance from a point to a centroid, result is
    index column that has smallest distance for each row"""
    return distances.argmin(axis=1)

small = np.array([])
while True:
    prev = small.copy()
    small = smdist(distances(data, centroids))  #for each data point, which centroid is closest to it?
    #now we need to update the centroids so that they
    if np.all(prev == small):
        break
    for i in range(N):
        #gives us new and improved location of each centroid
        centroids[i] = np.average(data[small==i], axis=0)
        #data[small==0] this finds all the points that are closest to centroid 0
    plotting(data, centroids, small)

