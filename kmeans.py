import numpy as np
import collections

class K_Means:

    def __init__(self, k=2, tol = 0.001, max_iter = 300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classification = collections.defaultdict(list)

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[c]) for c in self.centroids]
                classification = distances.index(min(distances))
                self.classification[classification].append(featureset)

            prev = dict(self.centroids)

            for classification in self.classification:
                self.centroids[classification] = np.average(self.classification[classification], axis = 0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False
            if optimized:
                break

    def predict(self,x):
        distances = [np.linalg.norm(x-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification



# def distEclud(vecA,vecB):
#     return np.sqrt(np.sum(np.power(vecA-vecB,2)))
#
# def randCent(dataset,k):
#     n = dataset.shape[1]
#     centroids = np.mat(np.zeros((k,n)))
#     for i in range(n):
#         minI = min(dataset[:,i])
#         rangeI = np.float(max(dataset[:,i]) - minI)
#         centroids[:,i] = minI + rangeI*np.random.rand(k,1)
#     return centroids
#
# def kMeans(dataset, k, distMeans = distEclud, createCent = randCent):
#     m = dataset.shape[0]
#     clusterAssment = np.mat(np.zeros((m,2))) # index of cluster, distance
#     centroids = createCent(dataset,k)
#     clusterChanged = True
#     while clusterChanged:
#         clusterChanged = False
#         for i in range(m):
#             minDist = float('-inf')
#             minIndex = -1
#             for j in range(k):
#                 distJI = distMeans(centroids[j,:], dataset[i,:])
#                 if distJI < minDist:
#                     minDist = distJI
#                     minIndex = j
#             if clusterAssment[i,0] != minIndex:
#                 clusterChanged = True
#         for cent in range(k):
#             ptsInClust = dataset[np.nonzero(clusterAssment[:,0]==cent)]
#             centroids[cent,:] = np.mean(ptsInClust, axis=0)
#     return centroids, clusterAssment
