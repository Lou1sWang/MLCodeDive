import knn
group,labels = knn.createDataSet()
print(group,labels)
print(knn.classify0([0,0], group, labels, 3))
