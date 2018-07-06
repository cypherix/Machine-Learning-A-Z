#Hierarchical Clustering

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values

#Using the dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dentogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dentogram')
plt.xlabel('Customers')
plt.ylabel('Euclidan distances')
plt.show()

#Fitting HC to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
yhc = hc.fit_predict(X)

#Visualising the clusters
plt.scatter(X[yhc == 0, 0], X[yhc == 0, 1], s = 100, c= 'red', label = 'Cluster 1')
plt.scatter(X[yhc == 1, 0], X[yhc == 1, 1], s = 100, c= 'blue', label = 'Cluster 2')
plt.scatter(X[yhc == 2, 0], X[yhc == 2, 1], s = 100, c= 'green', label = 'Cluster 3')
plt.scatter(X[yhc == 3, 0], X[yhc == 3, 1], s = 100, c= 'cyan', label = 'Cluster 4')
plt.scatter(X[yhc == 4, 0], X[yhc == 4, 1], s = 100, c= 'magenta', label = 'Cluster 5')
plt.title('Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()