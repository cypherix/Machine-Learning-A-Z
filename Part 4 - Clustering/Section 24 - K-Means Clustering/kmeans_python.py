#K Means

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values

#Using the elbow methord to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
   kmeans = KMeans(n_clusters = i, random_state = 0)
   kmeans.fit(X)
   wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Methord')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Applying Kmeasn to the mall dataset
kmeans = KMeans(n_clusters = 5, random_state = 0)
ykmeans = kmeans.fit_predict(X)

#Visualizing the clusters
plt.scatter(X[ykmeans == 0, 0], X[ykmeans == 0, 1], s = 100, c= 'red', label = 'Cluster 1')
plt.scatter(X[ykmeans == 1, 0], X[ykmeans == 1, 1], s = 100, c= 'blue', label = 'Cluster 2')
plt.scatter(X[ykmeans == 2, 0], X[ykmeans == 2, 1], s = 100, c= 'green', label = 'Cluster 3')
plt.scatter(X[ykmeans == 3, 0], X[ykmeans == 3, 1], s = 100, c= 'cyan', label = 'Cluster 4')
plt.scatter(X[ykmeans == 4, 0], X[ykmeans == 4, 1], s = 100, c= 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c= 'yellow', label = 'Centroids')
plt.title('Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()