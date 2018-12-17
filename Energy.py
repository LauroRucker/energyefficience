import pandas as pd
# import only K means from inside the cluster/sk learn
from sklearn.cluster import KMeans
from sklearn.externals import joblib

# Open
data = pd.read_csv("files/data.csv", delimiter=';', usecols = [0,1,2,3,4,5,6,7,8])
data.dropna()
print(data)
x = data.iloc[1:, 0:9].values
clusterizador = KMeans(n_clusters=9, init='random')
print(clusterizador.fit(x))
print('Show centroids')
print(clusterizador.cluster_centers_)
 # Distances
print('Distances')
distances = clusterizador.fit_transform(x)
print(distances)
# # Set cluster labels for each instance in the class
print('Labels')
labels = clusterizador.labels_
print(labels)
for idx, val in enumerate(labels):
     print(idx, val)
#
new_instance = [[0.71, 710.50, 269.50, 220.50, 3.50, 3, 0.40, 3, 14.03]]
print(clusterizador.predict(new_instance))

#saving the cluster

joblib.dump(clusterizador, 'clusterab.joblib')
#loading the cluster
clusterizador = joblib.load('clusterab.joblib')
new_instance = [[0.71, 710.50, 269.50, 220.50, 3.50, 3, 0.40, 3, 14.03]]
print(clusterizador.predict(new_instance))
#print(x)
