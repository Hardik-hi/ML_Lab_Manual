 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

 
df = pd.read_csv('kmeans_data.csv')

 
df

# ## Plot the Data

 
plt.scatter(df['Longitude'],df['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

 
x = df.iloc[:,1:3]
x

 
from sklearn.cluster import KMeans

kmeans = KMeans(3)
kmeans.fit(x)

 
identified_clusters = kmeans.fit_predict(x)
identified_clusters

 
data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters 
print(data_with_clusters)
plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Clusters'],cmap='rainbow')


