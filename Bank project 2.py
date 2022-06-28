
# coding: utf-8

# In[32]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')

# suppress warnings from final output
import warnings
warnings.simplefilter("ignore")


# In[34]:


#load data
df = pd.read_csv('banknote-authentication.csv')
df_labels = pd.read_csv('banknote-authentication.csv')


# In[18]:


# set up to view all the info of the columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[19]:


df.head()


# In[20]:


df.shape


# In[21]:


df.describe()


# In[22]:


df.info()


# In[24]:


df[df.duplicated()].shape[0]


# In[60]:


plt.figure(figsize = [7, 7])
plt.xlabel('V1. Variance')
plt.ylabel('V2. Skewness')
plt.scatter(df.V1, df.V2);


# In[61]:


data = np.column_stack(( df.V1, df.V2))  # we use only V1 and V2

# compute K-Means
km_res = KMeans(n_clusters = 2).fit(data)
clusters = km_res.cluster_centers_

# put the assigned labels to the original dataset
df['KMeans'] = km_res.labels_

#plot out the result
g = sb.FacetGrid(data = df, hue = 'KMeans', size = 5)
g.map(plt.scatter, 'V1', 'V2')

g.add_legend();
plt.xlabel('V1. Variance')
plt.ylabel('V2. Skewness')

plt.scatter(clusters[:,0], clusters[:,1], s=300, marker='^', c='r', alpha=0.8)


# In[66]:


from sklearn.datasets.samples_generator import (make_blobs,
                                                make_circles,
                                                make_moons)
from sklearn.cluster import KMeans, SpectralClustering

n_iter = 9
fig, ax = plt.subplots(3, 3, figsize=(16, 16))
ax = np.ravel(ax)
centers = []
for i in range(n_iter):
    # Run local implementation of kmeans
    km = KMeans(n_clusters=2,
                max_iter=3)
    km.fit(data)
    centroids = km.cluster_centers_
    centers.append(centroids)
    ax[i].scatter(data[km.labels_ == 0, 0], data[km.labels_ == 0, 1],
                   label='Cluster 1')
    ax[i].scatter(data[km.labels_ == 1, 0], data[km.labels_ == 1, 1],
                   label='Cluster 2')
    ax[i].scatter(centroids[:, 0], centroids[:, 1],
                  c='r', marker='^', s=300, label='Cluster Center', alpha=0.8)
    ax[i].legend(loc='lower right')
    ax[i].set_aspect('equal')
    
plt.tight_layout();


# In[29]:


km_res.cluster_centers_


# In[39]:


df['KMeans'] = km_res.labels_
df.groupby('KMeans').describe()


# In[63]:


# plot the data with Correct labels
g = sb.FacetGrid(data = df_labels, hue = 'Class', size = 5)
g.map(plt.scatter, 'V1', 'V2')
g.add_legend()
plt.xlabel('V1. Variance')
plt.ylabel('V2. Skewness')
plt.title("Data With Correct Lables")


# plot the data computed by K-Means
g = sb.FacetGrid(data = df, hue = 'KMeans', size = 5)
g.map(plt.scatter, 'V1', 'V2')
g.add_legend()
plt.xlabel('V1. Variance')
plt.ylabel('V2. Skewness')
plt.title("K-Means Result");


# In[43]:


# correct the labels
df["KMeans"] = df["KMeans"].map({0: 1, 1: 2})


# In[44]:


correct = 0

for i in range(0,1372):
    if df.Class[i] == df["KMeans"][i]:
        correct+=1
print(correct/1371)

