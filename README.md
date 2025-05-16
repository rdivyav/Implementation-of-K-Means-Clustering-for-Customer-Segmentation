# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and clean the customer data for analysis.

2. Select relevant features such as income or spending score.

3. Determine the optimal number of clusters (K) using the Elbow Method.

4. Apply the K-Means algorithm to group customers into K clusters.

5. Assign each customer to the nearest cluster based on centroids.

6. Visualize the clusters and interpret the customer segments.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Divya R V
RegisterNumber: 212223100005 
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\Folders\ML\DATASET-20250226\Mall_Customers.csv")
data.head()
data.info()
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/fe206b73-8820-49b6-a024-4ccb799c9ea5)

```
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow method")
```
![image](https://github.com/user-attachments/assets/210e10d4-12d4-43e0-947b-4e89838f2589)

```
km = KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
```
![image](https://github.com/user-attachments/assets/fbb0b476-0d35-4d2f-ad17-aeab6117fb62)
```
y_pred = km.predict(data.iloc[:,3:])
y_pred
```
![image](https://github.com/user-attachments/assets/ae392f89-ad20-4299-b8d8-02527c771be2)

```
data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster 0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster 1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster 2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster 3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster 4")

plt.legend()
plt.title("customer segmentation")
```
![image](https://github.com/user-attachments/assets/db80391e-43e4-42e5-b6a8-2de2b8602967)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
