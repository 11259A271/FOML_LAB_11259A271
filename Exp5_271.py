import pandas as pd
from sklearn.cluster import KMeans
# Load the customer data
data = pd.read_csv("Mall_Customers.csv")
# Select the features for clustering
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]
# Create the K-Means model
model = KMeans(n_clusters=5,n_init=10,random_state=1)
# Train the model
model.fit(X)
# Add the group/cluster number to the dataset
data["Group"] = model.labels_
# Display the number of customers in each group
print("Number of customers in each group:")
print(data["Group"].value_counts().sort_index())
# Create a new customer
new_customer = [[75, 85]]
# Predict the group of the new customer
group = model.predict(new_customer)[0]
print("\nThis customer belongs to group:", group)
import matplotlib.pyplot as plt          # the charting library

x = data["Annual Income (k$)"]                 # income column
y = data["Spending Score (1-100)"]             # spending column
plt.scatter(x, y, c=model.labels_, cmap="tab10", s=30)  # colour each dot by its group
cen = model.cluster_centers_                   # the 5 group centres
plt.scatter(cen[:,0], cen[:,1], marker="X", s=200, c="black")  # mark the centres
plt.xlabel("income (k$)"); plt.ylabel("spending score")
plt.title("5 customer groups"); plt.show()
