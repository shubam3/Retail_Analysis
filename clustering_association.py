import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
# Load the dataset
file_path = "~/Retail_Transaction_Dataset.csv"
df = pd.read_csv(file_path)

# Select features for clustering (numerical columns)
features = ['Price', 'Quantity', 'DiscountApplied(%)']
X = df[features]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Subsample the data for quicker calculations
df_sampled = df.sample(frac=0.2, random_state=5805)
X_sampled = df_sampled[features]
X_scaled_sampled = scaler.fit_transform(X_sampled)

# find the optimal number of clusters using the Elbow Method and Silhouette Score
wcss = []
silhouette_scores = []

for k in range(2, 11):  # Testing for k = 2 to 10
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, max_iter=100)
    kmeans.fit(X_scaled_sampled)
    wcss.append(kmeans.inertia_)
    # Calculate silhouette score on the sampled data
    silhouette_scores.append(silhouette_score(X_scaled_sampled, kmeans.labels_))

# Plot WCSS (Elbow Method)
plt.figure()
plt.plot(range(2, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Plot Silhouette Scores
plt.figure()
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()

# Find the optimal number of clusters (based on silhouette score)
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

# Apply K-means with the optimal number of clusters to the full dataset
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, max_iter=100)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze cluster centroids
centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
print("Cluster Centroids:\n", centroids)

# -----------------------------------
# Scatter Plot of Clusters
# -----------------------------------
x_feature_idx = 0  # 'Price'
y_feature_idx = 1  # 'Quantity'

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, x_feature_idx], X_scaled[:, y_feature_idx],
            c=df['Cluster'], cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, x_feature_idx],
            kmeans.cluster_centers_[:, y_feature_idx],
            s=300, c='red', marker='X')
plt.title("Clusters visualized on Scaled Price vs. Scaled Quantity")
plt.xlabel("Scaled Price")
plt.ylabel("Scaled Quantity")
plt.show()
# -----------------------------------
# Scatter Plot of Clusters
# -----------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(8,6))
# Use the cluster labels as hue
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['Cluster'], palette='Set2', alpha=0.7)

# Plot the cluster centroids
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=300, c='red', marker='X', label='Centroids')

plt.title(f'K-means Clusters (k={optimal_k})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

##DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Select numerical features for DBSCAN
features = ['Price', 'Quantity', 'DiscountApplied(%)']  # Replace with relevant numerical columns
X = df[features]

# Check for variability in data
print("Feature Descriptive Statistics:")
print(X.describe())

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualize the data
sns.pairplot(df[features])
plt.show()

# Use K-distance plot to determine eps
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Sort distances and plot
distances = np.sort(distances[:, 4])
# Perform DBSCAN with tuned parameters
dbscan = DBSCAN(eps=1.0, min_samples=3, n_jobs=-1)
dbscan_labels = dbscan.fit_predict(X_scaled)
df['Cluster'] = dbscan_labels

# Display cluster counts
print("\nCluster Counts:")
print(df['Cluster'].value_counts())

# Dimensionality Reduction with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Perform DBSCAN on reduced dimensions
dbscan_pca = DBSCAN(eps=1.0, min_samples=3, n_jobs=-1)
dbscan_labels_pca = dbscan_pca.fit_predict(X_pca)

# Add PCA-based cluster labels to the dataset
df['PCA_Cluster'] = dbscan_labels_pca

# Switch to KMeans if DBSCAN does not work
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

df['KMeans_Cluster'] = kmeans_labels

print("\nKMeans Cluster Counts:")
print(df['KMeans_Cluster'].value_counts())

# Visualize KMeans results
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', s=5)
plt.xlabel("Price (Scaled)")
plt.ylabel("Quantity (Scaled)")
plt.title("KMeans Clustering Results")
plt.colorbar(label='Cluster')
plt.show()

#Aprior
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Prepare data for Apriori
retail_data = df[['CustomerID', 'ProductCategory']].pivot_table(
    index='CustomerID', columns='ProductCategory', aggfunc=lambda x: 1, fill_value=0
)

# Ensure binary format
retail_data = retail_data.applymap(lambda x: 1 if x > 0 else 0)

# Apply Apriori algorithm with reduced min_support
frequent_itemsets = apriori(retail_data, min_support=0.003, use_colnames=True)

# Check if frequent itemsets are generated
if frequent_itemsets.empty:
    print("No frequent itemsets found. Trying lowering min_support further.")
else:
    print(f"Frequent Itemsets:\n{frequent_itemsets}")

    # Generate association rules with reduced min_threshold
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.003,
                              num_itemsets=len(frequent_itemsets))

    if rules.empty:
        print("No association rules found. Trying lowering the min_threshold or adjust other parameters.")
    else:
        print("\nAssociation Rules:")
        print(rules)


