import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset (Replace this with the actual Iris dataset)
# The dataset should include 4 numeric variables and 1 categorical variable (SpeciesType)
# Example: data = pd.read_csv('iris.csv')

# Sample Iris dataset (replace this with your actual data)
data = pd.read_csv("/Users/riwadesai/Documents/mathfordatascience/Iris_Data.txt")

# Select the numeric features for PCA
numeric_features = data.iloc[:, :-1].values

# Step 1: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_features)

# Step 2: Compute the covariance matrix or correlation matrix
cov_matrix = np.cov(scaled_data, rowvar=False)
# If you want to use the correlation matrix instead, uncomment the line below:
# corr_matrix = np.corrcoef(scaled_data, rowvar=False)

# Step 3: Calculate the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort the eigenvectors in descending order of eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Step 5: Choose the top 2 eigenvectors (for the first two principal components)
k = 2
top_eigenvectors = sorted_eigenvectors[:, :k]

# Step 6: Project the original data onto the top 2 eigenvectors to obtain the principal components
principal_components = np.dot(scaled_data, top_eigenvectors)

# Report the first two principal component loading vectors
print("First two principal component loading vectors:")
print(top_eigenvectors)

# Report the first two principal components and their proportion of variance explained
print("First two principal components:")
print(principal_components)

# Calculate the proportion of variance explained by the first two principal components
explained_variance_ratio = sorted_eigenvalues[:k] / np.sum(sorted_eigenvalues)
print("Proportion of variance explained by the first two principal components:")
print(explained_variance_ratio)

# Plot all the points using the first two principal components
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Scatter Plot of First Two Principal Components')
plt.show()

# Create separate plots for each species type using the first two principal components
species_types = data.iloc[:, -1].values

for species_type in np.unique(species_types):
    indices = species_types == species_type
    plt.scatter(principal_components[indices, 0], principal_components[indices, 1], label=species_type)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Scatter Plot of First Two Principal Components (Separate Plots for Species Types)')
plt.legend()
plt.show()
