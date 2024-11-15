import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('/Users/tommasoferracina/Downloads/ML_Project/mldata.csv')

# Perform PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(data)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Color the points based on their label
labels = data['label']
ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], c=labels)
# Set labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA Plot')

# Show the plot
plt.show()