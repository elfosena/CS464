import numpy as np
import gzip
import matplotlib.pyplot as plt

# Function to read pixel data from the dataset
def read_pixels(data_path):
    with gzip.open(data_path) as f:
        pixel_data = np.frombuffer(f.read(), 'B', offset=16).astype('float32')
    normalized_pixels = pixel_data / 255
    flattened_pixels = normalized_pixels.reshape(-1, 28 * 28)
    return flattened_pixels

# Function to read label data from the dataset
def read_labels(data_path):
    with gzip.open(data_path) as f:
        label_data = np.frombuffer(f.read(), 'B', offset=8)
    return label_data

images = read_pixels("data/train-images-idx3-ubyte.gz")
labels = read_labels("data/train-labels-idx1-ubyte.gz")

# Centering images
mean = np.mean(images, axis=0)
centered_images = images - mean

# Calculating covarience matrix
covariance_matrix = np.cov(centered_images, rowvar=False)

# Calculating eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Taking first 10 largest principal components
first_10_pc_eigenvalues = eigenvalues[0:10]
first_10_pc = eigenvectors[:, 0:10]

# Question 1.1 Calculation of their PVE
pve_first_10 = first_10_pc_eigenvalues / sum(eigenvalues)
print("Proportion of Variance Explained (PVE) for the first 10 principal components:")
print(pve_first_10[:10])

# Bar graph of their PVE
fig, ax = plt.subplots()
ax.bar(range(1, 11), pve_first_10)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Proportion of Variance Explained (PVE)')
ax.set_title('PVE for the First 10 Principal Components')
plt.show()

# Question 1.2
pve = eigenvalues/sum(eigenvalues)
num_components_70 = np.argmax(np.cumsum(pve) >= 0.7) + 1

print(f"Number of principal components to explain 70% of the data: {num_components_70}")

# Question 1.3

scaled_principal_components = (eigenvectors[:, :10].T).reshape(-1, 28, 28)
scaled_principal_components = (scaled_principal_components - np.min(scaled_principal_components)) / (
        np.max(scaled_principal_components) - np.min(scaled_principal_components))

# Displaying images
fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
for i, ax in enumerate(axes):
    ax.imshow(scaled_principal_components[i], cmap='Greys_r')
    ax.axis('off')
plt.show()

# Question 1.4
projected_data = np.dot(centered_images[:100], eigenvectors[:, :2])

plt.scatter(projected_data[:, 0], projected_data[:, 1], c=labels[:100], cmap='nipy_spectral_r', alpha=0.8)
plt.colorbar()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Projection of the first 100 images onto the first 2 principal components')
plt.show()

def reconstruct_image(data, eigenvectors, mean, k):
    data_reshaped = data.reshape(1, -1)
    reconstruction_coefficients = np.dot(data_reshaped - mean, eigenvectors[:, :k])
    reconstructed_data = np.dot(reconstruction_coefficients, eigenvectors[:, :k].T)
    reconstructed_data += mean
    
    return reconstructed_data

plt.imshow(centered_images[0].reshape(28, 28), cmap='Greys_r') 
plt.title("Original Image")
plt.axis('off')
plt.show()
    
# Question 1.5
k_values = [1, 50, 100, 250, 500, 784]
for k in k_values:
    reconstructed_image = reconstruct_image(centered_images[0], eigenvectors, mean, k)
    plt.imshow(reconstructed_image.reshape(28, 28), cmap='Greys_r') 
    plt.title(f'Reconstructed Image with {k} Principal Components')
    plt.axis('off')
    plt.show()