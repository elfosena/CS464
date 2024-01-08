import matplotlib.pyplot as plt
import numpy as np
import gzip

# One-hot encoding of the labels
def one_hot_encoding(label_data):
    encoded_labels = np.zeros((label_data.size, label_data.max() + 1))
    encoded_labels[np.arange(label_data.size), label_data] = 1
    return encoded_labels

# Function to read pixel data from the dataset
def read_pixels(data_path):
    with gzip.open(data_path) as f:
        pixel_data = np.frombuffer(f.read(), 'B', offset=16).astype('float32')
    normalized_pixels = pixel_data / 255
    flattened_pixels = normalized_pixels.reshape(-1, 28*28)  # Flatten the normalized pixels
    return flattened_pixels

# Function to read label data from the dataset
def read_labels(data_path):
    with gzip.open(data_path) as f:
        label_data = np.frombuffer(f.read(), 'B', offset=8)
    one_hot_encoding_labels = one_hot_encoding(label_data)
    return one_hot_encoding_labels

def read_dataset():
    X_train = read_pixels("data/train-images-idx3-ubyte.gz")
    y_train = read_labels("data/train-labels-idx1-ubyte.gz")
    X_test = read_pixels("data/t10k-images-idx3-ubyte.gz")
    y_test = read_labels("data/t10k-labels-idx1-ubyte.gz")
    return X_train, y_train, X_test, y_test

# Softmax activation function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Cross-entropy loss with L2 regularization
def cross_entropy_loss(y_true, y_pred, weights, l2_reg_coefficient):
    num_samples = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / num_samples
    l2_regularization = (l2_reg_coefficient / 2) * np.sum(weights**2)
    return loss + l2_regularization

# Gradient descent for weight updates with L2 regularization
def gradient_descent(X, y_true, y_pred, weights, l2_reg_coefficient, learning_rate):
    error = y_pred - y_true
    gradient = np.dot(X.T, error) + l2_reg_coefficient * weights
    weights -= learning_rate * gradient
    return weights

# Logistic Regression training function
def train_logistic_regression(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, l2_reg_coefficient, weight_initialization):
    num_features = X_train.shape[1]
    num_classes = y_train.shape[1]
    num_samples = X_train.shape[0]

    if weight_initialization == 'normal':
        weights = np.random.normal(0, 1, size=(num_features, num_classes))
    elif weight_initialization == 'uniform':
        weights = np.random.uniform(-1, 1, size=(num_features, num_classes))
    elif weight_initialization == 'zero':
        weights = np.zeros((num_features, num_classes))

    best_val_accuracy = 0.0
    best_weights = None
    val_accuracies = []

    for epoch in range(epochs):
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            end = min(end, num_samples)

            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # Forward pass
            logits = np.dot(X_batch, weights)
            predictions = softmax(logits)

            # Backward pass
            weights = gradient_descent(X_batch, y_batch, predictions, weights, l2_reg_coefficient, learning_rate)

        # Validate after each epoch
        val_logits = np.dot(X_val, weights)
        val_predictions = softmax(val_logits)
        val_pred_labels = np.argmax(val_predictions, axis=1)
        val_accuracy = np.mean(val_pred_labels == np.argmax(y_val, axis=1))
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} ")
        
        # Update best weights if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = weights.copy()

    return best_weights, val_accuracies

# Question 2.1
# Read the dataset
X_train, y_train, X_test, y_test = read_dataset()

# Create validation set
X_val = X_train[:10000]
y_val = y_train[:10000]
X_train = X_train[10000:]
y_train = y_train[10000:]

# Set hyperparameters letf as the values selected to give the best results
batch_size = 64
learning_rate = 1e-3
l2_reg_coefficient = 1e-2
epochs = 100

# Train the logistic regression model
final_weights, val_accuracies = train_logistic_regression(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, l2_reg_coefficient, 'zero')

# Evaluate on test set
test_logits = np.dot(X_test, final_weights)
test_predictions = softmax(test_logits)
test_pred_labels = np.argmax(test_predictions, axis=1)

# Display test accuracy and confusion matrix
test_accuracy = np.mean(test_pred_labels == np.argmax(y_test, axis=1))
conf_matrix = np.dot(y_test.T, test_predictions)
conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True)
conf_matrix = conf_matrix * 1000
conf_matrix = conf_matrix.astype(int)

print("Test Accuracy:", str(test_accuracy * 100))
print("Confusion Matrix:\n", conf_matrix)
"""
# For Question 2.2

weight_initializations = ['zero', 'uniform', 'normal']
plt.figure(figsize=(15, 12))

for weight_initialization in weight_initializations:
    weights, val_accuracies = train_logistic_regression(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, l2_reg_coefficient, weight_initialization)
    plt.plot(range(1, epochs + 1), val_accuracies, label=f'Weight Init: {weight_initialization}')

plt.title('Validation Accuracy for Different Weight Initialization Techniques')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show() """

""" l2_reg_coefficients = [0.01, 0.0001, 0.000000001]
plt.figure(figsize=(15, 12))

for l2 in l2_reg_coefficients:
    weights, val_accuracies = train_logistic_regression(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, l2, 'normal')    
    plt.plot(range(1, epochs + 1), val_accuracies, label=f'Regularization Coefficients: {l2}')

plt.title('Validation Accuracy for Different Regularization Coefficients')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show() """

""" learning_rates = [1e-1, 1e-3, 1e-4, 1e-5]
plt.figure(figsize=(15, 12))

for lr in learning_rates:
    weights, val_accuracies = train_logistic_regression(X_train, y_train, X_val, y_val, epochs, batch_size, lr, l2_reg_coefficient, 'normal')    
    plt.plot(range(1, epochs + 1), val_accuracies, label=f'Learning Rate: {lr}')

plt.title('Validation Accuracy for Different Learning Rates')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()  """

""" batch_sizes = [1, 64, 50000]
plt.figure(figsize=(15, 12))

for bs in batch_sizes:
    weights, val_accuracies = train_logistic_regression(X_train, y_train, X_val, y_val, epochs, bs, learning_rate, l2_reg_coefficient, 'normal')    
    plt.plot(range(1, epochs + 1), val_accuracies, label=f'Batch Size: {bs}')

plt.title('Validation Accuracy for Different Batch Sizes')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()"""

# For Question 2.4
# Code to visualize weights

plt.matshow(final_weights.T, cmap=plt.cm.gray, vmin=0.5 * final_weights.min(), vmax=0.5 * final_weights.max())
plt.show()

# Question 2.5
precision = np.zeros(10)
recall = np.zeros(10)
f1 = np.zeros(10)
f2 = np.zeros(10)

for i in range(10):
    true_positive = conf_matrix[i, i]
    false_positive = np.sum(conf_matrix[:, i]) - true_positive
    false_negative = np.sum(conf_matrix[i, :]) - true_positive

    precision[i] = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall[i] = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0
    f2[i] = 5 * (precision[i] * recall[i]) / (4 * precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0

for i in range(10):
    print(f"Class {i} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1 Score: {f1[i]:.4f}, F2 Score: {f2[i]:.4f}")
