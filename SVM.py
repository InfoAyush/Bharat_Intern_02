import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
def load_images_from_folder(folder, label=None):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize images to 64x64
            images.append(img)
            if label is not None:
                labels.append(label)
    return images, labels

# Paths to the dataset folders
cats_folder = 'cats'  # Replace with actual path
dogs_folder = 'dogs'  # Replace with actual path
test_folder = 'test1'  # Replace with actual path to test1 folder
output_folder = 'result'  # Replace with desired output path

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load training images
cat_images, cat_labels = load_images_from_folder(cats_folder, 'cat')
dog_images, dog_labels = load_images_from_folder(dogs_folder, 'dog')

# Combine the training datasets
images = cat_images + dog_images
labels = cat_labels + dog_labels

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize the images
images = images / 255.0

# Flatten the images for SVM
n_samples, height, width = images.shape
images_flatten = images.reshape(n_samples, height * width)

# Encode the labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_flatten, labels_encoded, test_size=0.2, random_state=42)

# Train the SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on validation set: {accuracy:.2f}')
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Load and preprocess the test images
test_images, _ = load_images_from_folder(test_folder)
test_images = np.array(test_images) / 255.0
test_images_flatten = test_images.reshape(len(test_images), height * width)

# Make predictions on the test images
test_predictions = svm.predict(test_images_flatten)

# Convert predictions to labels
test_labels = le.inverse_transform(test_predictions)

# Visualize some test predictions and save the results
def plot_sample_images(images, predictions, output_folder):
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Pred: {predictions[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'test_predictions.png'))
    plt.close()

# Save sample test images with their predictions
plot_sample_images(test_images, test_labels, output_folder)
