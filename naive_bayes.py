import os
from PIL import Image
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Verify images
def verify_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()  # Verify if it is, in fact, an image
                img.close()   # Close the image file after verification
            except (IOError, SyntaxError) as e:
                print(f"Bad file: {file}")
                os.remove(os.path.join(root, file))  # Optionally remove bad files
verify_images('Dataset/Training_set')
verify_images('Dataset/Test_set')

# Function to load images and labels
def load_images(directory, target_size=(64, 64)):
    images = []
    labels = []
    label_map = {}
    label_count = 0

    for root, dirs, files in os.walk(directory):
        for subdir in dirs:
            if subdir not in label_map:
                label_map[subdir] = label_count
                label_count += 1

            subdir_path = os.path.join(root, subdir)
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                try:
                    img = Image.open(file_path)
                    img = img.resize(target_size)
                    img = np.array(img)
                    if img.shape == (64, 64, 3):  # Ensure the image is in the correct shape
                        images.append(img.flatten())
                        labels.append(label_map[subdir])
                except Exception as e:
                    print(f"Failed to process file: {file_path}, Error: {e}")

    return np.array(images), np.array(labels), label_map

# Load training and test images
train_images, train_labels, label_map = load_images('Dataset/Training_set')
test_images, test_labels, _ = load_images('Dataset/Test_set')

# Normalize the pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Split the training data for validation
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(train_images, train_labels)

# Evaluate the classifier
train_predictions = nb_classifier.predict(train_images)
val_predictions = nb_classifier.predict(val_images)
test_predictions = nb_classifier.predict(test_images)

train_accuracy = accuracy_score(train_labels, train_predictions)
val_accuracy = accuracy_score(val_labels, val_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

print("\nClassification Report on Test Set:")
print(classification_report(test_labels, test_predictions, target_names=list(label_map.keys())))

# Making a single prediction
test_image_path = 'Dataset/predict.png'
try:
    test_image = Image.open(test_image_path)
    test_image = test_image.resize((64, 64))
    test_image = np.array(test_image).flatten().reshape(1, -1)
    test_image = test_image / 255.0

    result = nb_classifier.predict(test_image)
    predicted_class = list(label_map.keys())[list(label_map.values()).index(result[0])]
    print("Prediction:", predicted_class)
except Exception as e:
    print(f"Failed to process file: {test_image_path}, Error: {e}")
