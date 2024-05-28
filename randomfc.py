import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Verify images
def verify_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()  # Verify if it is, in fact, an image
            except (IOError, SyntaxError) as e:
                print(f"Bad file: {file}")
                os.remove(os.path.join(root, file))  # Optionally remove bad files

verify_images('Dataset/Training_set')
verify_images('Dataset/Test_set')

# Preprocess images and extract features using VGG16
def preprocess_images(data_dir, target_size=(64, 64)):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=32,
        class_mode=None,  # This means our generator will only yield batches of data, no labels
        shuffle=False  # Keep data in same order as labels
    )

    model = VGG16(weights='imagenet', include_top=False)
    features = model.predict(generator, verbose=1)
    labels = generator.classes
    return features, labels

# Load and preprocess data
train_features, train_labels = preprocess_images('Dataset/Training_set')
test_features, test_labels = preprocess_images('Dataset/Test_set')

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_classifier.fit(train_features.reshape(len(train_features), -1), train_labels)

# Evaluate on test set
test_predictions = rf_classifier.predict(test_features.reshape(len(test_features), -1))
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(classification_report(test_labels, test_predictions))

# Make a single prediction
def predict_single_image(image_path, model, label_encoder):
    img = Image.open(image_path)
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    features = model.predict(img)
    prediction = label_encoder.inverse_transform(rf_classifier.predict(features.reshape(1, -1)))
    return prediction[0]

# Example usage for single prediction
label_map = {0: 'cheetah', 1: 'fox', 2: 'hyena'}
label_encoder = LabelEncoder()
label_encoder.fit(list(label_map.values()))

test_image_path = 'Dataset/predict.png'
prediction = predict_single_image(test_image_path, model, label_encoder)
print("Prediction:", prediction)
