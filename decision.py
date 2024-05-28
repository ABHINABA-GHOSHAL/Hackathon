import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the VGG16 model for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features using VGG16
def extract_features(data_generator, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # VGG16 output shape
    labels = np.zeros(shape=(sample_count, 6))  # Number of classes
    
    i = 0
    for inputs_batch, labels_batch in data_generator:
        features_batch = base_model.predict(inputs_batch)
        features[i * 32 : (i + 1) * 32] = features_batch
        labels[i * 32 : (i + 1) * 32] = labels_batch
        i += 1
        if i * 32 >= sample_count:
            break
    return features.reshape(sample_count, 7 * 7 * 512), labels

# Initialize the ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the dataset
training_set = train_datagen.flow_from_directory('Dataset/Training_set',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory('Dataset/Test_set',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')

# Extract features for training and test sets
train_features, train_labels = extract_features(training_set, len(training_set.filenames))
test_features, test_labels = extract_features(test_set, len(test_set.filenames))

# Initialize Decision Tree Classifier
decision_tree = DecisionTreeClassifier()

# Train Decision Tree on extracted features
decision_tree.fit(train_features, np.argmax(train_labels, axis=1))

# Evaluate the Decision Tree Classifier
train_predictions = decision_tree.predict(train_features)
test_predictions = decision_tree.predict(test_features)

train_accuracy = accuracy_score(np.argmax(train_labels, axis=1), train_predictions)
test_accuracy = accuracy_score(np.argmax(test_labels, axis=1), test_predictions)

print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

print("\nClassification Report on Test Set:")
print(classification_report(np.argmax(test_labels, axis=1), test_predictions))
