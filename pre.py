import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import RMSprop

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('Dataset/training set',
                                                 target_size=(64, 64),
                                                 batch_size=16,
                                                 class_mode='categorical')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/Test set',
                                            target_size=(64, 64),
                                            batch_size=16,
                                            class_mode='categorical')

# Using a pretrained VGG16 model
base_model = VGG16(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Adding custom layers
cnn = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Adding dropout for regularization
    layers.Dense(3, activation='softmax')  # Assuming 3 classes
])
from tensorflow.keras.callbacks import EarlyStopping

# Define the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=5,          # Number of epochs to wait after the last improvement
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)
# Compiling the CNN
cnn.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set, validation_data=test_set, epochs=20,callbacks=[early_stopping])

# Print class indices to verify the mapping
print(training_set.class_indices)

# Making a single prediction
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('dataset/predict3.png', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)

# Print the raw prediction result for debugging
print("Raw prediction result:", result)
test_loss, test_accuracy = cnn.evaluate(test_set)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Map the prediction to class label
predicted_class = np.argmax(result, axis=-1)
class_indices = training_set.class_indices

for class_name, index in class_indices.items():
    if index == predicted_class:
        prediction = class_name

print("Prediction:", prediction)
