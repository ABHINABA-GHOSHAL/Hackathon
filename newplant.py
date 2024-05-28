import pickle
with open("model_pickle1",'rb') as f:
    cnn=pickle.load(f)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from PIL import Image

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('Plants/Train_Set_Folder',
                                                 target_size=(224, 224),  
                                                 batch_size=32,
                                                 class_mode='categorical')

"""### Preprocessing the Test set"""

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('Plants/Validation_Set_Folder',
                                            target_size=(224, 224),  
                                            batch_size=32,
                                            class_mode='categorical')


# Making a single prediction
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('Plants/pred3.jpg', target_size=(224, 224))  # Adjust target_size as necessary
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)

# Print the raw prediction result for debugging
print("Raw prediction result:", result)


# Map the prediction to class label
predicted_class = np.argmax(result, axis=-1)
class_indices = training_set.class_indices

for class_name, index in class_indices.items():
    if index == predicted_class:
        prediction = class_name

print("Prediction:", prediction)
test_loss, test_accuracy = cnn.evaluate(test_set)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
