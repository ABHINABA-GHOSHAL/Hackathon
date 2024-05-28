
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from PIL import Image




def verify_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                
                # Convert palette images with transparency to RGBA
                if img.mode == 'P':
                    img = img.convert('RGBA')
                    img.save(img_path)
                    img.close()
                    img = Image.open(img_path)
                    
                img.verify()  # Verify if it is, in fact, an image
                img.close()
                
            except (IOError, SyntaxError) as e:
                print(f"Bad file: {file}")
                os.remove(img_path)  # Optionally remove bad files

# Usage
verify_images('Dataset/Training_set')
verify_images('Dataset/Test_set')

# Your other code

tf.__version__


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('Dataset/Training_set',
                                                 target_size=(224, 224),  
                                                 batch_size=32,
                                                 class_mode='categorical')


test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('Dataset/Test_set',
                                            target_size=(224, 224),  
                                            batch_size=32,
                                            class_mode='categorical')



base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base VGG16 model
for layer in base_model.layers:
    layer.trainable = False

# Customizing the top layers of the model
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(6, activation='softmax')(x)

# Create the final model
cnn = Model(inputs=base_model.input, outputs=output)

"""## Part 3 - Training the CNN

### Compiling the CNN
"""

# Define the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5,         
    restore_best_weights=True 
)

cnn.compile(optimizer=RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


cnn.fit(x=training_set, validation_data=test_set, epochs=20, callbacks=[early_stopping])
with open("model_pickle",'wb') as f:
    pickle.dump(cnn,f)


# Print class indices to verify the mapping
print(training_set.class_indices)

# Making a single prediction
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('Dataset/predict.png', target_size=(224, 224))  # Adjust target_size as necessary
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
