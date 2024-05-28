
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
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
import numpy as np

with open("model_pickle1",'rb') as f:
    model=pickle.load(f)
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

# Initialize the Flask application
app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the file from the POST request
    file = request.files['file']
    
    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'Plants', file.filename)
    file.save(file_path)

    # Load the image and preprocess it
    img = image.load_img(file_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make prediction
    result = model.predict(img)
    predicted_class = np.argmax(result, axis=-1)
    class_indices = training_set.class_indices
    for class_name, index in class_indices.items():
        if index == predicted_class:
            prediction = class_name
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
