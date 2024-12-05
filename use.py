from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)


model = tf.keras.models.load_model('h5_plant_disease_model.h5')


image_size = (128, 128) 


class_labels = [
    'Apple Scab Disease',
    'Black Rot Disease',
    'Cedar Apple rust Disease',
    'Healthy Apple Leaf',
    'Healthy BlueBerry',
    'Cherry (including sour) - Healthy',
    'Cherry (including sour) - Powdery mildew',
    'Corn (maize) - Cercospora leaf spot Gray leaf spot',
    'Corn (maize) - Common rust',
    'Healthy Corn Maize',
    'Corn Northern Leaf Blind',
    'Grape Black rot',
    'Grape Esca',
    'Healthy Grape',
    'Grape Leaf Blind',
    'Orange Huanglongbing (Citrus greening)',
    'Peach Bacterial spot',
    'Healthy Peach',
    'Peach Bell Bacterial',
    'Peach Bell Healthy',
    'Potato Early Blight',
    'Potato Late Blight'
]


def preprocess_image(image):
    image = image.convert("RGB")  
    image = image.resize(image_size)  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0) 
    return image


def predict_image(image):
    processed_image = preprocess_image(image) 
    predictions = model.predict(processed_image)  
    predicted_class_index = np.argmax(predictions, axis=1)[0] 
    predicted_class_label = class_labels[predicted_class_index]  
    return predicted_class_label


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    if request.method == 'POST':
       
        file = request.files['image']
        if file:
            
            image = Image.open(file.stream) 
            prediction = predict_image(image)  
    return render_template('base.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
