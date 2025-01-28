from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import io

app = Flask(__name__)  # Corrected line

# Load your trained model
model = load_model('pneumonia_detection_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Read the file content and convert to a BytesIO object
        file_content = file.read()
        img = load_img(io.BytesIO(file_content), target_size=(256, 256))
        
        # Preprocess the image
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict using the model
        result = model.predict(img_array)
        class_probabilities = result[0]
        
        if class_probabilities[0] > class_probabilities[1]:
            prediction = "NORMAL"
            message = "Don't worry. You are Healthy!!!"
        else:
            prediction = "PNEUMONIA"
            message = "This is just a Prediction! Take Precautions & Please consult the Doctor for the Confirmation."
            
        return render_template('result.html', prediction=prediction, message=message)

if __name__ == '__main__':  # Corrected line
    app.run(debug=True)
