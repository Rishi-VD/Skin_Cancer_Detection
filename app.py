import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow import keras
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'model/skin_cancer_model.keras'

try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    class_names = [
        'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma',
        'melanoma', 'nevus', 'pigmented benign keratosis',
        'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
    ]
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    class_names = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Mapping 9 classes to binary Benign/Malignant
binary_mapping = {
    'actinic keratosis': 'Malignant',
    'basal cell carcinoma': 'Malignant',
    'dermatofibroma': 'Benign',
    'melanoma': 'Malignant',
    'nevus': 'Benign',
    'pigmented benign keratosis': 'Benign',
    'seborrheic keratosis': 'Benign',
    'squamous cell carcinoma': 'Malignant',
    'vascular lesion': 'Benign'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        if model:
            processed_image = preprocess_image(filepath)
            prediction = model.predict(processed_image)[0]  # shape (9,)

            # Sum probabilities for benign vs malignant
            malignant_classes = ['actinic keratosis', 'basal cell carcinoma', 'melanoma', 'squamous cell carcinoma']
            benign_classes = ['dermatofibroma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'vascular lesion']

            malignant_prob = sum([prediction[class_names.index(c)] for c in malignant_classes])
            benign_prob = sum([prediction[class_names.index(c)] for c in benign_classes])

            if malignant_prob > benign_prob:
                label = "Malignant"
                confidence = malignant_prob * 100
            else:
                label = "Benign"
                confidence = benign_prob * 100

            return jsonify({
                'prediction': label,
                'confidence': f'{confidence:.2f}%',
                'success': True
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500

    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
