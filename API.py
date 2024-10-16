from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import io
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np


app = Flask(__name__)
swagger = Swagger(app)

# Загрузка моделей
heart_model = joblib.load('Models\\classification\\heart_model.pkl')
diabetes_model = joblib.load('Models\\classification\\diabetes_model.pkl')
brain_tumor_model = load_model('Models\\cnn\\brain_tumor_model.h5')
pneumonia_model = load_model('Models\\cnn\\pneumonia_model.h5')
# model3 = joblib.load('model3.pkl')

@app.route('/predict_image/brain_tumor', methods=['POST'])
def predict_image_brain():
    labels = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img_bytes = io.BytesIO(file.read())
    img = keras_image.load_img(img_bytes, target_size=(128, 128))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = brain_tumor_model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    probabilities = dict(zip(labels, [round(float(num), 2) for num in predictions[0]]))
    return jsonify({
        'predicted_class': labels[int(predicted_class)],
        'probabilities': probabilities
    }), 200

@app.route('/predict_image/pneumonia', methods=['POST'])
def predict_image_pneumonia():
    labels = ['normal', 'pneumonia']

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img_bytes = io.BytesIO(file.read())
    img = keras_image.load_img(img_bytes, target_size=(128, 128))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = pneumonia_model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    probabilities = dict(zip(labels, [round(float(num), 2) for num in predictions[0]]))
    return jsonify({
        'predicted_class': labels[int(predicted_class)],
        'probabilities': probabilities
    }), 200

@app.route('/predict_params/heart_attack', methods=['POST'])
def predict_params_heart():
    labels = ['normal', 'risk']
    feature_names = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
       'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
       'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
       'ST_Slope_Flat', 'ST_Slope_Up']

    data = request.json
    if 'parameters' not in data:
        return jsonify({'error': 'No parameters provided'}), 400

    try:
        parameters = data['parameters']
        input_data = pd.DataFrame([parameters],
                                  columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                                           'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])

        input_data_encoded = pd.get_dummies(input_data)

        missing_cols = set(feature_names) - set(input_data_encoded.columns)
        for col in missing_cols:
            input_data_encoded[col] = False
        input_data_encoded = input_data_encoded[feature_names]

        predictions = heart_model.predict_proba(input_data_encoded)
        predicted_class = np.argmax(predictions[0])

        probabilities = dict(zip(labels, [round(float(num), 2) for num in predictions[0]]))
        return jsonify({
            'predicted_class': labels[int(predicted_class)],
            'probabilities': probabilities
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_params/diabetes', methods=['POST'])
def predict_params_diabetes():
    labels = ['normal', 'risk']
    data = request.json
    if 'parameters' not in data:
        return jsonify({'error': 'No parameters provided'}), 400

    try:
        parameters = data['parameters']
        input_data = np.array(parameters).reshape(1, -1)
        predictions = diabetes_model.predict_proba(input_data)
        predicted_class = np.argmax(predictions[0])

        probabilities = dict(zip(labels, [round(float(num), 2) for num in predictions[0]]))
        return jsonify({
            'predicted_class': labels[int(predicted_class)],
            'probabilities': probabilities
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)