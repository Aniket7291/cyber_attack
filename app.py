from flask import Flask, render_template, request, jsonify
import h5py
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('cyber_attack_detection.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json['data']

    # Preprocess the data (you might need to adjust this based on your model)
    data = np.array(data)
    data = data.reshape(1, -1)  # Reshape the data if neededr̥

    # Make prediction
    prediction = model.predict(data)

    # Convert prediction to human-readable format (optional)
    # For example, if it's binary classification, you might convert 0/1 to 'Normal'/'Attack'

    # Return the prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
