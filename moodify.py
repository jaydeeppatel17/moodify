from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the model
moodDetector = load_model("moodifyEngine.h5")

# Define a dictionary to map class indices to emotion labels
label_dictionary = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Define a helper function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    return normalized.reshape((1,48,48,1))

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    
    # Read the image as a numpy array
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image
    input_data = preprocess_image(image)
    
    # Make a prediction using the model
    prediction = np.argmax(moodDetector.predict(input_data), axis=-1)
    
    # Get the corresponding emotion label
    emotion = label_dictionary.get(prediction[0])
    
    # Return the prediction as a JSON response
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
