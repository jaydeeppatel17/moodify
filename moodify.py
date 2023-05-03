import cv2
import numpy as np
from keras.models import load_model

label_dictionary = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprise"}

# Load the model
moodDetector = load_model("moodifyEngine.h5")

# Load the image
img = cv2.imread('sadladki.jpg')

# Define a function to reshape and rotate the image
def reshape_and_rotate(image):
    W = 48
    H = 48
    image = image.reshape(W, H)
    image = np.flip(image, axis=1)
    image = np.rot90(image)
    return image

# Preprocess image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (48, 48))
normalized = resized / 255.0
n = reshape_and_rotate(normalized)
input_data = n.reshape((1,48,48))

# Make prediction
prediction = np.argmax(moodDetector.predict(input_data), axis=-1)

# Print result
print("The predicted emotion is:", label_dictionary.get(prediction[0]))

# Show image

