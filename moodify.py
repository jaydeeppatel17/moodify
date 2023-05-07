import uvicorn
import numpy as np
import cv2
import requests
import json
import io
from fastapi import FastAPI, File, UploadFile
from keras.models import load_model

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

# Define the prediction function
def predict(image):
    # Preprocess the image
    input_data = preprocess_image(image)
    
    # Make a prediction using the model
    prediction = np.argmax(moodDetector.predict(input_data), axis=-1)
    
    # Get the corresponding emotion label
    emotion = label_dictionary.get(prediction[0])
    
    return emotion

# Define the FastAPI app
app = FastAPI()

# Define a route to handle file upload
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image as a bytes object
    contents = await file.read()
    image = np.asarray(bytearray(contents), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Make a prediction using the Flask code
    emotion = predict(image)
    
    # Return the prediction as JSON
    return {"emotion": emotion}

# Define the Streamlit app
def st_app():
    st.title('Moodify Engine')
    st.write('Upload an image and the Moodify Engine will detect the emotion in the image!')
    
    # Create a file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read the image as a numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Upload the image to the FastAPI server
        response = requests.post("http://localhost:8000/predict", files={"file": uploaded_file})
        prediction = json.loads(response.content.decode('utf-8'))
        
        # Show the result
        st.write('The emotion detected in the image is:', prediction["emotion"])

if __name__ == '__main__':
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Run the Streamlit app
    st_app()
