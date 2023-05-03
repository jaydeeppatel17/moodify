from keras.models import load_model
import cv2
from google.colab.patches import cv2_imshow
label_dictionary={0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Neutral",5:"Sad",6:"Surprise"}
moodDetector = load_model("moodifyEngine.h5")
img = cv2.imread('/content/exp.jpeg')
# img = cv2.imread('/content/Dataset/FER_2013/test/surprise/PrivateTest_10089743.jpg')
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
print(moodDetector.predict(input_data))
# Make prediction
prediction = np.argmax(moodDetector.predict(input_data), axis=-1)

# Print result
print("The predicted emotion is:", label_dictionary.get(prediction[0]))

# Show image
cv2_imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
