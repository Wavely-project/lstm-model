
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

app = Flask(__name__)

# Load the model
model = load_model('action.h5')
actions = np.array(['hello', 'thanks', 'iloveyou'])

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilitiesactions = np.array(['hello', 'thanks', 'iloveyou'])

# Function to make detections using MediaPipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

# Function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def get_prediction(video_path):
    # Read the video file
    cap = cv2.VideoCapture(video_path)
    sequence = []
    print("start.")
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = mediapipe_detection(frame, holistic)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

    # Make predictions and return the result
    prediction_index = np.argmax(res)
    prediction = actions[prediction_index]
    print("end")
    return jsonify({'prediction': prediction})

app = Flask(__name__) 

@app.route('/', methods=['GET'])
def test():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    print("in predict")
    videofile = request.files['videofile']
    # print("")
    video_path = "./videos/" + videofile.filename
    videofile.save(video_path)



    return get_prediction(video_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

