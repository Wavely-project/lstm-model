
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
model = load_model('./model_lstm_6_classes_0.98.h5')
actions = np.array(['drink', 'eat', 'goodbye', 'hello', 'help', 'how are you'])

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

    # Load the model and define other necessary variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    # Load the video file
    cap = cv2.VideoCapture(video_path)
    start_time = time.time()
    print('start')
    print("total frames count: ",int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        # If there are no more frames, break the loop
        if not ret:
            print("No more frames. Exiting loop.")
            break

        # print("Frame read successfully.")

        # Make detections
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = mediapipe_detection(frame, holistic)
            # print("Detections made.")

        # Prediction logic
        keypoints = extract_keypoints(results)
        lh_rh = keypoints[1536:]
        #Remove z Axis From Landmarks
        for z in range(2,lh_rh.shape[0],3):
            lh_rh[z] = None
        #Remove NaN Data
        lh_rh = lh_rh[np.logical_not(np.isnan(lh_rh))]
        sequence.append(lh_rh)
        sequence = sequence[-30:]
        # print("keypoints extracted.")
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(f"Model prediction made. {actions[np.argmax(res)]}")
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
            # sequence = []
            # if len(sentence) > 5:
            #     sentence = sentence[-5:]

        

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("end.\nTotal time taken: {:.2f} seconds".format(elapsed_time))
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

   
    return jsonify({'prediction': sentence})

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

