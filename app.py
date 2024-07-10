from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from flask_cors import CORS
import subprocess
import shutil
import warnings
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
warnings.filterwarnings('ignore')


app = Flask(__name__)
CORS(app)

# Load the model
model = load_model('./LSTM29.h5')
actions = np.array(['drink','eat','friend','goodbye','hello','help','how are you','no','yes','please','sorry','thanks','cry','i','they','you','what','name','teacher','family','happy','love','sad','laugh','neighbor','ok','read','write','school'])

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

def get_prediction(video_path: str) -> jsonify:
    cap = cv2.VideoCapture(video_path)
    start_time = time.time()
    print('start')
    print("total frames count: ",int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open the video file'}), 400

    mp_holistic = mp.solutions.holistic  # MediaPipe Holistic model
    sequence = []
    predictions = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            if keypoints is not None:
                # Only take the keypoints for the hands (assuming this from your original code)
                lh_rh = keypoints[1536:]
                lh_rh = lh_rh.reshape(-1, 3)[:, :2].flatten()
                sequence.append(lh_rh)
        print("sequence length: ",len(sequence))
        if len(sequence) > 30:
            sequence = sequence[10:40]

        if sequence:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions = actions[np.argmax(res)]
            print(f"Model prediction made. {predictions}")
        else:
            return jsonify({'error': 'No valid keypoints extracted'}), 400

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("end.\nTotal time taken: {:.2f} seconds".format(elapsed_time))
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'prediction': predictions})

@app.route('/', methods=['GET'])
def test():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    print("in predict")
    videofile = request.files['videofile']
    # print("")
    video_path = "./videos/" + videofile.filename
    try:
        videofile.save(video_path)
        app.logger.info(f"File saved successfully at {video_path}")
        
    except Exception as e:
        app.logger.error(f"Failed to save file: {e}")
        return jsonify({'error': 'Failed to save file'}), 500

    return get_prediction(video_path)

@app.errorhandler(500)
def handle_500_error(exception):
    app.logger.error(f"Server Error: {exception}")
    return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(port=3006, debug=True, host='0.0.0.0')

