import cv2
import mediapipe
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model



def predict_gesture(frame):
    print("Predicting gesture...")

    # Convert the binary buffer to an cv2 image
    image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Load the classification model and labels
    model = load_model('hand_gestures_model.h5', compile=False)
    labels = []
    with open('gestures.txt', 'r') as file:
        labels = file.readlines()
    labels = [label.strip() for label in labels]
    
    mediapipe_hands = mediapipe.solutions.hands
    hands = mediapipe_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)

    media_draw = mediapipe.solutions.drawing_utils
    
    results = hands.process(image)

    predicted_label = None
    confidence = 0.0
    
    if results.multi_hand_landmarks:
        print("Hand detected")
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand bounding box 
            media_draw.draw_landmarks(image, hand_landmarks, mediapipe_hands.HAND_CONNECTIONS)
            landmarks_x = [landmark.x for landmark in hand_landmarks.landmark]
            landmarks_y = [landmark.y for landmark in hand_landmarks.landmark]
            x_min = min(landmarks_x)
            x_max = max(landmarks_x)
            y_min = min(landmarks_y)
            y_max = max(landmarks_y)
            
            # Adding padding to the bounding box
            padding = 0.1
            
            x_min = max(0, int((x_min - padding) * image.shape[1]))
            x_max = min(image.shape[1], int((x_max + padding) * image.shape[1]))
            y_min = max(0, int((y_min - padding) * image.shape[0]))
            y_max = min(image.shape[0], int((y_max + padding) * image.shape[0]))
                        
            # Crop hand from the image considering both hands
            hand_image = image[y_min:y_max, x_min:x_max]

            # Preprocess the cropped hand image
            hand_image = cv2.resize(hand_image, (224, 224))  # Resize to match model input shape
            hand_image = hand_image / 255.0  # Normalize pixel 

            # Predict gesture using the model
            prediction = model.predict(np.expand_dims(hand_image, axis=0))
            confidence = np.max(prediction)
            predicted_label = labels[np.argmax(prediction)]

            if confidence < 0.5:
                predicted_label = "The ASL gesture is not recognized"

    else:
        print("No hand detected")
    hands.close()
    return predicted_label, confidence


app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(frame):
    try:
        gesture, confidence = predict_gesture(frame)
        emit('gesture', {'gesture': gesture, 'confidence': int(confidence * 100)})
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
