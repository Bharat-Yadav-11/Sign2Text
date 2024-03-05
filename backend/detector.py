# import cv2
# import mediapipe
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# tf.get_logger().setLevel('ERROR')

# # Load the classification model and labels
# model = load_model('detector\hand_gestures_model.h5')
# labels = ['A', 'B', 'C']  # Load from gestures.txt or use as per your file

# camera = cv2.VideoCapture(0)

# mediapipe_hands = mediapipe.solutions.hands
# hands = mediapipe_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# media_draw = mediapipe.solutions.drawing_utils

# letter_count = {'A': 0, 'B': 0, 'C': 0}  # Count of captured images for each letter

# while True:

#     ret, image = camera.read()
#     if not ret:
#         print("No camera found, exiting...")
#         exit(1)

#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     results = hands.process(image)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw hand landmarks
#             media_draw.draw_landmarks(image, hand_landmarks, mediapipe_hands.HAND_CONNECTIONS)
#             # Get hand bounding box 
#             landmarks_x = [landmark.x for landmark in hand_landmarks.landmark]
#             landmarks_y = [landmark.y for landmark in hand_landmarks.landmark]
#             x_min = min(landmarks_x)
#             x_max = max(landmarks_x)
#             y_min = min(landmarks_y)
#             y_max = max(landmarks_y)
            
#             # Adding padding to the bounding box
#             padding = 0.1
            
#             x_min = max(0, int((x_min - padding) * image.shape[1]))
#             x_max = min(image.shape[1], int((x_max + padding) * image.shape[1]))
#             y_min = max(0, int((y_min - padding) * image.shape[0]))
#             y_max = min(image.shape[0], int((y_max + padding) * image.shape[0]))
                        
#             # Crop hand from the image considering both hands
#             hand_image = image[y_min:y_max, x_min:x_max]

#             # Preprocess the cropped hand image
#             hand_image = cv2.resize(hand_image, (224, 224))  # Resize to match model input shape
#             hand_image = hand_image / 255.0  # Normalize pixel 
            
#             cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#             # Predict gesture using the model
#             prediction = model.predict(np.expand_dims(hand_image, axis=0))
#             confidence = np.max(prediction)
#             predicted_label = labels[np.argmax(prediction)]

#             # Convert the hand_image back to the range [0, 255]
#             hand_image_display = (hand_image * 255).astype(np.uint8)

#             # Display the cropped hand image
#             cv2.imshow('Hand', cv2.cvtColor(hand_image_display, cv2.COLOR_RGB2BGR))

            
#             if confidence < 0.8:
#                 predicted_label = "The ASL gesture is not recognized"
                
#             print(f"""Predicted label: {predicted_label if confidence > 0.8 else 'None'} with confidence: {confidence.round(2)}""")

            
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# hands.close()
# camera.release()
# cv2.destroyAllWindows()
import os
import io
import cv2
import mediapipe
import numpy as np
from tensorflow.keras.models import load_model



def predict_gesture(frame):
    print("Predicting gesture...")

    # Convert the binary buffer to an cv2 image
    image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Load the classification model and labels
    model = load_model('backend\hand_gestures_model.h5', compile=False)
    labels = ['A', 'B', 'C']  # Load from gestures.txt or use as per your file

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

            if confidence < 0.8:
                predicted_label = "The ASL gesture is not recognized"

    else:
        print("No hand detected")
    hands.close()
    return predicted_label, confidence

if __name__ == "__main__":
    # Example of usage
    # Capture a frame from the camera or load it from a video file
    frame = cv2.imread('backend/0.jpg')  # Replace 'frame.jpg' with your frame path
    predicted_label, confidence = predict_gesture(frame)
    print(f"Predicted label: {predicted_label}, Confidence: {confidence}")
