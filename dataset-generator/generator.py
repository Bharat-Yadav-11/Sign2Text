# import cv2
# import mediapipe
# import os
# import time

# # Create a directory to save the images if it doesn't exist
# if not os.path.exists("data"):
#     os.makedirs("data")

# camera = cv2.VideoCapture(0)

# mediapipe_hands = mediapipe.solutions.hands
# hands = mediapipe_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# media_draw = mediapipe.solutions.drawing_utils

# # Loop through each letter
# for letter in ['A', 'B', 'C']:
#     print(f"Please make the ASL sign for {letter}")
#     time.sleep(5)  # Wait for 5 seconds
#     print("Capturing images...")
    
#     for _ in range(300):
#         ret, image = camera.read()
#         if not ret:
#             continue

#         image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#         results = hands.process(image)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Draw hand landmarks
#                 media_draw.draw_landmarks(image, hand_landmarks, mediapipe_hands.HAND_CONNECTIONS)
#                 # Get hand bounding box 
#                 bbox = mediapipe_hands.HandLandmark
#                 x_min = x_max = hand_landmarks.landmark[bbox.WRIST].x
#                 y_min = y_max = hand_landmarks.landmark[bbox.WRIST].y
#                 for landmark in hand_landmarks.landmark:
#                     x_min = min(x_min, landmark.x)
#                     x_max = max(x_max, landmark.x)
#                     y_min = min(y_min, landmark.y)
#                     y_max = max(y_max, landmark.y)
                
#                 # Adding padding to the bounding box
#                 padding = 0.1
                
#                 x_min = max(0, int((x_min - padding) * image.shape[1]))
#                 x_max = min(image.shape[1], int((x_max + padding) * image.shape[1]))
#                 y_min = max(0, int((y_min - padding) * image.shape[0]))
#                 y_max = min(image.shape[0], int((y_max + padding) * image.shape[0]))
                            
#                 # Crop hand from the image considering both hands
#                 hand_image = image[y_min:y_max, x_min:x_max]

#                 # Draw bounding box
#                 cv2.rectangle(hand_image, (0, 0), (hand_image.shape[1], hand_image.shape[0]), (0, 255, 0), 2)

#                 # Display the cropped hand with landmarks and bounding box
#                 cv2.imshow('Sign2Text', cv2.resize(cv2.cvtColor(hand_image, cv2.COLOR_RGB2BGR), (224, 224)))
                
#                 # Save images in high quality and color
#                 if letter_count[letter] < 100:
#                     # Save image without bounding box
#                     filename = f"data/{letter}_{letter_count[letter]}.jpg"            
#                     cv2.imwrite(filename, cv2.cvtColor(hand_image, cv2.COLOR_RGB2BGR))
#                     letter_count[letter] += 1

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# hands.close()
# camera.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe
import os
import time
import glob

# Create a directory to save the images if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Initialize MediaPipe Hands
mediapipe_hands = mediapipe.solutions.hands
hands = mediapipe_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

media_draw = mediapipe.solutions.drawing_utils

# Get the paths to the training images for each letter
train_dir = "ASL_Alphabet_Dataset/asl_alphabet_train"
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

for letter in letters:
    counter = 1
     
    if not os.path.exists(f"data/{letter}"):
         os.makedirs(f"data/{letter}")
        
    images_path = os.path.join(train_dir, letter, "*.jpg")
    images = glob.glob(images_path)
    for img_path in images:
        print(f"Processing image: {img_path}")
        image = cv2.imread(img_path)
        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect hands
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                media_draw.draw_landmarks(image, hand_landmarks, mediapipe_hands.HAND_CONNECTIONS)
                # Get hand bounding box 
                bbox = mediapipe_hands.HandLandmark
                x_min = x_max = hand_landmarks.landmark[bbox.WRIST].x
                y_min = y_max = hand_landmarks.landmark[bbox.WRIST].y
                for landmark in hand_landmarks.landmark:
                    x_min = min(x_min, landmark.x)
                    x_max = max(x_max, landmark.x)
                    y_min = min(y_min, landmark.y)
                    y_max = max(y_max, landmark.y)
                
                # Adding padding to the bounding box
                padding = 0.1
                
                x_min = max(0, int((x_min - padding) * image.shape[1]))
                x_max = min(image.shape[1], int((x_max + padding) * image.shape[1]))
                y_min = max(0, int((y_min - padding) * image.shape[0]))
                y_max = min(image.shape[0], int((y_max + padding) * image.shape[0]))
                            
                # Crop hand from the image considering both hands
                hand_image = image[y_min:y_max, x_min:x_max]

                # Draw bounding box
                cv2.rectangle(hand_image, (0, 0), (hand_image.shape[1], hand_image.shape[0]), (0, 255, 0), 2)

                
                filename = f"data/{letter}/{letter}_{counter}.jpg"
                counter += 1
                # cv2.imwrite(filename, cv2.cvtColor(hand_image, cv2.COLOR_RGB2BGR))
                
                # Resize the saved image to 244x244
                resized_image = cv2.resize(hand_image, (244, 244))
                cv2.imwrite(filename, cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
        
hands.close()
