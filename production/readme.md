# Sign2Text | A Gesture Recognition application with Flask and Machine Learning

This project demonstrates real-time hand gesture recognition using Flask, Mediapipe, and a pre-trained TensorFlow model. It captures frames from the client-side camera, processes them using the Mediapipe library to detect hands, and then predicts the gesture using a pre-trained model.

## Installation

1. Clone the repository:

```
git clone https://github.com/Om-Mishra7/Sign2Text.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Run the Flask app:

```
python app.py
```

In case the above command does not work, try:

```
python3 app.py
```

2. Open a web browser and go to `http://localhost:5000`.

3. Allow access to your camera if prompted.

4. Your camera feed will appear on the web page. Perform hand gestures in front of the camera, and the predicted gesture will be displayed on the webpage in real-time.


## Troubleshooting

- If you encounter any issues with dependencies, make sure you have installed them correctly by referring to the `requirements.txt` file.
- If the camera feed does not appear or the hand gestures are not recognized, check if your camera is properly connected and the lighting conditions are suitable for hand detection.

In case of any other issues, feel free to open an issue on the repository.
