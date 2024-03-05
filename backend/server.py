# app.py (Flask App)
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from detector import predict_gesture
import base64

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(frame):
    try:
        gesture, confidence = predict_gesture(frame)
        emit('gesture', {'gesture': gesture, 'confidence': str(confidence)})
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    socketio.run(app, debug=True)
