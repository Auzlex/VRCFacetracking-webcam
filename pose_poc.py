import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from datetime import datetime
import time 
import zmq
import numpy as np

from pprint import pprint
import cv2
import json 


data = []

# Create a ZeroMQ context
context = zmq.Context()
socket = context.socket(zmq.PUB)  # Use a PUSH socket
socket.bind("tcp://*:5555")  # Bind to a TCP port



def gen_json_points(result: mp.tasks.vision.PoseLandmarkerResult) -> json:
    data = {}
    for i in [11,12,13,14,15,16,23,24]:
        data[i] = [
            result.pose_world_landmarks[0][i].x,
            result.pose_world_landmarks[0][i].y,
            result.pose_world_landmarks[0][i].z, 
            result.pose_world_landmarks[0][i].visibility, 
            ]
    return json.dumps(data)

def gen_mp_frame(capture: cv2.VideoCapture):
    ret, frame = capture.read()
    if not ret:
        return
    # could just specify 1440p and then select midway point as center
    height, width = frame.shape[:2]
    square_size = min(width, height)
    x1 = (width - square_size) // 2
    y1 = (height - square_size) // 2
    square_frame = frame[y1:y1 + square_size, x1:x1 + square_size]
    square_frame = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)  # Convert color from BGR to RGB
    square_frame = np.array(square_frame, dtype=np.uint8)  # Ensure it's the correct type  

    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=square_frame)
    return mp_frame


cap = cv2.VideoCapture(0)
model_path = './pose/pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the live stream mode:
def pose_result_callback(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    points_send = gen_json_points(result)
    data.append(points_send)
    socket.send_string(points_send)


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=pose_result_callback,
    num_poses=1,
    )



fps = 60
with PoseLandmarker.create_from_options(options) as landmarker:
    for i in range(100000):
        landmarker.detect_async(gen_mp_frame(cap), int(time.time() * 1000))
        print(i)
        time.sleep(1/fps)


socket.close()
context.term()
context.destroy()

