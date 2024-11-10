import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from datetime import datetime
import time
import zmq
import numpy as np
import socket

from pprint import pprint
import cv2
import json


data = []

# # Create a ZeroMQ context
# context = zmq.Context()
# socket = context.socket(zmq.PUB)  # Use a PUSH socket
# socket.bind("tcp://*:5555")  # Bind to a TCP port


class TCPSender:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def send_message(self, message_dict):
        try:
            message_json = json.dumps(message_dict)
            message_bytes = (message_json + "\n").encode('utf-8')
            self.sock.sendall(message_bytes)
        except Exception as e:
            print(f"Error sending message: {e}")

    def disconnect(self):
        self.sock.close()


sender = TCPSender('localhost', 5555)



BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class BlendShape:

    # these require no additional compute
    mp_unified_translation = {
        # "eyeBlinkLeft": ["aaaaaaaaaaaaaa"], # no equivalent
        # "eyeBlinkRight": ["aaaaaaaaaaaaaa"], # no equivalent
        # "mouthRollLower": ["aaaaaaaaaaaaaa"], # no equivalent
        # "mouthRollUpper": ["aaaaaaaaaaaaaa"], # no equivalent
        # "mouthShrugLower": ["aaaaaaaaaaaaaa"], # no equivalent
        # "mouthShrugUpper": ["aaaaaaaaaaaaaa"], # no equivalent
        "browDownLeft": ["BrowDownLeft"],
        "browDownRight": ["BrowDownRight"],
        "browInnerUp": ["BrowInnerUpRight", "BrowInnerUpLeft"],
        "browOuterUpLeft": ["BrowOuterUpLeft"],
        "browOuterUpRight": ["BrowOuterUpRight"],
        "cheekPuff": ["CheekPuff"], # inacurate detection
        "cheekSquintLeft": ["CheekSquintLeft"],  # inacurate detection
        "cheekSquintRight": ["CheekSquintRight"],  # inacurate detection
        "eyeLookDownLeft": ["EyeLookDownLeft"],
        "eyeLookDownRight": ["EyeLookDownRight"],
        "eyeLookInLeft": ["EyeLookInRight"], # intentional
        "eyeLookInRight": ["EyeLookInLeft"], #
        "eyeLookOutLeft": ["EyeLookOutRight"],# 
        "eyeLookOutRight": ["EyeLookOutLeft"], # intentional
        "eyeLookUpLeft": ["EyeLookUpLeft"],
        "eyeLookUpRight": ["EyeLookUpRight"],
        "eyeSquintLeft": ["EyeSquintLeft"],
        "eyeSquintRight": ["EyeSquintRight"],
        "eyeWideLeft": ["EyeWideLeft"],
        "eyeWideRight": ["EyeWideRight"],
        "jawForward": ["JawForward"], # inacurate detection
        "jawLeft": ["JawLeft"], # inacurate detection
        "jawOpen": ["JawOpen"],
        "jawRight": ["JawRight"], # inacurate detection
        "mouthClose": ["MouthClosed"],
        "mouthDimpleLeft": ["MouthDimpleLeft"], 
        "mouthDimpleRight": ["MouthDimpleRight"],
        "mouthFrownLeft": ["MouthFrownLeft"],
        "mouthFrownRight": ["MouthFrownRight"],
        "mouthFunnel": ["LipFunnel"],
        "mouthLeft": ["MouthLeft"], 

        # some mouth stuff needs to be inverted
        "mouthRight": ["MouthRight"], 
        "mouthPucker": ["LipPucker"],
        "mouthSmileLeft": ["MouthSmileLeft"],
        "mouthSmileRight": ["MouthSmileRight"],
        "mouthStretchLeft": ["MouthStretchLeft"], 
        "mouthStretchRight": ["MouthStretchRight"],
        "noseSneerLeft": ["NoseSneerLeft"], # inmacurate detection
        "noseSneerRight": ["NoseSneerRight"], # inmacurate detection

        "mouthUpperUpLeft": ["MouthUpperUpLeft"], 
        "mouthUpperUpRight": ["MouthUpperUpRight"], 
        "mouthLowerDownLeft": ["MouthLowerDownLeft"],
        "mouthLowerDownRight": ["MouthLowerDownRight"],
        "mouthPressLeft": ["MouthPressLeft"],
        "mouthPressRight": ["MouthPressRight"],
    }

    mp_multiplier = {
        "BrowDownLeft" : 2,
        "BrowDownRight" : 2,
        "BrowInnerUpRight": 1,
        "BrowInnerUpLeft": 1,
        "BrowOuterUpLeft": 1,
        "BrowOuterUpRight": 1,
        "CheekPuff": 1,
        "CheekSquintLeft" : 1,
        "CheekSquintRight" : 1,
        "EyeLookDownLeft": 1,
        "EyeLookDownRight": 1,
        "EyeLookInLeft": 1, 
        "EyeLookInRight": 1,
        "EyeLookOutLeft": 1, 
        "EyeLookOutRight": 1,
        "EyeLookUpLeft": 1.6,
        "EyeLookUpRight": 1.6,
        "EyeSquintLeft": 1,
        "EyeSquintRight": 1,
        "EyeWideLeft": 5, # this can be much much lower but because glasses it is high
        "EyeWideRight": 5, # about 5-10 seems to do the trick wo glassses
        "JawForward": 2,
        "JawLeft": 5,
        "JawOpen": 1,
        "JawRight": 5,
        "MouthClosed": 2,
        "MouthDimpleLeft": 6,
        "MouthDimpleRight": 10,
        "MouthFrownLeft": 5,
        "MouthFrownRight": 5,
        "LipFunnel" : 1.8,
        "MouthLeft": 1.5,
        "MouthRight": 1.5,
        "LipPucker": 0.7,
        "MouthSmileLeft": 3,
        "MouthSmileRight": 3,
        "MouthStretchLeft": 5,
        "MouthStretchRight": 5,
        "NoseSneerLeft": 4,
        "NoseSneerRight": 4,
        "MouthUpperUpLeft": 1,
        "MouthUpperUpRight": 1,
        "MouthLowerDownLeft": 10,
        "MouthLowerDownRight": 10,
        "MouthPressLeft": 5,
        "MouthPressRight": 5,
    }

    mp_offset = {
        "BrowDownLeft" : 0.1,
        "BrowDownRight" : 0.1,
        "BrowInnerUpRight": 0,
        "BrowInnerUpLeft": 0,
        "BrowOuterUpLeft": 0,
        "BrowOuterUpRight": 0,
        "CheekPuff": 0,
        "CheekSquintLeft" : 0, 
        "CheekSquintRight" : 0, 
        "EyeLookDownLeft": 0,
        "EyeLookDownRight": 0,
        "EyeLookInLeft": 0, 
        "EyeLookInRight": 0,
        "EyeLookOutLeft": 0, 
        "EyeLookOutRight": 0, 
        "EyeLookUpLeft": 0,
        "EyeLookUpRight": 0,
        "EyeSquintLeft": 0,
        "EyeSquintRight": 0,
        "EyeWideLeft": 0,
        "EyeWideRight": 0,
        "JawForward": 0.25,
        "JawLeft": 0,
        "JawOpen": 0,
        "JawRight": 0,
        "MouthClosed": 0,
        "MouthDimpleLeft": 0,
        "MouthDimpleRight": 0,
        "MouthFrownLeft": 0,
        "MouthFrownRight": 0,
        "LipFunnel" : 0,
        "MouthLeft": 0,
        "MouthRight": 0,
        "LipPucker": 0,
        "MouthSmileLeft": 0,
        "MouthSmileRight": 0,
        "MouthStretchLeft": 0,
        "MouthStretchRight": 0,
        "NoseSneerLeft": 0,
        "NoseSneerRight": 0,
        "MouthUpperUpLeft": 0,
        "MouthUpperUpRight": 0,
        "MouthLowerDownLeft": 0,
        "MouthLowerDownRight": 0,
        "MouthPressLeft": 0,
        "MouthPressRight": 0,

    }


    mp_bound = {
        "BrowDownLeft" : [0,1],
        "BrowDownRight" : [0,1],
        "BrowInnerUpRight": [0,1],
        "BrowInnerUpLeft": [0,1],
        "BrowOuterUpLeft": [0,1],
        "BrowOuterUpRight": [0,1],
        "CheekPuff": [0,1], 
        "CheekSquintLeft" : [0,1], 
        "CheekSquintRight" : [0,1], 
        "EyeLookDownLeft": [0,1],
        "EyeLookDownRight": [0,1],
        "EyeLookInLeft": [0,1], 
        "EyeLookInRight": [0,1],
        "EyeLookOutLeft": [0,1], 
        "EyeLookOutRight": [0,1],
        "EyeLookUpLeft": [0,1],
        "EyeLookUpRight": [0,1],
        "EyeSquintLeft": [0,1],
        "EyeSquintRight": [0,1],
        "EyeWideLeft": [0,1],
        "EyeWideRight": [0,1],
        "JawForward": [0,1],
        "JawLeft": [0,1],
        "JawOpen": [0,1],
        "JawRight": [0,1],
        "MouthClosed": [0,1],
        "MouthDimpleLeft": [0,1],
        "MouthDimpleRight": [0,1],
        "MouthFrownLeft": [0,1],
        "MouthFrownRight": [0,1],
        "LipFunnel" : [0,1],
        "MouthLeft": [0,1],
        "MouthRight": [0,1],
        "LipPucker": [0,1], 
        "MouthSmileLeft": [0,1],
        "MouthSmileRight": [0,1],
        "MouthStretchLeft": [0,1],
        "MouthStretchRight": [0,1],
        "NoseSneerLeft": [0,1],
        "NoseSneerRight": [0,1],
        "MouthUpperUpLeft": [0,1],
        "MouthUpperUpRight": [0,1],
        "MouthLowerDownLeft": [0,1],
        "MouthLowerDownRight": [0,1],
        "MouthPressLeft": [0,1],
        "MouthPressRight": [0,1],

    }


    def bound(self, low, high, value):
        return max(low, min(high, value))

    



    def __init__(self, init_obj: FaceLandmarkerResult):
        self.categories = {}
        categories = init_obj.face_blendshapes[0]
        for category in categories:
            self.categories[category.category_name] = category.score

    def serialize(self):
        translated = {}
        for key in self.mp_unified_translation.keys():
            categories = self.mp_unified_translation[key]
            for cat in categories:
                translated[cat] = self.bound(
                    self.mp_bound.get(cat, [0,1])[0],
                    self.mp_bound.get(cat, [0,1])[1],
                    self.categories[key] * self.mp_multiplier.get(cat, 1) + self.mp_offset.get(cat, 0)
                    )

        return json.dumps(translated)


class MPFace:

    def __init__(self, capture: int, modelpath: str, fps=60):
        self.capture = cv2.VideoCapture(capture) # opencv webcam feed
        self.model_path = modelpath
        self.fps = fps
        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=modelpath),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.process,
            output_face_blendshapes=True,
            )
        self.landmarker = FaceLandmarker.create_from_options(self.options)

    def process(self, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        try:
            if result.face_blendshapes:
                data = BlendShape(result).serialize()
                sender.connect()
                sender.send_message(data)
                sender.disconnect()
    
                print("sent values")
        except Exception as e:
            print(e)
            return False
        

    def gen_mp_frame(self, capture: cv2.VideoCapture):
        
        ret, frame = capture.read()
        if not ret:
            return
        cv2.flip(frame, 1)
        # could just specify 1440p and then select midway point as center
        # height, width = frame.shape[:2]
        # square_size = min(width, height)
        # x1 = (width - square_size) // 2
        # y1 = (height - square_size) // 2
        # square_frame = frame[y1:y1 + square_size, x1:x1 + square_size]
        # square_frame = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)  # Convert color from BGR to RGB
        # square_frame = np.array(square_frame, dtype=np.uint8)  # Ensure it's the correct type
        square_frame = frame
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=square_frame)
        return mp_frame

    def run(self):
        with self.landmarker as landmarker:
            while True:
                try:
                    landmarker.detect_async(self.gen_mp_frame(self.capture), int(time.time() * 1000))
                    time.sleep(1 / self.fps)
                    
                except KeyboardInterrupt:
                    break

cap = 0
model_path = "./face/face_landmarker.task"
MPface = MPFace(capture=cap, modelpath=model_path, fps=90)

try:
    MPface.run()
except KeyboardInterrupt:
    pass

# socket.close()
# context.term()
# context.destroy()