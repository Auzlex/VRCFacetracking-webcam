import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from datetime import datetime
import time
import numpy as np
import socket

from pprint import pprint
import cv2
import json
import json5 # handle json docs with comments


data = []




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


class BlendShapeParams:
    # translation from mediapipe to unified keys
    unified_mp_translation = {
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


    path = ""
    tuning = {} 
    params = {}

    def __init__(self, tuning_params_path: str) -> None:
        self.path = tuning_params_path
        self.tuning = json5.load(open(tuning_params_path, "r"))


    def bound(self, low: float, high: float, value: float) -> float:
        return max(low, min(high, value))   
    
    def min(self, category: str)-> float:
        return self.tuning[category]["min"]

    def max(self, category: str)-> float:
        return self.tuning[category]["max"]
    
    def mul(self, category: str)-> float:
        return self.tuning[category]["mul"]
    
    def offset(self, category: str)-> float:
        return self.tuning[category]["offset"]
    
    def custom_funct(self, category: str)-> str:
        return self.tuning[category]["cfunct"]
    
    def call_custom_funct(self, cfunct: str, value: float)-> float:
        func = locals().get(cfunct)
        if callable(func):
            return func(value)
        else:
            return 0
    
    
    def update(self, result: FaceLandmarkerResult) -> dict:
        # process LFMR for easy access
        result_dict = {}
        for param in result.face_blendshapes[0]:
            result_dict[param.category_name] = param.score

        # some mp keys need to be mapped to multiple unified keys
        # might do the reverse to remove a single loop for readability
        # check if a custom function is needed for processing
        # if not, use provided multipliers and offsets
        for mpk, uv in self.unified_mp_translation.items(): 
            for category in uv:
                cfunct = self.custom_funct(category)
                if not cfunct:
                    self.params[category] = self.bound(
                        self.min(category), 
                        self.max(category), 
                        self.mul(category) * result_dict[mpk] + self.offset(category)
                    )
                else:
                    self.params[category] = self.call_custom_funct(cfunct, mpk)
        return self.serialize()

    def serialize(self):
        return json.dumps(self.params)
    
    def update_tuning(self):
        self.tuning = json5.load(open(self.path, "r"))







class MPFace:

    def __init__(self, capture: int, modelpath: str, tuning_path: str, fps=60):
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
        self.params = BlendShapeParams(tuning_path)

    def process(self, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        try:
            if result.face_blendshapes:
                data = self.params.update(result)
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
MPface = MPFace(capture=cap, modelpath=model_path, tuning_path="./param_tuning.jsonc", fps=90)

try:
    MPface.run()
except KeyboardInterrupt:
    pass