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


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class BlendShape:

    # these require no additional compute
    mp_unified_translation = {
        "browDownLeft": ["BrowDownLeft"],
        "browDownRight": ["BrowDownRight"],
        "browInnerUp": ["BrowInnerUpRight", "BrowInnerUpLeft"],
        "browOuterUpLeft": ["BrowOuterUpLeft"],
        "browOuterUpRight": ["BrowOuterUpRight"],
        "cheekPuff": ["CheekPuff"],
        "cheekSquintLeft": ["CheekSquintLeft"],
        "cheekSquintRight": ["CheekSquintRight"],
        # "eyeBlinkLeft": ["aaaaaaaaaaaaaa"], # no equivalent
        # "eyeBlinkRight": ["aaaaaaaaaaaaaa"], # no equivalent
        "eyeLookDownLeft": ["EyeLookDownLeft"],
        "eyeLookDownRight": ["EyeLookDownRight"],
        "eyeLookInLeft": ["EyeLookInLeft"],
        "eyeLookInRight": ["EyeLookInRight"],
        "eyeLookOutLeft": ["EyeLookOutLeft"],
        "eyeLookOutRight": ["EyeLookOutRight"],
        "eyeLookUpLeft": ["EyeLookUpLeft"],
        "eyeLookUpRight": ["EyeLookUpRight"],
        "eyeSquintLeft": ["EyeSquintLeft"],
        "eyeSquintRight": ["EyeSquintRight"],
        "eyeWideLeft": ["EyeWideLeft"],
        "eyeWideRight": ["EyeWideRight"],
        "jawForward": ["JawForward"],
        "jawLeft": ["JawLeft"],
        "jawOpen": ["JawOpen"],
        "jawRight": ["JawRight"],
        "mouthClose": ["MouthClosed"],
        "mouthDimpleLeft": ["MouthDimpleLeft"],
        "mouthDimpleRight": ["MouthDimpleRight"],
        "mouthFrownLeft": ["MouthFrownLeft"],
        "mouthFrownRight": ["MouthFrownRight"],
        "mouthFunnel": ["LipFunnel"],
        "mouthLeft": ["MouthLeft"],
        "mouthLowerDownLeft": ["MouthLowerDownLeft"],
        "mouthLowerDownRight": ["MouthLowerDownRight"],
        "mouthPressLeft": ["MouthPressLeft"],
        "mouthPressRight": ["MouthPressRight"],
        "mouthPucker": ["LipPucker"],
        "mouthRight": ["MouthRight"],
        # "mouthRollLower": ["aaaaaaaaaaaaaa"], # no equivalent
        # "mouthRollUpper": ["aaaaaaaaaaaaaa"], # no equivalent
        # "mouthShrugLower": ["aaaaaaaaaaaaaa"], # no equivalent
        # "mouthShrugUpper": ["aaaaaaaaaaaaaa"], # no equivalent
        "mouthSmileLeft": ["MouthSmileLeft"],
        "mouthSmileRight": ["MouthSmileRight"],
        "mouthStretchLeft": ["MouthStretchLeft"],
        "mouthStretchRight": ["MouthStretchRight"],
        "mouthUpperUpLeft": ["MouthUpperUpLeft"],
        "mouthUpperUpRight": ["MouthUpperUpRight"],
        "noseSneerLeft": ["NoseSneerLeft"],
        "noseSneerRight": ["NoseSneerRight"],
    }

    def __init__(self, init_obj: FaceLandmarkerResult):
        self.categories = {}

        categories = init_obj.face_blendshapes[0]

        for category in categories:
            self.categories[category.category_name] = category.score

    def serialize(self):
        translated = {}
        for key in self.mp_unified_translation.keys():
            categories = self.mp_unified_translation[key]
            for category in categories:
                translated[category] = self.categories[key]
        return json.dumps(translated)


class MPFace:

    def __init__(self, capture: int, modelpath: str, fps=60):
        self.capture = cv2.VideoCapture(capture)
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
        data = BlendShape(result).serialize()
        socket.send_string(data)

    def gen_mp_frame(self, capture: cv2.VideoCapture):
        ret, frame = capture.read()
        if not ret:
            return
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
                    print(time.time())
                except KeyboardInterrupt:
                    break

cap = 0
model_path = "./face/face_landmarker.task"
MPface = MPFace(capture=cap, modelpath=model_path)

MPface.run()



socket.close()
context.term()
context.destroy()