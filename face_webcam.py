import socket
import threading
import tkinter as tk
from tkinter import messagebox
import math
import statistics
import copy
import io
import time
from pprint import pprint
import json

import mediapipe as mp
import cv2


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class TCPSender:
    def __init__(self, host: str, port: int, callback) -> None:
        """
        host: host to connect to
        port: port to connect to
        callback: function to call when it can't send messages anymore
        """
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self.callback = callback

    def connect(self) -> None:
        """
        Connect to the VRCFT server
        it will try to connect for ~15 mins, after which it will exit the program
        """

        for i in range(1, 20):
            if not self.connected:
                try:
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.connect((self.host, self.port))
                    self.connected = True
                except ConnectionRefusedError as e:
                    print(
                        f"Error connecting to VRCFT, waiting {5*i} seconds to try again..."
                    )
                    time.sleep(i)
        if self.connected:
            print(f"Connected to VRCFT")
        else:
            print("Failed to connect to VRCFT. Exiting...")
            self.callback()

    def send_message(self, message_dict) -> None:
        """
        Send a message to the VRCFT server
        message_dict: dictionary to send
        serializes a dictionary to json and sends it away
        On failure, it exits the program
        """
        message_json = json.dumps(message_dict)
        message_bytes = (message_json + "\n").encode("utf-8")
        try:
            self.sock.sendall(message_bytes)
        except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
            print("Connection to VRCFT lost. Likely that VRFT was closed.")
            # self.disconnect()
            # time.sleep(5)
            # print("Reconnecting...")
            # self.connect()
            # self.send_message(message_dict)
            print("I have no idea how to handle this yet, exiting...")
            self.callback()

    def disconnect(self) -> None:
        """
        Disconnect from the VRCFT server
        """
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
        self.connected = False


class BlendShapeParams:
    # translation from mediapipe to unified keys
    unified_mp_translation: dict[str, list[str]] = {
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
        "cheekPuff": ["CheekPuff"],  # inacurate detection
        "cheekSquintLeft": ["CheekSquintLeft"],  # inacurate detection
        "cheekSquintRight": ["CheekSquintRight"],  # inacurate detection
        "eyeLookInLeft": ["EyeLookInRight"],  # intentional
        "eyeLookInRight": ["EyeLookInLeft"],  #
        "eyeLookOutLeft": ["EyeLookOutRight"],  #
        "eyeLookOutRight": ["EyeLookOutLeft"],  # intentional
        "eyeLookUpLeft": ["EyeLookUpLeft"],
        "eyeLookUpRight": ["EyeLookUpRight"],
        "eyeLookDownLeft": ["EyeLookDownLeft"],
        "eyeLookDownRight": ["EyeLookDownRight"],
        "eyeSquintLeft": ["EyeSquintLeft"],
        "eyeSquintRight": ["EyeSquintRight"],
        "eyeWideRight": ["EyeWideLeft", "EyeOpennessLeft"],  # intentional
        "eyeWideLeft": ["EyeWideRight", "EyeOpennessRight"],  # intentional
        "jawForward": ["JawForward"],  # inacurate detection
        "jawLeft": ["JawRight"],  # inacurate detection
        "jawOpen": ["JawOpen"],
        "jawRight": ["JawLeft"],  # inacurate detection
        "mouthClose": ["MouthClosed"],
        "mouthDimpleLeft": ["MouthDimpleLeft"],
        "mouthDimpleRight": ["MouthDimpleRight"],
        "mouthFrownLeft": ["MouthFrownLeft"],
        "mouthFrownRight": ["MouthFrownRight"],
        "mouthFunnel": ["LipFunnel"],
        "mouthLeft": ["MouthLeft"],
        "mouthRight": ["MouthRight"],
        "mouthPucker": ["LipPucker"],
        "mouthSmileLeft": ["MouthSmileLeft"],
        "mouthSmileRight": ["MouthSmileRight"],
        "mouthStretchLeft": ["MouthStretchLeft"],
        "mouthStretchRight": ["MouthStretchRight"],
        "noseSneerLeft": ["NoseSneerLeft"],  # inmacurate detection
        "noseSneerRight": ["NoseSneerRight"],  # inmacurate detection
        "mouthUpperUpLeft": ["MouthUpperUpLeft"],
        "mouthUpperUpRight": ["MouthUpperUpRight"],
        "mouthLowerDownLeft": ["MouthLowerDownLeft"],
        "mouthLowerDownRight": ["MouthLowerDownRight"],
        "mouthPressLeft": ["MouthPressLeft"],
        "mouthPressRight": ["MouthPressRight"],
    }

    #  "raw" parameter storage (used in raw processing and EMA)
    params = {}
    # EMA'd parameters from last frame
    last_frame = {}
    stats = {
        "EyeLookInRight": [],
        "EyeLookInLeft": [],
        "EyeLookOutRight": [],
        "EyeLookOutLeft": [],
        "EyeWideLeft": [],
        "EyeWideRight": [],
        "EyeOpennessLeft": [],
        "EyeOpennessRight": [],
    }

    tresholds = {
        "EyeLookInRight": 0.0119,
        "EyeLookInLeft": 0.0140,
        "EyeLookOutRight": 0.0128,
        "EyeLookOutLeft": 0.0112,
        "EyeWideLeft": 0.0071,
        "EyeWideRight": 0.0050,
        "EyeOpennessLeft": 0.0237,
        "EyeOpennessRight": 0.0225,
    }

    def eye_openness_left(self, value: float) -> float:
        """
        Custom function for EyeOpennessLeft processing.
        instead of offset + mul* value, it uses mul* pow(value, 1/offset)
        which is a vaguely sigmoidal function
        """
        a = self.mul("EyeOpennessLeft")
        b = self.offset("EyeOpennessLeft")

        return self.bound(
            self.min("EyeOpennessLeft"),
            self.max("EyeOpennessLeft"),
            a * math.pow(value, 1 / b),
        )

    def eye_openess_right(self, value: float) -> float:
        """
        Custom function for EyeOpennessLeft processing.
        instead of offset + mul* value, it uses mul* pow(value, 1/offset)
        which is a vaguely sigmoidal function
        """
        a = self.mul("EyeOpennessRight")
        b = self.offset("EyeOpennessRight")
        return self.bound(
            self.min("EyeOpennessRight"),
            self.max("EyeOpennessRight"),
            a * math.pow(value, 1 / b),
        )

    def __init__(self, tuning_params_path: str) -> None:
        self.path = tuning_params_path
        self.tuning = json.load(open(tuning_params_path, "r"))

    def bound(self, low: float, high: float, value: float) -> float:
        return max(low, min(high, value))

    def min(self, category: str) -> float:
        return self.tuning[category]["min"]

    def max(self, category: str) -> float:
        return self.tuning[category]["max"]

    def mul(self, category: str) -> float:
        return self.tuning[category]["mul"]

    def offset(self, category: str) -> float:
        return self.tuning[category]["offset"]

    def custom_funct(self, category: str) -> str:
        """
        return custom function name of a particular category
        """
        return self.tuning[category]["cfunct"]

    def call_custom_funct(self, cfunct: str, value: float) -> float:
        """
        call custom function of a particular category"""
        f = getattr(self, cfunct, None)

        if callable(f):
            return f(value)
        else:
            return value

    def process_raw_params(self, result: FaceLandmarkerResult) -> None:
        """
        process FaceLandmarkerResult based on tuning parameters
        """
        # process LFMR for easy access
        result_dict = {}
        for param in result.face_blendshapes[0]:
            result_dict[param.category_name] = param.score

        # some mp keys need to be mapped to multiple unified keys
        # might do the reverse to remove a single loop for readability
        # check if a custom function is needed for processing
        # if not, use provided multipliers and offsets

        # mpk=mediapipe key, uv=unified value
        for mpk, uv in self.unified_mp_translation.items():
            for category in uv:
                cfunct = self.custom_funct(category)
                if not cfunct:
                    self.params[category] = self.bound(
                        self.min(category),
                        self.max(category),
                        self.mul(category) * result_dict[mpk] + self.offset(category),
                    )
                else:
                    self.params[category] = self.call_custom_funct(
                        cfunct, result_dict[mpk]
                    )

    def EMA(self, M: int = 4) -> dict[str, float]:
        """
        Exponential Moving Average
        processes tuned parameters to smooth them over
        M : int is the smoothing period
        """

        alpha = 2 / (M + 1)

        if not self.last_frame:
            self.last_frame = self.params
            return self.last_frame
        else:
            current_ema = self.last_frame

        new_ema = {}
        # self.param = new values
        for k, v in self.params.items():
            new_ema[k] = alpha * v + (1 - alpha) * current_ema[k]

        self.last_frame = new_ema
        return self.last_frame

    def treshold(self, old: dict[str, float], mul: int = 2):
        """
        if the change in param is less than the (mul * treshold), set it to the old value to reduce noise
        treshold is the experimentally determined standard deviation of the change in params between frames
        old: dict[str, float] is the previous frame
        mul: int is the the multiplier
        """

        for k, v in self.tresholds.items():
            if (
                old.get(k)
                and self.last_frame.get(k)
                and abs(self.last_frame[k] - old[k]) < v * mul
            ):
                self.last_frame[k] = old[k]

    def collect_stats(self, old: dict[str, float], new: dict[str, float]) -> None:
        if not old:
            return
        for k in self.stats.keys():
            try:
                self.stats[k].append(new[k] - old[k])
            except KeyError:
                pass

    def print_stats(self) -> None:
        for k, v in self.stats.items():
            mean = statistics.fmean(v)
            std = statistics.stdev(v)
            print(f"{k}: {mean:.4f} +/- {std:.4f}")

    def update(self, result: FaceLandmarkerResult) -> str:
        """
        processes FaceLandmarkerResult and updates parameters
        does raw processing, EMA and tresholding
        """
        old = copy.deepcopy(self.last_frame)
        self.process_raw_params(result)
        self.EMA()
        self.treshold(old)
        self.collect_stats(old, self.last_frame)
        return self.serialize()

    def serialize(self) -> str:
        return json.dumps(self.last_frame)

    def update_tuning(self) -> None:
        self.tuning = json.load(open(self.path, "r"))


class MPFace:

    def __init__(self, capture: int, modelpath: str, tuning_path: str, fps: int = 60) -> None:
        self.capture = cv2.VideoCapture(capture)  # opencv webcam feed
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

        self.root = tk.Tk()
        self.root.title("Blend Shape Parameters")
        self.exit = False

        self.create_interface()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.sender = TCPSender("localhost", 5555, self.on_close)
        self.sender.connect()
        threading.Thread(target=self.run, daemon=True).start()

        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_close()
            return

    def process(self, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        if result.face_blendshapes:
            data = self.params.update(result)
            self.sender.send_message(data)

    def gen_mp_frame(self, capture: cv2.VideoCapture):

        ret, frame = capture.read()
        if not ret:
            return
        # cv2.flip(frame, 1)
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

    def run(self) -> None:
        try:
            with self.landmarker as landmarker:
                while not self.exit:
                    landmarker.detect_async(
                        self.gen_mp_frame(self.capture), int(time.time() * 1000)
                    )
                    time.sleep(1 / self.fps)
        except KeyboardInterrupt:
            return None

    def create_interface(self) -> None:
        # Create a frame for the parameters
        param_frame = tk.Frame(self.root)
        param_frame.pack(pady=10, padx=10)

        self.entries = {}

        # Create a header for mul and offset
        header_frame = tk.Frame(param_frame)
        header_frame.pack()

        tk.Label(header_frame, text="Parameter").grid(row=0, column=0)
        tk.Label(header_frame, text="Multiplier (mul)").grid(row=0, column=1)
        tk.Label(header_frame, text="Offset").grid(row=0, column=2)

        # Organize parameters in columns
        column_count = 3  # Number of columns
        params_per_column = 10  # How many parameters per column

        # Create column frames
        column_frames = []
        for col in range(column_count):
            column_frame = tk.Frame(param_frame)
            column_frame.pack(side=tk.LEFT, padx=10)
            column_frames.append(column_frame)

        for idx, (key, values) in enumerate(self.params.tuning.items()):
            col_idx = (
                idx // params_per_column % column_count
            )  # Determine which column to put the parameter in
            frame = tk.Frame(column_frames[col_idx])
            frame.pack(pady=5)

            tk.Label(frame, text=key, width=20, anchor="w").pack(side=tk.LEFT)

            mul_entry = tk.Entry(frame, width=10)
            mul_entry.insert(0, str(values["mul"]))
            mul_entry.pack(side=tk.LEFT)
            mul_entry.bind(
                "<Return>",
                lambda event, k=key: self.update_mul_from_entry(k, mul_entry.get()),
            )
            mul_entry.bind(
                "<FocusOut>",
                lambda event, k=key, e=mul_entry: self.update_mul_from_entry(
                    k, e.get()
                ),
            )

            offset_entry = tk.Entry(frame, width=10)
            offset_entry.insert(0, str(values["offset"]))
            offset_entry.pack(side=tk.LEFT)
            offset_entry.bind(
                "<Return>",
                lambda event, k=key: self.update_offset_from_entry(
                    k, offset_entry.get()
                ),
            )
            offset_entry.bind(
                "<FocusOut>",
                lambda event, k=key, e=offset_entry: self.update_offset_from_entry(
                    k, e.get()
                ),
            )

            self.entries[key] = {"mul_entry": mul_entry, "offset_entry": offset_entry}

        save_button = tk.Button(
            self.root, text="Save[does not actually save them]", command=self.run_save
        )
        save_button.pack(pady=10)

    def update_mul(self, key: str, value: float) -> None:
        self.params.tuning[key]["mul"] = float(value)
        self.entries[key]["mul_entry"].delete(0, tk.END)
        self.entries[key]["mul_entry"].insert(0, str(value))

    def update_offset(self, key: str, value: float) -> None:
        self.params.tuning[key]["offset"] = float(value)
        self.entries[key]["offset_entry"].delete(0, tk.END)
        self.entries[key]["offset_entry"].insert(0, str(value))

    def update_mul_from_entry(self, key: str, value: str) -> None:
        try:
            float_value = float(value)
            self.update_mul(key, float_value)
        except ValueError:
            messagebox.showwarning(
                "Invalid Input", "Please enter a valid float value for 'mul'."
            )

    def update_offset_from_entry(self, key: str, value: str) -> None:
        try:
            float_value = float(value)
            self.update_offset(key, float_value)
        except ValueError:
            messagebox.showwarning(
                "Invalid Input", "Please enter a valid float value for 'offset'."
            )

    def run_save(self) -> None:
        output = io.StringIO()
        pprint(self.params.tuning, stream=output, width=120)
        pretty_output = output.getvalue().replace("'", '"')
        with open("./param_tuning.jsonc", "w") as f:
            f.write(pretty_output)
        self.show_save_message()

    def show_save_message(self) -> None:
        # Schedule a message box to show after saving is done
        self.root.after(
            0, lambda: messagebox.showinfo("Save", "Parameters saved successfully!")
        )

    def on_close(self) -> None:
        self.params.print_stats()
        print("Exiting...")
        self.exit = True
        self.sender.disconnect()
        self.root.destroy()


# change this line below.
cap = 0
model_path = "./face/face_landmarker.task"
MPface = MPFace(
    capture=cap, modelpath=model_path, tuning_path="./param_tuning.jsonc", fps=100
)
