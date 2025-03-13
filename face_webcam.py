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
from PIL import Image, ImageTk

import mediapipe as mp
import cv2


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class TCPSender:
    def __init__(self, host: str, port: int) -> None:
        """
        host: host to connect to
        port: port to connect to
        """
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1  # Start with 1 second delay
        self.reconnect_thread = None
        self.should_reconnect = True
        self.connection_lock = threading.Lock()  # Add lock for thread safety
        self.last_connection_time = 0  # Track last successful connection

    def connect(self) -> None:
        """
        Connect to the VRCFT server
        it will try to connect for ~15 mins, after which it will exit the program
        """
        with self.connection_lock:
            self.reconnect_attempts = 0
            self.reconnect_delay = 1
            
            while self.reconnect_attempts < self.max_reconnect_attempts and self.should_reconnect:
                try:
                    if self.sock:
                        self.sock.close()
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.connect((self.host, self.port))
                    self.connected = True
                    self.last_connection_time = time.time()
                    print(f"Connected to VRCFT")
                    return
                except ConnectionRefusedError as e:
                    print(f"Error connecting to VRCFT, waiting {self.reconnect_delay} seconds to try again...")
                    time.sleep(self.reconnect_delay)
                    self.reconnect_attempts += 1
                    self.reconnect_delay *= 2  # Exponential backoff
                except Exception as e:
                    print(f"Unexpected error connecting to VRCFT: {str(e)}")
                    time.sleep(self.reconnect_delay)
                    self.reconnect_attempts += 1
                    self.reconnect_delay *= 2

            if self.should_reconnect:
                print("Failed to connect to VRCFT after multiple attempts. Will keep trying...")
                self.start_reconnect_thread()

    def start_reconnect_thread(self) -> None:
        """
        Start a background thread to continuously attempt reconnection
        """
        if self.reconnect_thread is None or not self.reconnect_thread.is_alive():
            self.reconnect_thread = threading.Thread(target=self.reconnect_loop, daemon=True)
            self.reconnect_thread.start()

    def reconnect_loop(self) -> None:
        """
        Continuously attempt to reconnect to VRCFT
        """
        while self.should_reconnect:
            with self.connection_lock:
                try:
                    if self.sock:
                        self.sock.close()
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.connect((self.host, self.port))
                    self.connected = True
                    self.last_connection_time = time.time()
                    print(f"Reconnected to VRCFT")
                except Exception as e:
                    print(f"Reconnection attempt failed: {str(e)}")
                    self.connected = False
                    time.sleep(self.reconnect_delay)
                    self.reconnect_delay *= 2  # Exponential backoff
                    if self.reconnect_delay > 30:  # Cap the delay at 30 seconds
                        self.reconnect_delay = 30

    def send_message(self, message_dict) -> None:
        """
        Send a message to the VRCFT server
        message_dict: dictionary to send
        serializes a dictionary to json and sends it away
        On failure, it will attempt to reconnect
        """
        with self.connection_lock:
            if not self.connected:
                print("Not connected to VRCFT, attempting to reconnect...")
                self.start_reconnect_thread()
                return  # Skip sending this message, but don't exit

            message_json = json.dumps(message_dict)
            message_bytes = (message_json + "\n").encode("utf-8")
            try:
                self.sock.sendall(message_bytes)
            except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
                print(f"Connection to VRCFT lost: {str(e)}")
                self.connected = False
                self.start_reconnect_thread()  # Start reconnection attempts

    def disconnect(self) -> None:
        """
        Disconnect from the VRCFT server
        """
        with self.connection_lock:
            self.should_reconnect = False  # Stop reconnection attempts
            if self.sock:
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                    self.sock.close()
                except:
                    pass
            self.sock = None
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
        
        # Handle zero or near-zero offset values
        if abs(b) < 0.0001:  # If offset is too close to zero
            # Use linear scaling without minimum value to allow complete closure
            scaled_value = a * value
            return self.bound(
                self.min("EyeOpennessLeft"),  # Allow complete closure
                self.max("EyeOpennessLeft"),
                scaled_value
            )

        # Use power function for non-zero offsets
        scaled_value = a * math.pow(value, 1 / b)
        return self.bound(
            self.min("EyeOpennessLeft"),  # Allow complete closure
            self.max("EyeOpennessLeft"),
            scaled_value
        )

    def eye_openess_right(self, value: float) -> float:
        """
        Custom function for EyeOpennessRight processing.
        instead of offset + mul* value, it uses mul* pow(value, 1/offset)
        which is a vaguely sigmoidal function
        """
        a = self.mul("EyeOpennessRight")
        b = self.offset("EyeOpennessRight")
        
        # Handle zero or near-zero offset values
        if abs(b) < 0.0001:  # If offset is too close to zero
            # Use linear scaling without minimum value to allow complete closure
            scaled_value = a * value
            return self.bound(
                self.min("EyeOpennessRight"),  # Allow complete closure
                self.max("EyeOpennessRight"),
                scaled_value
            )

        # Use power function for non-zero offsets
        scaled_value = a * math.pow(value, 1 / b)
        return self.bound(
            self.min("EyeOpennessRight"),  # Allow complete closure
            self.max("EyeOpennessRight"),
            scaled_value
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

# print command to diff file with last commit
# git diff --name-only HEAD^

class MPFace:

    def __init__(self, capture: int, modelpath: str, tuning_path: str, fps: int = 60) -> None:
        self.capture_device_id = capture
        self.exit = False
        self.thread_error = False
        
        # Create separate camera captures for each thread with specific backend
        self.display_capture = cv2.VideoCapture(capture, cv2.CAP_DSHOW)  # Use DirectShow backend
        self.tracking_capture = cv2.VideoCapture(capture, cv2.CAP_DSHOW)
        
        # Set camera properties
        for cap in [self.display_capture, self.tracking_capture]:
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set a reasonable resolution
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffering
            
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
        
        # Add error handling for window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create interface elements
        self.create_interface()

        # Frame to hold the camera feed and the entry + button
        frame = tk.Frame(self.root)
        frame.pack()

        # Status label for face detection
        self.status_label = tk.Label(frame, text="Status: No Face Detected", fg="red")
        self.status_label.pack(side=tk.TOP, pady=5)

        # Label for camera device ID
        text_label = tk.Label(frame, text="Camera device ID")
        text_label.pack(side=tk.LEFT, pady=10)

        # Entry box for changing the capture device ID
        self.device_entry = tk.Entry(frame, width=10)
        self.device_entry.insert(0, str(self.capture_device_id))
        self.device_entry.pack(side=tk.LEFT, padx=5)

        # Change button to call 
        change_button = tk.Button(frame, text="Change", command=self.change_cam_feed)
        change_button.pack(side=tk.LEFT, padx=5)

        # Create video feed label
        self.video_label = tk.Label(frame)
        self.video_label.pack(side=tk.RIGHT)

        # Create toggle button for tuning parameters
        self.toggle_button = tk.Button(self.root, text="Show Tuning Options", command=self.toggle_tuning)
        self.toggle_button.pack(pady=5)
        
        # Initialize TCP sender
        self.sender = TCPSender("localhost", 5555)
        self.sender.connect()
        
        # Start threads with error handling
        try:
            self.tracking_thread = threading.Thread(target=self.run, daemon=True)
            self.display_thread = threading.Thread(target=self.gen_tk_feed, daemon=True)
            self.tracking_thread.start()
            self.display_thread.start()
            
            # Add thread monitoring
            self.root.after(1000, self.check_threads)
            
            # Start the main loop
            self.root.mainloop()
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            self.on_close()
            return

    def check_threads(self) -> None:
        """Monitor thread health and handle errors"""
        if not self.exit:
            if not self.tracking_thread.is_alive() or not self.display_thread.is_alive():
                print("One or more threads have stopped unexpectedly")
                self.thread_error = True
                self.on_close()
            else:
                # Schedule next check
                self.root.after(1000, self.check_threads)

    def change_cam_feed(self) -> None:
        try:
            new_device_id = int(self.device_entry.get())
            # Try to open the new camera device
            test_capture = cv2.VideoCapture(new_device_id)
            if not test_capture.isOpened():
                messagebox.showerror("Error", f"Could not open camera device {new_device_id}")
                return
            test_capture.release()
            
            # If we get here, the new device is valid
            self.capture_device_id = new_device_id
            
            # Update both captures
            self.display_capture.release()
            self.tracking_capture.release()
            self.display_capture = cv2.VideoCapture(new_device_id, cv2.CAP_DSHOW)
            self.tracking_capture = cv2.VideoCapture(new_device_id, cv2.CAP_DSHOW)
            
            # Update preferences
            with open("./preferences.json", "r") as f:
                prefs = json.load(f)
            with open("./preferences.json", "w") as f:
                prefs["capture_device"] = new_device_id
                json.dump(prefs, f)
                
            messagebox.showinfo("Success", f"Successfully switched to camera device {new_device_id}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the camera device ID")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to change camera device: {str(e)}")

    def process(self, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        if result.face_blendshapes:
            data = self.params.update(result)
            if self.sender.connected:
                #print(f"[{timestamp_ms}] Sending tracking data...")
                self.sender.send_message(data)
                #print(f"[{timestamp_ms}] Tracking data sent successfully")
            else:
                print(f"[{timestamp_ms}] VRCFT not connected, skipping data transmission")
            # Update status label to show face is detected
            self.root.after(0, lambda: self.status_label.config(
                text="Status: Face Detected", 
                fg="green"
            ))
        else:
            # Update status label to show no face detected
            self.root.after(0, lambda: self.status_label.config(
                text="Status: No Face Detected", 
                fg="red"
            ))

    def gen_mp_frame(self, capture: cv2.VideoCapture):
        if not capture.isOpened():
            print("Tracking camera is not opened. Attempting to reconnect...")
            capture.release()
            capture.open(self.capture_device_id, cv2.CAP_DSHOW)
            if not capture.isOpened():
                print("Failed to reconnect to tracking camera")
                return None
                
        # Try to grab frame multiple times if needed
        for attempt in range(3):  # Try up to 3 times
            ret, frame = capture.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                return mp_frame
            print(f"Frame capture attempt {attempt + 1} failed, retrying...")
            time.sleep(0.01)  # Small delay between attempts
            
        print("Could not capture frame from tracking camera after multiple attempts") 
        return None
    
    def gen_tk_feed(self):
        print("Display thread started")
        frame_count = 0
        while not self.exit:
            if not self.display_capture.isOpened():
                print("Display camera is not opened. Attempting to reconnect...")
                self.display_capture.release()
                self.display_capture.open(self.capture_device_id, cv2.CAP_DSHOW)
                if not self.display_capture.isOpened():
                    print("Failed to reconnect to display camera")
                    time.sleep(1)
                    continue
                    
            # Try to grab frame multiple times if needed
            for attempt in range(3):
                ret, frame = self.display_capture.read()
                if ret:
                    cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2_image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.configure(image=imgtk)
                    self.video_label.image = imgtk
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"Display thread: Processed {frame_count} frames")
                    break
                print(f"Display frame capture attempt {attempt + 1} failed, retrying...")
                time.sleep(0.01)
                
            time.sleep(1 / self.fps)

    def run(self) -> None:
        print("Tracking thread started")
        frame_count = 0
        try:
            with self.landmarker as landmarker:
                while not self.exit:
                    if not self.tracking_capture.isOpened():
                        print("Tracking camera is not opened. Attempting to reconnect...")
                        self.tracking_capture.release()
                        self.tracking_capture.open(self.capture_device_id, cv2.CAP_DSHOW)
                        if not self.tracking_capture.isOpened():
                            print("Failed to reconnect to tracking camera")
                            time.sleep(1)
                            continue
                            
                    mp_frame = self.gen_mp_frame(self.tracking_capture)
                    if mp_frame is not None:
                        try:
                            frame_count += 1
                            if frame_count % 30 == 0:  # Log every 30 frames
                                print(f"Tracking thread: Processed {frame_count} frames")
                            landmarker.detect_async(
                                mp_frame, int(time.time() * 1000)
                            )
                        except Exception as e:
                            print(f"Error in detect_async: {str(e)}")
                            time.sleep(1)
                    time.sleep(1 / self.fps)
        except KeyboardInterrupt:
            print("Tracking thread interrupted")
            return None
        except Exception as e:
            print(f"Error in tracking thread: {str(e)}")
            self.on_close()

    def create_interface(self) -> None:
        # Create a frame for the parameters
        self.param_frame = tk.Frame(self.root)
        self.param_frame.pack(pady=10, padx=10)
        
        # Initially hide the tuning parameters frame
        self.param_frame.pack_forget()

        self.entries = {}

        # Create a header for mul and offset
        header_frame = tk.Frame(self.param_frame)
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
            column_frame = tk.Frame(self.param_frame)
            column_frame.pack(side=tk.LEFT, padx=10)
            column_frames.append(column_frame)

        for idx, (key, values) in enumerate(self.params.tuning.items()):
            col_idx = idx // params_per_column % column_count
            frame = tk.Frame(column_frames[col_idx])
            frame.pack(pady=5)

            tk.Label(frame, text=key, width=20, anchor="w").pack(side=tk.LEFT)

            mul_entry = tk.Entry(frame, width=10)
            mul_entry.insert(0, str(values["mul"]))
            mul_entry.pack(side=tk.LEFT)
            mul_entry.bind("<Return>", lambda event, k=key, e=mul_entry: self.update_mul_from_entry(k, e.get()))
            mul_entry.bind("<KP_Enter>", lambda event, k=key, e=mul_entry: self.update_mul_from_entry(k, e.get()))
            mul_entry.bind("<FocusOut>", lambda event, k=key, e=mul_entry: self.update_mul_from_entry(k, e.get()))

            offset_entry = tk.Entry(frame, width=10)
            offset_entry.insert(0, str(values["offset"]))
            offset_entry.pack(side=tk.LEFT)
            offset_entry.bind("<Return>", lambda event, k=key, e=offset_entry: self.update_offset_from_entry(k, e.get()))
            offset_entry.bind("<KP_Enter>", lambda event, k=key, e=offset_entry: self.update_offset_from_entry(k, e.get()))
            offset_entry.bind("<FocusOut>", lambda event, k=key, e=offset_entry: self.update_offset_from_entry(k, e.get()))
            self.entries[key] = {"mul_entry": mul_entry, "offset_entry": offset_entry}

        save_button = tk.Button(
            self.param_frame, text="Save Parameters", command=self.run_save
        )
        save_button.pack(pady=5)

    def toggle_tuning(self) -> None:
        if self.param_frame.winfo_ismapped():
            self.param_frame.pack_forget()
        else:
            self.param_frame.pack(pady=10, padx=10)


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
        """Handle application shutdown"""
        print("Closing application...")
        self.exit = True
        
        # Stop threads gracefully
        print("Waiting for threads to finish...")
        if hasattr(self, 'tracking_thread') and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=1.0)
        if hasattr(self, 'display_thread') and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
            
        # Release resources
        print("Releasing camera captures...")
        if hasattr(self, 'display_capture'):
            self.display_capture.release()
        if hasattr(self, 'tracking_capture'):
            self.tracking_capture.release()
            
        print("Disconnecting from VRCFT...")
        if hasattr(self, 'sender'):
            self.sender.disconnect()
            
        print("Destroying window...")
        if hasattr(self, 'root'):
            self.root.destroy()


# change this line below.


try:
    prefs = json.load(open("./preferences.json", "r"))
except FileNotFoundError:
    prefs = {
    "capture_device" : 0,
    "tuning_parameters" : "./param_tuning.jsonc",
    "fps" : 100
}

model_path = "./face/face_landmarker.task"
MPface = MPFace(
    capture=prefs["capture_device"], modelpath=model_path, tuning_path=prefs["tuning_parameters"], fps=prefs["fps"]
)

