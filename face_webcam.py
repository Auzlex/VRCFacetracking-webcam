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
from tkinter import ttk


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
        # Create loading window
        self.loading_root = tk.Tk()
        self.loading_root.title("Loading")
        self.loading_root.geometry("300x100")
        
        # Add loading label
        loading_label = tk.Label(self.loading_root, text="Scanning for cameras...")
        loading_label.pack(pady=20)
        
        # Add progress label
        self.progress_label = tk.Label(self.loading_root, text="")
        self.progress_label.pack(pady=10)
        
        # Update the window
        self.loading_root.update()
        
        # Get available camera devices
        self.available_cameras = self.get_available_cameras()
        
        # Destroy loading window
        self.loading_root.destroy()
        
        # Initialize main window
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

        # Frame to hold the camera feed and the camera selection
        frame = tk.Frame(self.root)
        frame.pack()

        # Status label for face detection
        self.status_label = tk.Label(frame, text="Status: No Face Detected", fg="red")
        self.status_label.pack(side=tk.TOP, pady=5)

        # Camera selection frame
        camera_frame = tk.Frame(frame)
        camera_frame.pack(side=tk.TOP, pady=5)

        # Label for camera selection
        tk.Label(camera_frame, text="Select Camera:").pack(side=tk.LEFT, padx=5)

        # Create dropdown for camera selection
        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(camera_frame, 
                                          textvariable=self.camera_var,
                                          values=list(self.available_cameras.values()),
                                          state="readonly",
                                          width=40)
        self.camera_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Set initial selection
        initial_camera_name = self.available_cameras.get(capture, "Unknown Camera")
        self.camera_dropdown.set(initial_camera_name)
        
        # Bind selection change event
        self.camera_dropdown.bind('<<ComboboxSelected>>', self.on_camera_select)

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

    def get_available_cameras(self) -> dict[int, str]:
        """Get a dictionary of available camera devices with their names"""
        cameras = {}
        
        try:
            import win32com.client
            wmi = win32com.client.GetObject("winmgmts:")
            print("Scanning for camera devices...")
            self.progress_label.config(text="Scanning for camera devices...")
            self.loading_root.update()
            
            # First, get all camera devices from WMI
            camera_devices = []
            for device in wmi.InstancesOf("Win32_PnPEntity"):
                if hasattr(device, 'Name') and device.Name is not None:
                    name = device.Name.lower()
                    if "camera" in name or "webcam" in name or "video" in name:
                        camera_devices.append(device)
                        print(f"Found camera device: {device.Name}")
            
            # Then check more indices for available cameras
            for i in range(20):  # Check more indices
                self.progress_label.config(text=f"Checking camera index {i}...")
                self.loading_root.update()
                
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        # Try to get camera name from WMI
                        found_name = False
                        for device in camera_devices:
                            try:
                                # Try to match by device ID or description
                                if hasattr(device, 'DeviceID'):
                                    device_id = device.DeviceID.lower()
                                    if f"\\{i}" in device_id or f"\\{i}." in device_id:
                                        cameras[i] = device.Name
                                        found_name = True
                                        print(f"Matched camera {i} to device: {device.Name}")
                                        break
                            except:
                                continue
                        
                        # If no WMI match, try to get name from OpenCV
                        if not found_name:
                            try:
                                name = cap.get(cv2.CAP_PROP_DEVICE_DESCRIPTION)
                                if name and name.strip():
                                    cameras[i] = name.strip()
                                    print(f"Found camera {i} with OpenCV description: {name.strip()}")
                                else:
                                    cameras[i] = f"Camera {i}"
                                    print(f"Found camera {i} with default name")
                            except:
                                cameras[i] = f"Camera {i}"
                                print(f"Found camera {i} with default name")
                        
                        cap.release()
                except Exception as e:
                    print(f"Error checking camera {i}: {str(e)}")
                    continue
                
        except ImportError:
            print("win32com not available, falling back to basic camera detection")
            self.progress_label.config(text="Using basic camera detection...")
            self.loading_root.update()
            
            # Fallback to basic detection
            for i in range(20):  # Check more indices
                self.progress_label.config(text=f"Checking camera index {i}...")
                self.loading_root.update()
                
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        try:
                            name = cap.get(cv2.CAP_PROP_DEVICE_DESCRIPTION)
                            if name and name.strip():
                                cameras[i] = name.strip()
                                print(f"Found camera {i} with description: {name.strip()}")
                            else:
                                cameras[i] = f"Camera {i}"
                                print(f"Found camera {i} with default name")
                        except:
                            cameras[i] = f"Camera {i}"
                            print(f"Found camera {i} with default name")
                        cap.release()
                except Exception as e:
                    print(f"Error checking camera {i}: {str(e)}")
                    continue
        
        # If no cameras found, add a default camera
        if not cameras:
            print("No cameras found, adding default camera")
            cameras[0] = "Default Camera"
        
        print(f"Total cameras found: {len(cameras)}")
        self.progress_label.config(text=f"Found {len(cameras)} cameras")
        self.loading_root.update()
        time.sleep(1)  # Show the final count briefly
        
        return cameras

    def on_camera_select(self, event=None) -> None:
        """Handle camera selection change"""
        try:
            # Get selected camera name
            selected_name = self.camera_var.get()
            
            # Find corresponding device ID
            new_device_id = None
            for device_id, name in self.available_cameras.items():
                if name == selected_name:
                    new_device_id = device_id
                    break
            
            if new_device_id is None:
                messagebox.showerror("Error", "Could not find selected camera device")
                return
                
            # Try to open the new camera device
            test_capture = cv2.VideoCapture(new_device_id, cv2.CAP_DSHOW)
            if not test_capture.isOpened():
                messagebox.showerror("Error", f"Could not open camera device {new_device_id}")
                return
            test_capture.release()
            
            # If we get here, the new device is valid
            self.capture_device_id = new_device_id
            
            # Update both captures
            try:
                self.display_capture.release()
                self.tracking_capture.release()
                time.sleep(0.1)  # Small delay before reconnecting
                self.display_capture = cv2.VideoCapture(new_device_id, cv2.CAP_DSHOW)
                self.tracking_capture = cv2.VideoCapture(new_device_id, cv2.CAP_DSHOW)
                
                # Set camera properties for new captures
                for cap in [self.display_capture, self.tracking_capture]:
                    cap.set(cv2.CAP_PROP_FPS, self.fps)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception as e:
                print(f"Error switching cameras: {str(e)}")
                messagebox.showerror("Error", f"Failed to switch cameras: {str(e)}")
                # Try to restore previous camera
                try:
                    self.display_capture = cv2.VideoCapture(self.capture_device_id, cv2.CAP_DSHOW)
                    self.tracking_capture = cv2.VideoCapture(self.capture_device_id, cv2.CAP_DSHOW)
                except:
                    pass
                return
            
            # Update preferences
            with open("./preferences.json", "r") as f:
                prefs = json.load(f)
            with open("./preferences.json", "w") as f:
                prefs["capture_device"] = new_device_id
                json.dump(prefs, f)
                
            messagebox.showinfo("Success", f"Successfully switched to {selected_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to change camera device: {str(e)}")
            # Reset dropdown to previous selection
            self.camera_var.set(self.available_cameras.get(self.capture_device_id, "Unknown Camera"))

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
            try:
                capture.release()
                time.sleep(0.1)  # Small delay before reconnecting
                capture.open(self.capture_device_id, cv2.CAP_DSHOW)
                if not capture.isOpened():
                    print("Failed to reconnect to tracking camera, will retry...")
                    return None
                    
                # Set camera properties after reconnecting
                capture.set(cv2.CAP_PROP_FPS, self.fps)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception as e:
                print(f"Error reconnecting to tracking camera: {str(e)}")
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
            try:
                if not self.display_capture.isOpened():
                    print("Display camera is not opened. Attempting to reconnect...")
                    try:
                        self.display_capture.release()
                        time.sleep(0.1)  # Small delay before reconnecting
                        self.display_capture = cv2.VideoCapture(self.capture_device_id, cv2.CAP_DSHOW)
                        if not self.display_capture.isOpened():
                            print("Failed to reconnect to display camera, will retry...")
                            time.sleep(1)
                            continue
                            
                        # Set camera properties after reconnecting
                        self.display_capture.set(cv2.CAP_PROP_FPS, self.fps)
                        self.display_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.display_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.display_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception as e:
                        print(f"Error reconnecting to display camera: {str(e)}")
                        time.sleep(1)
                        continue
                        
                # Try to grab frame multiple times if needed
                for attempt in range(3):
                    try:
                        ret, frame = self.display_capture.read()
                        if ret and frame is not None:
                            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(cv2_image)
                            imgtk = ImageTk.PhotoImage(image=img)
                            self.video_label.configure(image=imgtk)
                            self.video_label.image = imgtk
                            frame_count += 1
                            if frame_count % 30 == 0:  # Log every 30 frames
                                print(f"Display thread: Processed {frame_count} frames")
                            break
                        else:
                            print(f"Frame capture attempt {attempt + 1} failed, retrying...")
                            time.sleep(0.01)
                    except cv2.error as e:
                        print(f"OpenCV error during frame capture: {str(e)}")
                        self.display_capture.release()
                        time.sleep(0.1)
                        self.display_capture = cv2.VideoCapture(self.capture_device_id, cv2.CAP_DSHOW)
                        time.sleep(0.1)
                        continue
                    except Exception as e:
                        print(f"Unexpected error during frame capture: {str(e)}")
                        time.sleep(0.1)
                        continue
                        
                time.sleep(1 / self.fps)
            except Exception as e:
                print(f"Error in display thread: {str(e)}")
                time.sleep(1)
                try:
                    self.display_capture.release()
                    self.display_capture = cv2.VideoCapture(self.capture_device_id, cv2.CAP_DSHOW)
                except:
                    pass

    def run(self) -> None:
        print("Tracking thread started")
        frame_count = 0
        try:
            with self.landmarker as landmarker:
                while not self.exit:
                    if not self.tracking_capture.isOpened():
                        print("Tracking camera is not opened. Attempting to reconnect...")
                        try:
                            self.tracking_capture.release()
                            time.sleep(0.1)  # Small delay before reconnecting
                            self.tracking_capture.open(self.capture_device_id, cv2.CAP_DSHOW)
                            if not self.tracking_capture.isOpened():
                                print("Failed to reconnect to tracking camera, will retry...")
                                time.sleep(1)
                                continue
                                
                            # Set camera properties after reconnecting
                            self.tracking_capture.set(cv2.CAP_PROP_FPS, self.fps)
                            self.tracking_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.tracking_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.tracking_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception as e:
                            print(f"Error reconnecting to tracking camera: {str(e)}")
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
            # Don't exit on error, just keep trying
            while not self.exit:
                time.sleep(1)
                try:
                    self.tracking_capture.release()
                    self.tracking_capture.open(self.capture_device_id, cv2.CAP_DSHOW)
                except:
                    pass

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

