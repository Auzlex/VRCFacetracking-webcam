
# VRChat Face Tracking Module Documentation

This program utilizes your webcam for face tracking, enabling eye and mouth tracking capabilities. It does not require a phone, allowing for a straightforward and convenient setup for VRChat users.

# Prerequisites
To get started with the VRChat Face Tracking Module, make sure you have the following:

1. **Python Installation**: Ensure that you have Python installed on your computer. If not, install from [here](https://www.python.org/downloads/windows/)
2. **Install Dependencies**: You’ll need to install the required dependencies. Open a command prompt and run:
   ```
   pip install -r .\requirements.txt
   ```
3. **VRC Face Tracking Module**: You can download the VRC Face Tracking module from [this link](https://github.com/TinyAtoms/VRCFaceTracking-MPmodule).


# Running the Program
Follow these steps to run the face tracking program:
1. **Clone this repository**: `git clone https://github.com/TinyAtoms/VRCFacetracking-webcam.git`
2. **Start VRCFT**: Launch the VRC Face Tracking (VRCFT) application.
3. **Run the Script**: Execute the following command in your terminal or command prompt:
   ```
   python face_webcam.py
   ```
4. **Adjust Parameters**: Feel free to tweak the parameters in the script as needed for your setup.
5. **Troubleshooting**: If you encounter an error like `ValueError: Please provide 'image_format' with 'data'.`, you may need to modify the line `cap=0` in the script. Try setting it to `1`, `2`, or another appropriate value.

# Known Issues
1. **Closing VRCFT**: Please note that closing VRCFT will also terminate this script.
2. **Eye Tracking**: Tracking how open your eyes are is kind of bad for people wearing glasses



# Tuning Parameters
Here are some guidelines to help you tune the performance. This may or may not be needed, as the current parameters have been set to my particular setup, which may be fine for most people.

1. **Mirror Feedback**: Stand in front of a VRChat mirror while the module is active. This will help you see the effects of your adjustments in real-time. The values sent to VRCFT are calculated as `offset + multiplier * raw_value`[^1].
2. **Reference Materials**: It’s helpful to have [this reference page](https://docs.vrcft.io/docs/tutorial-avatars/tutorial-avatars-extras/unified-blendshapes) open. It provides detailed explanations of each parameter's function.
3. **Testing Expressions**: Create different facial expressions and observe if your avatar reflects those expressions with similar intensity. For instance, check if fully opening your jaw results in your avatar’s jaw fully extending. If the movements don’t align, adjust the offset and multiplier values accordingly.
4. **Save Your Settings**: Remember to save your changes so that you don’t have to reconfigure everything the next time you use the module.

[^1]: The calculation is based on an exponential moving average of recent frames, with minor movements filtered out for some parameters. However, these details are not currently adjustable, so that's not really relevant.