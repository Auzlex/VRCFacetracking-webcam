

# prerequisites
1. a python installation
2. the dependencies in the requirements file. Install with `pip install -r .\requirements.txt`
3. [this VRCFT module](https://github.com/TinyAtoms/VRCFaceTracking-MPmodule)

# running the program
1. start VRCFT 
2. run `python face_webcam.py`
3. tweak parameters as needed.
4. you might need to adjust `cap=0` in the script if you get an error like `ValueError: Please provide 'image_format' with 'data'.` set it to 1, 2, etc.

# Issues
1. Closing VRFTC closes this module
2. closing and reopening this module multiple times while VRFTC is running will eventually make tracking stop working.

# Tuning parameters
1. Stand in front of a VRC mirror while the module is running. Most values sent to VRCFT are `offset + multiplier * raw_value`[^1]
2. it's helpful to have [this page](https://docs.vrcft.io/docs/tutorial-avatars/tutorial-avatars-extras/unified-blendshapes) open to see what each parameter does
3. Make some expressions, see if the avatar does what you do in similar intensity. For example, if you fully open your jaw, is your avatar's jaw fully extended? If you close it, is the same true for your avatar? if not, play with the offset and multiplier to get something in line.
4. Hit save so you don't have to do this next time you use it.




[^1]: not exactly. They're an exponential moving average based on the past couple frames with minor movements cut off for some eye stuff, but that is not relevant since those are not easily tweakable (yet?)