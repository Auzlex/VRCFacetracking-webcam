

# prerequisites
1. a python installation
2. the dependencies in the requirements file. Install with `pip install -r .\requirements.txt`
3. [this VRCFT module](https://github.com/TinyAtoms/VRCFaceTracking-MPmodule)

# running the program
1. start VRCFT 
2. run `python face_webcam.py`
3. tune parameters as needed

# Note
Important to note that VRCFT should be started before face_webcam, otherwise the python script will crash. The python program should also be stopped before VRCFT is closed for the same reason.
Also, the save button does nothing besides outputting the parameters in your shell. I tweak them and then copy over the shell output to the `param_tuning.jsonc` file.

# Issues/TODO
1. program crashes if it can't send data to the VRCFT module
2. save button does not work 
3. still feels kind of jittery
