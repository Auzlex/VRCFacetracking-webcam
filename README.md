

# prerequisites
1. a python installation
2. the dependencies in the requirements file. Install with `pip install -r .\requirements.txt`
3. [this VRCFT module](https://github.com/TinyAtoms/VRCFaceTracking-MPmodule)

# running the program
1. start VRCFT 
2. run `python face_webcam.py`
3. tune parameters as needed

# Issues
Important to note that VRCFT should be started before face_webcam, otherwise the python script will crash.
Also, the save button does nothing besides outputting the parameters in your shell. 
I tweak them and then copy over the shell output to the `param_tuning.jsonc` file

