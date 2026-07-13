# EDI Socket Detection Tool for AIMS 5.0 AI Toolbox

This tool detects `bed`, `socket` and `bottle` classes in input RGB images using a YOLOv8m model and
returns an annotated image. It can also publish detected center points to a ROS topic.

## Assets
- `assets/sockets_and_bottles.pt` - YOLO weights used for detection
- `assets/bed_augmented.pt` - alternative/auxiliary model weights (if required)

## Usage

Create a python virtual environment and install dependencies
```
python3 -m venv ./.venv
source ./.venv/bin/activate
cd /path/to/edi_socket_detection/
pip3 install -r requirements.txt
```

Then run the script with
```
cd src/
python3 detect_bed.py
```

## Notes
- Place any custom model weights into `assets/` and adjust the model path when running the script.
- The script works with a RealSense depth camera connected to the device. An example RealSense recording can be downloaded [here](https://mega.nz/folder/fVMDiBRI#YSjy0ooC4NSLbrYWLpk2YA). To use it, see the comments in detect_bed.py function start_realsense().

For more information on the AI Toolbox concept and tooling, see https://github.com/aims50toolbox/aitoolbox
