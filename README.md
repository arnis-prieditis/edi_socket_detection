# EDI Socket Detection Tool for AIMS 5.0 AI Toolbox

This tool detects `bed`, `socket` and `bottle` classes in input RGB images using a YOLO model and
returns an annotated image. It can also publish detected center points to a ROS topic.

## Assets
- `assets/sockets_and_bottles.pt` - YOLO weights used for detection
- `assets/bed_augmented.pt` - alternative/auxiliary model weights (if required)

## Usage

## Notes
- Place any custom model weights into `assets/` and adjust the notebook or `--model` argument when running the script.

For more information on the AI Toolbox concept and tooling, see https://github.com/aims50toolbox/aitoolbox
