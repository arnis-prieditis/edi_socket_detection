# Socket Detection

This technology detects the center position of sockets (see Figure below) on a moving conveyor belt and classifies each socket as either filled or empty.  
The GitHub repo for this tool is available [here](https://github.com/arnis-prieditis/edi_socket_detection).

## Tool description

The Socket Detection model is based on the YOLOv8m neural network. The training dataset consisted of 252 images, extracted from video footage recorded using an Intel RealSense D435 depth camera, positioned approximately 80 cm above the conveyor belt at a 10–15° tilt. Images were resized to 320×320 pixels, and manual labelling was performed to distinguish between empty sockets and those containing bottles — a necessary step due to the high visual similarity between the white conveyor belt and the white bottles.  
The tool outputs the socket center as a 3D point in the camera's frame of reference.

## Intended uses

-  The detected socket position can be used by a robotic system to determine whether a bottle should be inserted into the socket and to support synchronization between perception, conveyor motion, and robot actions.

## Limitations

- The system is constrained by camera hardware. For acceptable detection quality, the camera must support a minimum resolution of 1280×720 pixels (720p) and 30 FPS. Higher resolutions like 2560×1440 were tested but did not consistently yield better results and increased processing requirements. The model was only evaluated using Intel RealSense D435 and Logitech Brio 4K cameras, and its performance with other hardware remains untested.
- Detection performance was found to improve slightly with a 10° tilt, highlighting the importance of precise and consistent camera placement. 
- The model has a ±2–4 mm socket centre prediction error. Although small, it may be insufficient for industrial contexts that require sub-millimetre precision. 
- A powerful edge computing platform, such as NVIDIA Jetson Nano or higher, is required for real-time inference. Standard CPUs may not deliver the necessary performance in production conditions.
