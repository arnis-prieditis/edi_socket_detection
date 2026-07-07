
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

CENTER_TOPIC = "/socket_center"
H_CM = np.nan  # Set to a number, for example 82.0, to use fixed Heght or use np.nan for automatic distance
CAMERA_WIDTH = 848 #1280
CAMERA_HEIGHT = 480 #720
CAMERA_FPS = 10 #30
BOTTLE_CONFIDENCE = 0.95
SOCKET_CONFIDENCE = 0.80
BOTTLE_SMOOTHING_ALPHA = 0.35
BOTTLE_MAX_MISSED_FRAMES = 4
SOCKET_BOTTLE_SUPPRESS_IOA = 0.35


def publish_detected_center(publisher, center, height_cm, col_intr_matrix):
    if center is None or height_cm is None:
        return

    try:
        point_3d = pixel_to_3d_coord(center[0], center[1], col_intr_matrix, height_cm * 0.01) # 3d point in meters
        msg = PoseStamped()
        msg.header.frame_id = "socket_cam"
        msg.pose.position.x = float(point_3d[0])
        msg.pose.position.y = float(point_3d[1])
        msg.pose.position.z = float(point_3d[2])
        publisher.publish(msg)
    except Exception as exc:
        print(f"Warning: failed to publish detected center: {exc}")

def get_height_cm(center, depth_map):
    if np.isfinite(H_CM):
        return H_CM

    if center is None or depth_map is None:
        return None

    x, y = int(center[0]), int(center[1])
    if y < 0 or y >= depth_map.shape[0] or x < 0 or x >= depth_map.shape[1]:
        return None

    y1 = max(0, y - 2)
    y2 = min(depth_map.shape[0], y + 3)
    x1 = max(0, x - 2)
    x2 = min(depth_map.shape[1], x + 3)
    depth_patch = depth_map[y1:y2, x1:x2]
    valid_depths = depth_patch[np.isfinite(depth_patch) & (depth_patch > 0)]

    if valid_depths.size == 0:
        return None

    return float(np.median(valid_depths))


def draw_detection_info(frame, x, y, lines, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5 if frame.shape[1] <= 700 else 0.55
    thickness = 2
    line_height = int(22 * scale / 0.5)
    max_width = frame.shape[1] - x - 6

    for line in lines:
        text_width = cv2.getTextSize(line, font, scale, thickness)[0][0]
        while text_width > max_width and scale > 0.38:
            scale -= 0.03
            line_height = int(22 * scale / 0.5)
            text_width = cv2.getTextSize(line, font, scale, thickness)[0][0]

    total_height = line_height * len(lines)
    y = min(y, frame.shape[0] - 6)
    if y + total_height > frame.shape[0] - 6:
        y = max(16, y - total_height - 8)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * line_height),
                    font, scale, color, thickness)


def get_detection_color(normalized_name):
    if normalized_name == "bed":
        return (0, 255, 0)
    if normalized_name == "socket":
        return (0, 0, 255)
    if normalized_name == "bottle":
        return (255, 255, 0)
    return (255, 255, 0)


def box_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


def intersection_over_area(inner_bbox, outer_bbox):
    ix_min = max(inner_bbox[0], outer_bbox[0])
    iy_min = max(inner_bbox[1], outer_bbox[1])
    ix_max = min(inner_bbox[2], outer_bbox[2])
    iy_max = min(inner_bbox[3], outer_bbox[3])
    intersection = box_area((ix_min, iy_min, ix_max, iy_max))
    area = box_area(inner_bbox)
    if area <= 0:
        return 0.0
    return intersection / area


def draw_yolo_detection(frame, bbox, name, confidence, depth_map, color):
    x_min, y_min, x_max, y_max = [int(round(value)) for value in bbox]
    label = f"{name}: {confidence:.2f}"

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(frame, label, (x_min, max(0, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    cv2.circle(frame, (cx, cy), 6, color, -1)
    cross_len = 18
    cv2.line(frame, (cx - cross_len, cy), (cx + cross_len, cy), color, 2)
    cv2.line(frame, (cx, cy - cross_len), (cx, cy + cross_len), color, 2)

    cv2.line(frame, (cx, y_min), (cx, y_max), color, 1)
    cv2.line(frame, (x_min, cy), (x_max, cy), color, 1)

    w = x_max - x_min
    h = y_max - y_min
    box_height_cm = get_height_cm((cx, cy), depth_map)
    if box_height_cm is None:
        height_text = "Height=NA"
    else:
        height_text = f"Height={box_height_cm:.1f}cm"

    draw_detection_info(
        frame,
        x_min,
        min(frame.shape[0] - 5, y_max + 18),
        [f"c=({cx},{cy}) w={w} h={h}", height_text],
        color
    )

    return cx, cy


def pixel_to_3d_coord(pixel_x, pixel_y, intr_matrix, meters_z):
    pixel = np.array([pixel_x, pixel_y, 1.0])
    point_3d = meters_z * np.linalg.inv(intr_matrix) @ pixel
    return point_3d

def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
    config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_cm = depth_sensor.get_depth_scale() * 100.0
    align = rs.align(rs.stream.color)
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    intr = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx = float(intr.fx)
    fy = float(intr.fy)
    ppx = float(intr.ppx)
    ppy = float(intr.ppy)
    color_intr_matrix = np.array([
        [fx, 0.0, ppx],
        [0.0, fy, ppy],
        [0.0, 0.0, 1.0]
    ])
    return pipeline, align, depth_scale_cm, color_intr_matrix

def main():
    node = rclpy.create_node("detect_bed")
    socket_center_pub = Node.create_publisher(node, PoseStamped, CENTER_TOPIC, 10)

    pipeline, align, depth_scale_cm, col_intr_matrix = start_realsense()
    camera_width = CAMERA_WIDTH
    camera_height = CAMERA_HEIGHT

    # Initialize ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Load YOLO model
    model_path = "../assets/sockets_and_bottles.pt"  # sockets_and_bottles \\ bed_augmented
    model = YOLO(model_path)
    model_class_names = {int(class_id): name for class_id, name in model.names.items()}
    print(f"Loaded YOLO model: {model_path}")
    print(f"Model classes: {model_class_names}")


    try:
        print(f"Camera Resolution: {camera_width}x{camera_height}")
        if np.isfinite(H_CM):
            print(f"Using fixed H: {H_CM} cm")
        else:
            print("Using H from depth map")

        # --- CROP SETTINGS ---
        USE_CROP = True
        crop_width, crop_height = 640, 480

        center_x, center_y = camera_width // 2, camera_height // 2
        x1, y1 = max(0, center_x - crop_width // 2), max(0, center_y - crop_height // 2)
        x2, y2 = min(camera_width, center_x + crop_width // 2), min(camera_height, center_y + crop_height // 2)

        stable_bottle_box = None
        stable_bottle_confidence = 0.0
        stable_bottle_missed_frames = 0
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError as exc:
                print(f"RealSense frame error: {exc}")
                print("Camera stream stopped. Reconnect the camera and restart the script.")
                break

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            depth_map = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale_cm

            if USE_CROP:
                cropped_frame = frame[y1:y2, x1:x2]
                depth_map = depth_map[y1:y2, x1:x2]
            else:
                cropped_frame = frame

            center_bed = None
            center_socket = None

            # --- YOLO DETECTION ---
            model_confidence = min(BOTTLE_CONFIDENCE, SOCKET_CONFIDENCE)
            results = model(cropped_frame, conf=model_confidence, verbose=False)
            annotated_frame = cropped_frame.copy()
            yolo_detections = []
            bottle_candidates = []

            for result in results:
                for box in result.boxes:
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0].cpu().numpy())

                    name = model_class_names.get(class_id, f"class_{class_id}")
                    normalized_name = name.lower()
                    color = get_detection_color(normalized_name)
                    bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

                    if normalized_name == "socket" and confidence < SOCKET_CONFIDENCE:
                        continue

                    detection = {
                        "bbox": bbox,
                        "confidence": confidence,
                        "name": name,
                        "normalized_name": normalized_name,
                        "color": color,
                    }
                    yolo_detections.append(detection)

                    if normalized_name == "bottle":
                        if confidence >= BOTTLE_CONFIDENCE:
                            bottle_candidates.append(detection)
                        continue

            if bottle_candidates:
                best_bottle = max(bottle_candidates, key=lambda item: item["confidence"])
                if stable_bottle_box is None:
                    stable_bottle_box = best_bottle["bbox"]
                else:
                    stable_bottle_box = (
                        BOTTLE_SMOOTHING_ALPHA * best_bottle["bbox"]
                        + (1.0 - BOTTLE_SMOOTHING_ALPHA) * stable_bottle_box
                    )
                stable_bottle_confidence = best_bottle["confidence"]
                stable_bottle_missed_frames = 0
            elif stable_bottle_box is not None:
                stable_bottle_missed_frames += 1
                if stable_bottle_missed_frames > BOTTLE_MAX_MISSED_FRAMES:
                    stable_bottle_box = None
                    stable_bottle_confidence = 0.0

            for detection in yolo_detections:
                normalized_name = detection["normalized_name"]
                if normalized_name == "bottle":
                    continue
                if (
                    normalized_name == "socket"
                    and stable_bottle_box is not None
                    and intersection_over_area(detection["bbox"], stable_bottle_box) >= SOCKET_BOTTLE_SUPPRESS_IOA
                ):
                    continue

                cx, cy = draw_yolo_detection(
                    annotated_frame,
                    detection["bbox"],
                    detection["name"],
                    detection["confidence"],
                    depth_map,
                    detection["color"]
                )

                if normalized_name == "bed":
                    center_bed = (cx, cy)
                    bed_confidence = detection["confidence"]
                elif normalized_name == "socket":
                    center_socket = (cx, cy)
                    socket_confidence = detection["confidence"]

            if stable_bottle_box is not None:
                draw_yolo_detection(
                    annotated_frame,
                    stable_bottle_box,
                    "bottle",
                    stable_bottle_confidence,
                    depth_map,
                    get_detection_color("bottle")
                )

            publish_center = center_socket if center_socket is not None else center_bed
            publish_height_cm = get_height_cm(publish_center, depth_map)
            if publish_center is not None:
                publish_center = (publish_center[0] + x1, publish_center[1] + y1) # back to the original pixel coord
            publish_detected_center(socket_center_pub, publish_center, publish_height_cm, col_intr_matrix)

            # --- Show ---
            cv2.imshow("Camera Stream", annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        try:
            pipeline.stop()
        except RuntimeError as exc:
            print(f"Warning: failed to stop RealSense pipeline cleanly: {exc}")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rclpy.init()
    main()
    rclpy.shutdown()
