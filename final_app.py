import math
import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def load_models():
    """Load object detection and pose estimation models."""
    model_candidates = [
        "runs/detect/custom_bag_chair_model-3/weights/best.pt",
        "yolo11l.pt",
        "yolov8n.pt",
    ]
    # Pick first available candidate or default to yolo11l.pt / yolov8n.pt
    model_path = next((p for p in model_candidates if os.path.exists(p)), "yolov8n.pt")

    print(f"Loading object model from: {model_path}")
    object_model = YOLO(model_path)

    pose_candidates = ["yolo11s-pose.pt", "yolov8n-pose.pt"]
    pose_path = next((p for p in pose_candidates if os.path.exists(p)), "yolov8n-pose.pt")
    
    print(f"Loading pose model from: {pose_path}")
    pose_model = YOLO(pose_path)

    return object_model, pose_model


def get_video_path():
    """Retrieve video file path from command line arguments or directory defaults."""
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]

    video_candidates = ["longvid1.mp4", "1000093579.mp4"]
    for candidate in video_candidates:
        if os.path.exists(candidate):
            return candidate

    mp4_files = [f for f in os.listdir(".") if f.endswith(".mp4") and not f.startswith("output_")]
    if mp4_files:
        return mp4_files[0]

    return "longvid1.mp4"


def detect_bags(obj_pred, class_names):
    """Extract bag detections (boxes and tracker IDs) from YOLO output."""
    bags = []
    bag_labels = {"bag", "backpack", "handbag", "suitcase", "luggage"}

    if obj_pred.boxes is not None and len(obj_pred.boxes) > 0:
        boxes = obj_pred.boxes.xyxy.cpu().numpy()
        classes = obj_pred.boxes.cls.cpu().numpy().astype(int)

        # Retrieve tracking IDs if tracking is active; otherwise assign placeholder IDs
        if obj_pred.boxes.id is not None:
            ids = obj_pred.boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.arange(len(boxes))

        for box, cls_id, obj_id in zip(boxes, classes, ids):
            name = str(class_names.get(cls_id, cls_id)).lower()
            if name in bag_labels:
                bags.append({"id": int(obj_id), "box": box})

    return bags


def get_global_wrists(pose_pred, min_conf=0.3):
    """Extract left and right wrist keypoints with basic confidence filtering."""
    wrists = []
    if pose_pred.keypoints is not None and len(pose_pred.keypoints.xy) > 0:
        skeletons = pose_pred.keypoints.xy.cpu().numpy()
        has_conf = pose_pred.keypoints.conf is not None
        confs = pose_pred.keypoints.conf.cpu().numpy() if has_conf else None

        for idx, skel in enumerate(skeletons):
            if len(skel) > 10:
                l_conf = confs[idx][9] if has_conf else 1.0
                r_conf = confs[idx][10] if has_conf else 1.0

                wrists.append({
                    "left": skel[9] if l_conf >= min_conf else np.array([0, 0]),
                    "right": skel[10] if r_conf >= min_conf else np.array([0, 0]),
                })
    return wrists


def point_inside_box(point, box):
    """Check if 2D point (x, y) is inside the rectangular boundary."""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def is_wrist_inside_roi(wrists, roi_box, padding=25):
    """Determine whether any wrist is inside the expanded ROI."""
    rx1, ry1, rx2, ry2 = roi_box
    padded_roi = [rx1 - padding, ry1 - padding, rx2 + padding, ry2 + padding]

    for hand in wrists:
        for side in ("left", "right"):
            wx, wy = hand[side]
            if wx > 0 and wy > 0 and point_inside_box((wx, wy), padded_roi):
                return True
    return False


def is_hand_near_bag(bag_box, wrists):
    """Check if any wrist is in close physical proximity to the bag."""
    bx1, by1, bx2, by2 = bag_box
    bag_cx = (bx1 + bx2) / 2
    bag_cy = (by1 + by2) / 2
    proximity_limit = max(80, 0.6 * max(bx2 - bx1, by2 - by1))

    for hand in wrists:
        for side in ("left", "right"):
            wx, wy = hand[side]
            if wx > 0 and wy > 0:
                if math.dist((bag_cx, bag_cy), (wx, wy)) <= proximity_limit:
                    return True
    return False


def update_bag_state_machine(bag_tracker, bag_box, roi_box, wrists):
    """
    Sequential 4-Phase Transition Machine:
      - INITIAL: Bag sits inside ROI.
      - PICKING: Wrist + Bag detected inside ROI (grasping/lifting).
      - PICKED: Wrist + Bag moved completely out of the ROI.
      - PLACING: Wrist + Bag re-enter the ROI area after being PICKED.
      - PLACED: Bag stays inside ROI while wrists retract/exit.
    """
    bx1, by1, bx2, by2 = bag_box
    bag_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)

    bag_in_roi = point_inside_box(bag_center, roi_box)
    wrist_in_roi = is_wrist_inside_roi(wrists, roi_box)
    hand_near = is_hand_near_bag(bag_box, wrists)

    # State conditions
    cond_picking = bag_in_roi and wrist_in_roi and hand_near
    cond_picked = (not bag_in_roi) and (not wrist_in_roi)
    cond_placing = (bag_in_roi and wrist_in_roi) or (bag_in_roi and hand_near)
    cond_placed = bag_in_roi and (not wrist_in_roi) and (not hand_near)

    frame_counters = bag_tracker["frames"]

    def update_counter(name, active):
        frame_counters[name] = frame_counters[name] + 1 if active else 0
        return frame_counters[name] >= 3  # Stable for at least 3 consecutive frames

    current_state = bag_tracker["state"]

    if current_state in ("INITIAL", "PLACED"):
        # Transition to PICKING if hand grabs bag in ROI
        if update_counter("picking", cond_picking):
            bag_tracker["state"] = "PICKING"
    elif current_state == "PICKING":
        # Transition to PICKED once both have departed ROI
        if update_counter("picked", cond_picked):
            bag_tracker["state"] = "PICKED"
    elif current_state == "PICKED":
        # Transition to PLACING once bag and hand re-enter ROI
        if update_counter("placing", cond_placing):
            bag_tracker["state"] = "PLACING"
    elif current_state == "PLACING":
        # Transition to PLACED once hand releases bag in ROI
        if update_counter("placed", cond_placed):
            bag_tracker["state"] = "PLACED"

    return bag_tracker["state"]


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")

    object_model, pose_model = load_models()
    video_path = get_video_path()
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Failed to open video source: {video_path}")
        return

    # Track state history and ROI coordinates per bag ID
    bag_records = {}

    color_map = {
        "INITIAL": (200, 200, 200),
        "PICKING": (0, 165, 255),    # Orange
        "PICKED": (0, 0, 255),       # Red
        "PLACING": (255, 0, 255),    # Magenta
        "PLACED": (0, 255, 0),       # Green
    }

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        # Run detection and tracking
        obj_results = object_model.track(frame, persist=True, device=device, verbose=False)
        pose_results = pose_model(frame, device=device, verbose=False)

        obj_pred = obj_results[0]
        pose_pred = pose_results[0]

        bags = detect_bags(obj_pred, object_model.names)
        all_wrists = get_global_wrists(pose_pred)

        # Draw detected wrists once per frame
        for hand in all_wrists:
            for side in ("left", "right"):
                wx, wy = hand[side]
                if wx > 0 and wy > 0:
                    cv2.circle(frame, (int(wx), int(wy)), 6, (0, 255, 255), -1)

        # Process each bag
        for bag in bags:
            bag_id = bag["id"]
            bx1, by1, bx2, by2 = bag["box"]

            # Initialize tracker entry and lock initial ROI bounding box
            if bag_id not in bag_records:
                margin = 40
                roi_box = [
                    max(int(bx1) - margin, 0),
                    max(int(by1) - margin, 0),
                    int(bx2) + margin,
                    int(by2) + margin,
                ]
                bag_records[bag_id] = {
                    "roi": roi_box,
                    "state": "INITIAL",
                    "frames": {"picking": 0, "picked": 0, "placing": 0, "placed": 0},
                }

            record = bag_records[bag_id]
            roi = record["roi"]

            # Evaluate state machine
            current_state = update_bag_state_machine(record, bag["box"], roi, all_wrists)
            state_color = color_map.get(current_state, (255, 255, 255))

            # 1. Draw ROI Box
            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 128, 0), 2)
            cv2.putText(
                frame,
                f"Bag {bag_id} ROI",
                (rx1, max(ry1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 128, 0),
                2,
            )

            # 2. Draw Bag Bounding Box
            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), state_color, 3)
            cv2.putText(
                frame,
                f"Bag {bag_id}: {current_state}",
                (int(bx1), max(int(by1) - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                state_color,
                2,
            )

        # Render preview
        resized = cv2.resize(frame, (960, 540))
        cv2.imshow("Activity State Machine", resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()