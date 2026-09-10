import math
import os
import cv2
import numpy as np
from ultralytics import YOLO

def load_models():
    model_candidates = [
        "runs/detect/custom_bag_chair_model-3/weights/best.pt",
        "yolov11l.pt",
    ]
    model_path = next((p for p in model_candidates if os.path.exists(p)), model_candidates[-1])

    object_model = YOLO(model_path)
    pose_model = YOLO("yolo11s-pose.pt")
    return object_model, pose_model


def get_video_path():
    video_candidates = [
        "1000093579.mp4",
        "1000093632.mp4",
        "1000093653.mp4",
    ]
    return next((p for p in video_candidates if os.path.exists(p)), video_candidates[0])

def calculate_chair_overlap(bag_box, chair_box):
    bx1, by1, bx2, by2 = bag_box
    cx1, cy1, cx2, cy2 = chair_box

    x_left = max(bx1, cx1)
    y_top = max(by1, cy1)
    x_right = min(bx2, cx2)
    y_bottom = min(by2, cy2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bag_area = (bx2 - bx1) * (by2 - by1)

    if bag_area <= 0:
        return 0.0

    return intersection_area / bag_area

def detect_objects(obj_pred, class_names):
    """Return bags and chairs without relying on a fixed class-id order.

    The custom model uses ``bag``/``chair`` names, while the fallback COCO
    model uses names such as ``backpack`` and ``handbag``.  Looking up the
    class name makes both models work correctly.
    """
    seating_boxes = []
    bags = []
    bag_names = {"bag", "backpack", "handbag", "suitcase", "luggage"}
    chair_names = {"chair", "seat", "seating"}

    if obj_pred.boxes is not None and len(obj_pred.boxes) > 0:
        boxes = obj_pred.boxes.xyxy.cpu().numpy()
        classes = obj_pred.boxes.cls.cpu().numpy().astype(int)
        
        if obj_pred.boxes.id is not None:
            ids = obj_pred.boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.array([999] * len(boxes))

        for box, cls_id, obj_id in zip(boxes, classes, ids):
            class_name = str(class_names.get(cls_id, cls_id)).lower()
            if class_name in chair_names:
                seating_boxes.append(box)
            elif class_name in bag_names:
                bags.append({"id": obj_id, "box": box})

    return seating_boxes, bags

def get_global_wrists(pose_pred):
    """Extracts all valid wrist keypoints from the full frame scan."""
    wrists = []
    if pose_pred.keypoints is not None and len(pose_pred.keypoints.xy) > 0:
        skeletons = pose_pred.keypoints.xy.cpu().numpy()
        for skel in skeletons:
            if len(skel) > 10:
                wrists.append({
                    "left": skel[9],   # COCO Index 9
                    "right": skel[10]  # COCO Index 10
                })
    return wrists

def point_inside_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def is_hand_near_bag(bag_box, wrists):
    """Whether a detected wrist is close enough to be handling this bag."""
    bx1, by1, bx2, by2 = bag_box
    # Measure from the closest point of the box, not its centre.  A wrist at
    # the edge of a large bag is still handling it.
    proximity_threshold = max(60, 0.25 * max(bx2 - bx1, by2 - by1))

    for hand in wrists:
        for wrist in (hand["left"], hand["right"]):
            if wrist[0] > 0 and wrist[1] > 0:
                nearest_x = min(max(wrist[0], bx1), bx2)
                nearest_y = min(max(wrist[1], by1), by2)
                if math.dist((wrist[0], wrist[1]), (nearest_x, nearest_y)) <= proximity_threshold:
                    return True
    return False


def update_bag_state(bag_state, bag_box, seating_boxes, wrists):
    """Advance a bag through detected -> picking -> picked -> placing -> placed.

    Each condition must hold for three consecutive frames, which prevents a
    one-frame pose or object detection miss from changing the displayed label.
    """
    bx1, by1, bx2, by2 = bag_box
    bag_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)
    at_start_position = point_inside_box(bag_center, bag_state["ref_box"])
    on_chair = any(calculate_chair_overlap(bag_box, chair) >= 0.20 for chair in seating_boxes)
    hand_near = is_hand_near_bag(bag_box, wrists)

    conditions = {
        "hand_near": hand_near,
        "away_with_hand": not at_start_position and hand_near,
        "on_chair_with_hand": on_chair and hand_near,
        "on_chair_without_hand": on_chair and not hand_near,
        "back_at_start_without_hand": at_start_position and not hand_near,
    }
    for name, active in conditions.items():
        bag_state["frames"][name] = bag_state["frames"][name] + 1 if active else 0

    state = bag_state["state"]
    stable = lambda name: bag_state["frames"][name] >= 3

    if state == "BAG DETECTED" and stable("hand_near"):
        bag_state["state"] = "PICKING"
    elif state == "PICKING":
        if stable("away_with_hand"):
            bag_state["state"] = "PICKED"
        elif stable("back_at_start_without_hand"):
            bag_state["state"] = "BAG DETECTED"
    elif state == "PICKED" and stable("on_chair_with_hand"):
        bag_state["state"] = "PLACING"
    elif state == "PLACING" and stable("on_chair_without_hand"):
        bag_state["state"] = "PLACED"
    elif state == "PLACED" and stable("hand_near"):
        bag_state["state"] = "PICKING"

    return bag_state["state"]

def main():
    object_model, pose_model = load_models()
    video_path = get_video_path()
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Video file not found: {video_path}")
        return

    bag_states = {}

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        obj_results = object_model.track(frame, persist=True, device=0, verbose=False)
        pose_results = pose_model(frame, device=0, verbose=False)

        obj_pred = obj_results[0]
        pose_pred = pose_results[0]

        seating_boxes, bags = detect_objects(obj_pred, object_model.names)
        all_wrists = get_global_wrists(pose_pred)

        # Show detected chairs too, so it is clear which object is used for
        # the "BAG ON CHAIR" condition.
        for chair_box in seating_boxes:
            cx1, cy1, cx2, cy2 = chair_box.astype(int)
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 128, 0), 2)
            cv2.putText(frame, "CHAIR", (cx1, max(cy1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2)

        for bag in bags:
            bag_id = int(bag["id"])
            bag_box = bag["box"]
            bx1, by1, bx2, by2 = bag_box
            bag_cx = (bx1 + bx2) / 2
            bag_cy = (by1 + by2) / 2

            if bag_id not in bag_states:
                margin = 25
                bag_states[bag_id] = {
                    "ref_box": [max(bx1 - margin, 0), max(by1 - margin, 0), bx2 + margin, by2 + margin],
                    "state": "BAG DETECTED",
                    "frames": {
                        "hand_near": 0,
                        "away_with_hand": 0,
                        "on_chair_with_hand": 0,
                        "on_chair_without_hand": 0,
                        "back_at_start_without_hand": 0,
                    },
                }

            label = update_bag_state(bag_states[bag_id], bag_box, seating_boxes, all_wrists)
            color = {
                "BAG DETECTED": (0, 0, 255),
                "PICKING": (0, 165, 255),
                "PICKED": (0, 255, 0),
                "PLACING": (255, 0, 255),
                "PLACED": (255, 255, 0),
            }[label]

            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 3)
            cv2.putText(
                frame,
                f"Bag {bag_id}: {label}",
                (int(bx1), int(by1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

            # optional hand proximity drawing remains for visual support
            proximity_threshold = 120
            for hand in all_wrists:
                lw, rw = hand["left"], hand["right"]
                dist_l = math.dist((bag_cx, bag_cy), (lw[0], lw[1])) if lw[0] > 0 else float("inf")
                dist_r = math.dist((bag_cx, bag_cy), (rw[0], rw[1])) if rw[0] > 0 else float("inf")

                if lw[0] > 0 and lw[1] > 0:
                    cv2.circle(frame, (int(lw[0]), int(lw[1])), 8, (0, 255, 255), -1)
                    cv2.putText(frame, f"Dist: {int(dist_l)}px", (int(lw[0]) + 12, int(lw[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                if rw[0] > 0 and rw[1] > 0:
                    cv2.circle(frame, (int(rw[0]), int(rw[1])), 8, (0, 255, 255), -1)
                    cv2.putText(frame, f"Dist: {int(dist_r)}px", (int(rw[0]) + 12, int(rw[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        resized = cv2.resize(frame, (854, 480))
        cv2.imshow("Bag Chair Detection", resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
