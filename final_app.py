import math
import os
import cv2
import numpy as np
from ultralytics import YOLO

def load_models():
    model_path = "runs/detect/custom_bag_chair_model-3/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "yolov11l.pt"

    object_model = YOLO(model_path)
    pose_model = YOLO("yolo11s-pose.pt")
    return object_model, pose_model

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

def detect_objects(obj_pred):
    seating_boxes = []
    bags = []

    if obj_pred.boxes is not None and len(obj_pred.boxes) > 0:
        boxes = obj_pred.boxes.xyxy.cpu().numpy()
        classes = obj_pred.boxes.cls.cpu().numpy().astype(int)
        
        if obj_pred.boxes.id is not None:
            ids = obj_pred.boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.array([999] * len(boxes))

        for box, cls_id, obj_id in zip(boxes, classes, ids):
            if cls_id == 1:
                seating_boxes.append(box)
            elif cls_id == 0:
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

def main():
    object_model, pose_model = load_models()
    video_path = "1000093579.mp4"
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

        seating_boxes, bags = detect_objects(obj_pred)
        all_wrists = get_global_wrists(pose_pred)

        for bag in bags:
            bag_id = int(bag["id"])
            bag_box = bag["box"]
            bx1, by1, bx2, by2 = bag_box
            bag_cx = (bx1 + bx2) / 2
            bag_cy = (by1 + by2) / 2

            # Create a fixed ROI only once, when the bag is first detected.
            if bag_id not in bag_states:
                margin = 25
                ref_box = [
                    max(bx1 - margin, 0),
                    max(by1 - margin, 0),
                    bx2 + margin,
                    by2 + margin,
                ]
                bag_states[bag_id] = {
                    "ref_box": ref_box,
                    "picked": False,
                    "frames_outside": 0,
                }

            ref_box = bag_states[bag_id]["ref_box"]
            rx1, ry1, rx2, ry2 = ref_box
            roi_cx = (rx1 + rx2) / 2
            roi_cy = (ry1 + ry2) / 2
            roi_w = max(rx2 - rx1, 10)
            roi_h = max(ry2 - ry1, 10)

            # More stable than checking x/y separately.
            center_dist = math.hypot(bag_cx - roi_cx, bag_cy - roi_cy)
            move_threshold = 0.5 * max(roi_w, roi_h)
            bag_outside_roi = center_dist > move_threshold

            if bag_outside_roi:
                bag_states[bag_id]["frames_outside"] += 1
            else:
                bag_states[bag_id]["frames_outside"] = 0

            # Require a few stable frames before switching to picked-up state.
            picked_up = bag_states[bag_id]["frames_outside"] >= 3
            bag_states[bag_id]["picked"] = picked_up

            if picked_up:
                label = "PICKED UP"
                color = (0, 255, 0)
            else:
                label = "DROPPED / NOT PICKED"
                color = (0, 0, 255)

            cv2.rectangle(frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255, 128, 0), 1)
            cv2.putText(frame, "ROI", (int(rx1) + 5, int(ry1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)

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
