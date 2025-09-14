from ultralytics import YOLO
import os

# load yolo model and predict
model = YOLO("models/yolov11m-LP.pt")
def detect_plates_filtered(image_path, conf=0.25, imgsz=1536, max_det=50, max_ratio=0.03):
    print(f"Detecting Faces in {image_path}")
    results_raw = model.predict(
        source=image_path,
        save=True,
        show=False,
        project="images/predictions",
        name="plates",
        conf=conf, imgsz=imgsz, max_det=max_det

    )
    results_filtered = []

    for r in results_raw:
        h, w = r.orig_shape[:2]
        area_img = h * w

        # first filter: ratio to image size
        areas = (r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])
        keep_area = areas / area_img <= max_ratio

        # second filter: images beneath horizontal axis
        centers_y = (r.boxes.xyxy[:, 1] + r.boxes.xyxy[:, 3]) / 2
        keep_bottom = centers_y >= h / 2

        # filtering
        keep = keep_area & keep_bottom
        r.boxes = r.boxes[keep]

        results_filtered.append(r)

        print(f"{os.path.basename(r.path)}: {len(r.boxes)} boxes")

    return results_filtered
