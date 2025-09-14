from ultralytics import YOLO

# load yolo model and predict
model=YOLO("models/yolov11m-face.pt")
def detect_faces(image_path, conf=0.25, imgsz=1536, max_det=50):
    
    print(f"Detecting Faces in {image_path}")
    results = model.predict(
        source=image_path,
        show=False,
        save=True,
        project="images/predictions",
        name="faces", conf=conf, imgsz=imgsz, max_det=max_det
    )
    return results

