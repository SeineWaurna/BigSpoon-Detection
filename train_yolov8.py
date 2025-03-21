from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8x.pt")

    results = model.train(
        data="D:/AI/bigspoon/data/data.yaml",
        epochs=500,
        imgsz=416
        )