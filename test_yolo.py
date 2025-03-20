import cv2
from ultralytics import YOLO

# yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
model = YOLO("last.pt")

#image = cv2.imread("image.jpg")

cam = cv2.VideoCapture(0)

while True:
    ret, image=cam.read()
    image = cv2.resize(image, (640, 480))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)
    print(model.names)
    print(results[0].boxes)

    classes = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    bboxes = results[0].boxes.xyxy.cpu().numpy()

    print(classes)
    print(confs)
    print(bboxes)

    for i in range(len(classes)):
        class_ = classes[i]
        conf_ = confs[i]
        bbox_ = bboxes[i] #[x1, y1, x2, y2]
        
        x1, y1, x2, y2 = bbox_
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        class_name = model.names[class_]
        cv2.putText(image, f"{class_name}:{conf_*100:.2f}%", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        

    cv2.imshow("image", image)
    cv2.waitKey(1)