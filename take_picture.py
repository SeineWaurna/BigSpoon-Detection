import cv2

cap = cv2.VideoCapture(0)
counter = 0

while True:
    ret, frame = cap.read()
    
    cv2. imwrite(f"images/picture_{counter}.jpg", frame)
    counter += 1
    
    cv2.imshow("frame", frame)
    cv2.waitKey(1000)