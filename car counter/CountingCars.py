from ultralytics import YOLO
import cv2
import math
import numpy as np
from sortFilter import *

cap = cv2.VideoCapture(r'd:/image detection/yolo running/Videos/cars3.mp4')

display_width = 1280
display_height = 720

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
             "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
             "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
             "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
             "skis", "snowboard", "sports ball", "kite", "baseball", "baseball bat",
             "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
             "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
             "cake", "chair", "sofa", "potted plant", "bed", "dining table",
             "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
             "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
             "vase", "scissors", "teddy bear", "hair dryer", "toothbrush"]

traker = Sort(max_age=15, min_hits=3, iou_threshold=0.3) #max age isnumber of frame 

limit_line = [300, 400, 1400, 400] #for video cars3

total_counts = 0
counted_cars = dict()

# Vehicle class indices in YOLO
vehicle_indices = {
    "car": classNames.index("car"),
    "truck": classNames.index("truck"),
    "bus": classNames.index("bus")
}

# Load YOLOv8 model
model = YOLO("../yolo weight/yolov8n.pt")

# Parameters for car detection
# These can be adjusted based on your specific needs
CAR_CONFIDENCE_THRESHOLD = 0.3
MAX_CAR_AREA_RATIO = 0.15  # Maximum area ratio for a car (to filter out larger vehicles)
MIN_CAR_AREA_RATIO = 0.0005  # Minimum area ratio for a car
CAR_ASPECT_RATIO_MIN = 0.4  # Minimum width/height ratio for cars
CAR_ASPECT_RATIO_MAX = 2.5  # Maximum width/height ratio for cars

while True:
    success, img = cap.read()
    if not success:
        print("Video ended or failed to read frame")
        break
    
    # Get frame dimensions
    frame_height, frame_width = img.shape[:2]
    frame_area = frame_height * frame_width
    
    results = model(img, stream=True)

    detections = dets=np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # confidence level value
            conf = math.ceil(box.conf[0] * 100) / 100
            
            # class names
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            # Calculate box properties for filtering
            width = x2 - x1
            height = y2 - y1
            area = width * height
            area_ratio = area / frame_area
            aspect_ratio = width / height if height > 0 else 0
            
            # Filter for cars based on multiple criteria
            is_car = False
            
            if currentClass == "car" and conf > CAR_CONFIDENCE_THRESHOLD:
                # Additional filtering based on size and aspect ratio
                if (area_ratio < MAX_CAR_AREA_RATIO and 
                    area_ratio > MIN_CAR_AREA_RATIO and
                    aspect_ratio > CAR_ASPECT_RATIO_MIN and 
                    aspect_ratio < CAR_ASPECT_RATIO_MAX):
                        is_car = True
                    
            if is_car:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    
    resultTracker = traker.update(detections)
    cv2.line(img, (limit_line[0], limit_line[1]), (limit_line[2], limit_line[3]), (0, 0, 255), 5) #color is red and 5 is the thickness
    
    for result in resultTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result) 
        label = f'ID : {int(Id)}'
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.rectangle(img,
                        (max(0, x1), max(35, y1) - text_size[1] - 10),
                        (max(0, x1) + text_size[0] + 10, max(35, y1)),
                        (255, 0, 255), -1)  # -1 fills the rectangle
        cv2.putText(img, label,
                         (max(0, x1) + 5, max(35, y1) - 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
         
        l = 9  # corner length
        t = 1  # thickness
        # Top Left
        cv2.line(img, (x1, y1), (x1 + l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 + l), (255, 0, 255), t)
        # Top Right
        cv2.line(img, (x2, y1), (x2 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x2, y1), (x2, y1 + l), (255, 0, 255), t)
        # Bottom Left
        cv2.line(img, (x1, y2), (x1 + l, y2), (255, 0, 255), t)
        cv2.line(img, (x1, y2), (x1, y2 - l), (255, 0, 255), t)
        # Bottom Right
        cv2.line(img, (x2, y2), (x2 - l, y2), (255, 0, 255), t)
        cv2.line(img, (x2, y2), (x2, y2 - l), (255, 0, 255), t)
    
        cx, cy = x1+(x2-x1)//2, y1+(y2-y1)//2
        cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)   #5 is radius and color is purple

        if limit_line[0]<cx<limit_line[2] and limit_line[1]-15<cy<limit_line[1]+15 :
            if int(Id) not in counted_cars.keys():
                total_counts += 1
                counted_cars[int(Id)] = True
                cv2.line(img, (limit_line[0], limit_line[1]), (limit_line[2], limit_line[3]), (255, 0, 0), 5) #color is red and 5 is the thickness



    cv2.putText(img, f'Count : {total_counts}',(255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 8)
    #cvzone.putTextRect(img, f'Count : {total_counts}', (50,50))
    
    resized_img = cv2.resize(img, (display_width, display_height))
    cv2.imshow("Output", resized_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Window is closing ...")
        break

cap.release()
cv2.destroyAllWindows()