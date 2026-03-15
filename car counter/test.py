from ultralytics import YOLO
import cv2
import math
import numpy as np
from sortFilter import *
import time

# Video capture
cap = cv2.VideoCapture(r'd:/image detection/yolo running/Videos/cars3.mp4')

# Display settings
display_width = 1280
display_height = 720

# YOLO class names
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

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Load YOLOv8 model
model = YOLO("../yolo weight/yolov8n.pt")

# Parameters for car detection
CAR_CONFIDENCE_THRESHOLD = 0.3
MAX_CAR_AREA_RATIO = 0.15
MIN_CAR_AREA_RATIO = 0.0005
CAR_ASPECT_RATIO_MIN = 0.4
CAR_ASPECT_RATIO_MAX = 2.5

# To track cars that have been counted
car_count = 0
tracked_cars = set()

# For FPS calculation
prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("Video ended or failed to read frame")
        break
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time
    
    # Get frame dimensions
    frame_height, frame_width = img.shape[:2]
    frame_area = frame_height * frame_width
    
    # Run detection
    results = model(img, stream=True)
    
    # Initialize empty array for detections
    detections = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence level
            conf = math.ceil(box.conf[0] * 100) / 100
            
            # Get class name
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
                # Add to detections array for tracking
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    
    # Update tracker with new detections
    result_tracker = tracker.update(detections)
    
    # Draw car detections and IDs
    for result in result_tracker:
        x1, y1, x2, y2, track_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Add new tracked car to total count
        if int(track_id) not in tracked_cars:
            tracked_cars.add(int(track_id))
            car_count += 1
        
        # Create unique ID label
        id_label = f'ID: {int(track_id)}'
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        # Calculate text size for ID label
        id_text_size = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Draw background rectangle for ID
        cv2.rectangle(img,
                    (x1, y1 - id_text_size[1] - 10),
                    (x1 + id_text_size[0] + 10, y1),
                    (255, 0, 255), -1)
        
        # Add ID text
        cv2.putText(img, id_label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw corner markers
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
    
    # Draw counter and FPS information on top left
    counter_info = f'Cars Counted: {car_count}'
    fps_info = f'FPS: {int(fps)}'
    
    # Add counter info at top
    cv2.putText(img, counter_info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, fps_info, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Resize and display
    resized_img = cv2.resize(img, (display_width, display_height))
    cv2.imshow("Vehicle Tracking", resized_img)
    
    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Window is closing...")
        print(f"Total cars counted: {car_count}")
        break

cap.release()
cv2.destroyAllWindows()