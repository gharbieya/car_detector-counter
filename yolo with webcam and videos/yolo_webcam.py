from ultralytics import YOLO
import cv2
import math

#with web cam
capWebcam = cv2.VideoCapture(0)
capWebcam.set(3, 1280)
capWebcam.set(4, 720)
capWebcam = cv2.VideoCapture(0)

#with a recorded video
cap = cv2.VideoCapture(r'd:/image detection/yolo running/Videos/bicycle.mp4')
display_width = 1280  
display_height = 720 

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
             "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
             "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
             "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
             "skis", "snowboard", "sports ball", "kite", "baseball", "baseballkit",
             "baseball gloves", "skateborad", "surfboard", "tennis racket", "bottle", 
             "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
             "cake", "chair", "chair", "sofa", "potted plant", "bed","dining table",
             "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cellphone",
             "microwave", "oven", "toaster", "sink","refrigirator", "book", "crock",
             "vase", "scissors", "teddy bear", "hair dryer", "toothbrush", "lamp", "led", "bulb", "light" ]

model = YOLO("../yolo weight/yolov8l.pt")

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # confidence level value
            conf = math.ceil(box.conf[0] * 100) / 100
            print(conf)
            
            # class names
            cls = int(box.cls[0])
            label = f'{classNames[cls]} {conf}'
            
            # Get text size for proper background rectangle
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1 )[0]
            
            # Create background rectangle for text
            cv2.rectangle(img, 
                        (max(0, x1), max(35, y1) - text_size[1] - 10), 
                        (max(0, x1) + text_size[0] + 10, max(35, y1)), 
                        (255, 0, 255), -1)  # -1 fills the rectangle
            
            # Add text
            cv2.putText(img, label, 
                    (max(0, x1) + 5, max(35, y1) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1 )
            

    resized_img = cv2.resize(img, (display_width, display_height))
    cv2.imshow("Output", resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Window is closing ...")
        break

cap.release()
cv2.destroyAllWindows()

   
    # for r in results:
    #     boxes = r.boxes
    #     for box in boxes:

    #         x1, y1, x2, y2 = box.xyxy[0]
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
    #         conf = math.ceil(box.conf[0] * 100) / 100      #confidnce level

    #         cls = int(box.cls[0])    #cls is class
    #         currentClass = classNames[cls]
    #         if currentClass == "car" and conf > 0.3:
    #           cvzone.putTextRect(img, f'{currentClass}', f'{conf}', ( max(0, x1), max(35, y1)), scale = 0.6, thickness = 1, offset=3 )
    #           cvzone.cornerRect(img,(x1, y1, x2, y2), l=9)