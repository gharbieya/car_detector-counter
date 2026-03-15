#this file detects objects from recorded images using yolo architecture

from ultralytics import YOLO
import cv2

model = YOLO('../yolo weight/yolo11n.pt')
results = model(r'd:/image detection car counter ai model/yolo running/Images/6.jpg', show=True)
cv2.waitKey(0)