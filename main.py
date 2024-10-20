import cv2
import pytesseract
from gtts import gTTS
import os
from ultralytics import YOLO
import cvzone
import math
import time
import pyttsx3

# Initialize the Tesseract OCR engine
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the Text-to-Speech engine
engine = pyttsx3.init()

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Adjust width
cap.set(4, 480)  # Adjust height

# Initialize the YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Define COCO classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "pen", "pencil"
              ]

prev_frame_time = 0
new_frame_time = 0


# Function to convert text to speech
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()


# Main loop
while True:
    new_frame_time = time.time()
    success, img = cap.read()

    # Resize input image
    img = cv2.resize(img, (640, 480))  # Adjust to match webcam resolution

    # Perform object detection with YOLO
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            detected_text = classNames[cls]
            # Convert the detected text to speech
            text_to_speech(detected_text)

            cvzone.putTextRect(img, f'{detected_text} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Perform text detection using Tesseract
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_text = pytesseract.image_to_string(gray)
    # Convert the detected text to speech
    text_to_speech(detected_text)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(100)  # Reduce frame rate to 10 fps

# Release the webcam
cap.release()
cv2.destroyAllWindows()
