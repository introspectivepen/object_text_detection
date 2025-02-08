import cv2
import pytesseract
from gtts import gTTS
import os
from datetime import datetime
from playsound import playsound

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to generate unique filenames based on timestamp
def generate_filename(prefix, extension):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def detect_text(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform text detection using Tesseract
    text = pytesseract.image_to_string(gray)

    return text

def text_to_speech(text):
    # Convert text to speech
    tts = gTTS(text)
    filename = generate_filename("output", "mp3")
    tts.save(filename)
    playsound(filename)

# Webcam setup
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Show the frame
    cv2.imshow('Webcam', frame)

    # Check for key press
    key = cv2.waitKey(1)

    # Press 'c' key to capture image and perform text extraction
    if key & 0xFF == ord('c'):
        # Save the captured image
        filename = generate_filename("captured_image", "jpg")
        cv2.imwrite(filename, frame)

        # Perform text detection on the captured image
        detected_text = detect_text(frame)

        # Print the detected text
        print("Detected Text:", detected_text)

        # Convert the detected text to speech
        text_to_speech(detected_text)

    # Press 'q' key to exit the loop
    elif key & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

