Object Detection and Text Recognition with Voice Output
Overview
This project leverages computer vision techniques to perform real-time object detection and text recognition using YOLO (You Only Look Once) and Tesseract OCR (Optical Character Recognition). Detected objects and recognized text are then converted to speech using a text-to-speech engine.

Features
Real-time Object Detection: Identifies various objects using the YOLO model and displays them on the webcam feed.
Text Recognition: Uses Tesseract OCR to extract text from the video feed.
Text-to-Speech Output: Converts detected object names and recognized text into speech for auditory feedback.
Requirements
Python 3.x
OpenCV
Tesseract OCR
gTTS (Google Text-to-Speech)
pyttsx3
YOLOv8 (Ultralytics)
cvzone
You can install the required packages using pip:

bash
Copy code
pip install opencv-python pytesseract gtts pyttsx3 ultralytics cvzone
Make sure to install Tesseract OCR separately and set the pytesseract.pytesseract.tesseract_cmd to the path of the Tesseract executable.

Setup Tesseract OCR
Download and install Tesseract OCR from Tesseract's official repository.

Update the pytesseract.pytesseract.tesseract_cmd line in the code to point to the Tesseract installation path:

python
Copy code
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
How to Run the Application
Clone the repository:

bash
Copy code
git clone https://github.com/<your_username>/object-detection-text-recognition.git
cd object-detection-text-recognition
Ensure that you have the YOLOv8 model weights downloaded and placed in the Yolo-Weights directory.

Run the Python script:

bash
Copy code
python app.py
Grant permission for your webcam when prompted.

Usage
The application will open a webcam feed.
As objects are detected or text is recognized in the video feed, the corresponding names will be displayed on the screen and announced through speech.
The application continuously processes the webcam feed until it is manually closed.
License
This project is licensed under the MIT License. Feel free to modify and distribute as needed.

Acknowledgments
Thanks to the Ultralytics team for the YOLO model.
Thanks to the Tesseract OCR contributors for making text recognition accessible.
