import io
import os
import cv2
from Models.audio import say

# Import the required libraries
from google.cloud import vision_v1
from google.oauth2.service_account import Credentials

# Set the path to your service account key
key_path = 'C:\\Users\\nithi\\OneDrive\\Desktop\\project\\CODE COPY\\named-indexer-343506-aaf17bdfc53d.json'

# Create a credentials object using the service account key
credentials = Credentials.from_service_account_file(key_path)

# Create a client object for interacting with the Google Cloud Vision API
client = vision_v1.ImageAnnotatorClient(credentials=credentials)

# Initialize the video capture object to capture frames from the webcam
cap = cv2.VideoCapture(1)

# Set to keep track of unique words
unique_words = set()

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to JPEG format
    ret, buffer = cv2.imencode('.jpg', frame)
    image_data = buffer.tobytes()

    # Create an image object using the image data
    image = vision_v1.types.Image(content=image_data)

    # Perform OCR on the image using the Google Cloud Vision API
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Print the extracted text from the image
    for text in texts:
        # Only speak each word once
        if text.description not in unique_words:
            unique_words.add(text.description)
            say(text.description)

    # Display the captured frame with text annotations
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
