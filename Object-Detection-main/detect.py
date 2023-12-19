import torch
import cv2
import numpy as np
from gtts import gTTS
import os

# Model selection
model_s = torch.load("/home/aadarsha/Desktop/Object-Detection/yolov8s.pt", map_location="cpu") # Map to CPU for Raspberry Pi

# Sound mapping for detected objects
sound_mapping = {
  "person": "Person ahead.",
  "vehicle": "Vehicle approaching.",
  "pothole": "Caution, pothole ahead.",
  "door": "Door nearby.",
  "stopsign": "Stop sign detected.",
  "cellphone": "Cellphone in sight.",
  "crosswalk": "Crosswalk nearby.",
  "monitor": "Monitor detected.",
  "table": "Table nearby.",
  "window": "Window ahead.",
}


def detect_and_speak(camera_index=0):
  # Open camera
  cap = cv2.VideoCapture(camera_index)
  if not cap.isOpened():
    print("Error opening camera")
    exit(1)

  while True:
    # Capture frame
    ret, img = cap.read()
    if not ret:
      break

    # Predict with chosen model
    results_s = model_s(img)

    # Get detections and print them for debugging
    detections = results_s.pandas().xyxy[0]
    print(f"Detections: {detections}")

    # Check for empty detections (model might not detect anything)
    if detections.empty():
      print("No objects detected.")
      continue

    # Loop through detections
    for index, row in detections.iterrows():
      label = row["name"]
      x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])

      # Play sound based on mapping
      if label in sound_mapping:
        speech = sound_mapping[label]
        print(f"Playing audio for: {speech}")

        # Create audio data with gTTS
        tts = gTTS(speech, lang='en')

        # Create a temporary audio filename with a unique timestamp
        temp_filename = f"temp_{round(time.time())}.mp3"

        # Save the audio file
        tts.save(temp_filename)

        # Check if the temporary audio file exists (for debugging)
        if os.path.exists(temp_filename):
          print("Temporary audio file created.")

        # Play the audio file using omxplayer for Raspberry Pi
        os.system(f"omxplayer -o local {temp_filename}")

        # Delete the temporary audio file
        os.remove(temp_filename)

      # Draw bounding box (optional)
      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show the current frame (optional)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

  # Release camera resources
  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  detect_and_speak()