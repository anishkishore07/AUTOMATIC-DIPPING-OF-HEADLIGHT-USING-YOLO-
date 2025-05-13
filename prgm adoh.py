import cv2
import numpy as np
from ultralytics import YOLO
import time

# YOLO model setup
model = YOLO("yolov8n.pt")
names = model.model.names

# Video source and output paths
video_path = "C:\\Users\\anish\\OneDrive\\Desktop\\vs prgm\\src_vedio .mp4"
output_path = "C:\\Users\\anish\\OneDrive\\Desktop\\vs prgm\\combined_output"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# Video dimensions and output setup
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)
result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Headlight detection thresholds
DETECTION_THRESHOLD = 10
NO_DETECTION_THRESHOLD = 15
CONFIDENCE_THRESHOLD = 0.5

# Variables to manage beam states and vehicle detection
low_beam = False
frame_since_detection = 0
no_vehicle_frames = 0

def detect_headlights(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    headlights = [
        cv2.boundingRect(c) for c in contours if 50 < cv2.contourArea(c) < 500
    ]
    return headlights

def check_night_time():
    return True  # For demo, always True (night)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Check if it's night time
        night_time = check_night_time()

        # YOLO object detection
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.xyxy.cpu() if results[0].boxes.xyxy is not None else []
        confidences = (
            results[0].boxes.conf.cpu() if results[0].boxes.conf is not None else []
        )

        detect_oncoming = False
        headlights_detected = detect_headlights(frame) if night_time else []

        # Vehicle detection logic
        if night_time and headlights_detected:
            detect_oncoming = True

        if results[0].boxes.id is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, cls, track_id, confidence in zip(boxes, clss, track_ids, confidences):
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                # Check if the detected object is a vehicle
                if cls in [2, 3, 5, 7]:  # Classes: car, motorcycle, bus, truck
                    detect_oncoming = True

        # Headlight control logic
        if detect_oncoming:
            frame_since_detection += 1
            no_vehicle_frames = 0
            if frame_since_detection >= DETECTION_THRESHOLD and not low_beam:
                print("Oncoming vehicle detected: Switching to low beam.")
                low_beam = True
        else:
            no_vehicle_frames += 1
            frame_since_detection = 0
            if no_vehicle_frames >= NO_DETECTION_THRESHOLD and low_beam:
                print("No vehicles detected: Switching back to high beam.")
                low_beam = False

        result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

# Release resources
result.release()
cap.release()
cv2.destroyAllWindows()
print(f"Processing completed. Output saved to {output_path}")
