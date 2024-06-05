import torch
from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('weights/epochs_50/best_biscuit.pt')

# Load a video
video_path = 'inference_samples/parleg.mov'  # Change this to the path of your input video
output_path = 'output/video1.mov'  # Change this to the desired output path

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for the output video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the video writer
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Process and draw bounding boxes on the frame
    for result in results:
        boxes = result.boxes  # Access the boxes attribute
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
            conf = box.conf[0]  # Extract confidence score
            cls = box.cls[0]  # Extract class

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Put text (class and confidence score)
            cv2.putText(frame, f'{cls} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release the video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()