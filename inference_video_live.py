import torch
from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('weights/epochs_50/best_biscuit.pt')

# Open the camera
cap = cv2.VideoCapture(0)  # 0 is typically the default camera. Change if you have multiple cameras.

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for the output video
output_path = 'output/real_time_output.mov'  # Change this to the desired output path
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Process and draw bounding boxes on the frame
    for result in results:
        boxes = result.boxes  # Access the boxes attribute
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Extract bounding box coordinates
            conf = box.conf[0].item()  # Extract confidence score
            cls = int(box.cls[0].item())  # Extract class and convert to integer index

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Put text (confidence score)
            cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Real-Time YOLOv8 Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()