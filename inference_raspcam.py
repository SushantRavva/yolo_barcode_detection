import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from picamera2 import Picamera2
from PIL import Image

# Load YOLOv5 model (you can replace 'yolov5s' with your specific model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize Picamera2
picam2 = Picamera2()
picam2.start()

# Define a transform to convert images to the format expected by the model
transform = transforms.Compose([
    transforms.ToTensor()
])

def detect_objects(frame):
    # Convert frame to PIL image
    img = Image.fromarray(frame)
    # Apply transforms
    img = transform(img)
    # Add batch dimension
    img = img.unsqueeze(0)

    # Perform detection
    results = model(img)

    # Process results
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        label = model.names[int(cls)]
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        frame = cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

try:
    while True:
        # Capture image
        frame = picam2.capture_array()
        # Detect objects
        frame = detect_objects(frame)
        # Display the frame
        cv2.imshow('YOLOv5 Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()