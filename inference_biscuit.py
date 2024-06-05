import torch
from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

# Load an image
image_path = 'images1.png'  # Change this to the path of your image
img = cv2.imread(image_path)

results = model(img)


# Process and draw bounding boxes on the image
for result in results:
    boxes = result.boxes  # Access the boxes attribute
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
        conf = box.conf[0]  # Extract confidence score
        cls = box.cls[0]  # Extract class

        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Put text (class and confidence score)
        cv2.putText(img, f'{cls} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save and display the resulting image
output_path = '/content/sample_data/output.png'  # Change this to the desired output path
cv2.imwrite(output_path, img)
cv2.imshow('YOLOv8 Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()