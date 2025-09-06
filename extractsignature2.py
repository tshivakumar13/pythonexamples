import cv2
import os
import supervision as sv
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from huggingface_hub import login

# Replace 'your_token_here' with your actual Hugging Face token
login(token="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")

model_path = hf_hub_download(
  repo_id="tech4humans/yolov8s-signature-detector", 
  filename="yolov8s.pt"
)

model = YOLO(model_path)

image_path = "C:\\Users\\tshiv\\Downloads\\third.jpg"
image = cv2.imread(image_path)

results = model(image_path)

detections = sv.Detections.from_ultralytics(results[0])

box_annotator = sv.BoxAnnotator()
annotated_image = box_annotator.annotate(scene=image, detections=detections)

cv2.imshow("Detections", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save each detected signature as a separate PNG file
output_dir = "C:\\Users\\tshiv\\Downloads\\op"
os.makedirs(output_dir, exist_ok=True)

for i, box in enumerate(detections.xyxy):
    x1, y1, x2, y2 = map(int, box)
    signature_crop = image[y1:y2, x1:x2]
    #
    # crop_filename = os.path.join(output_dir, f"signature_{i+1}.png")
    #cv2.imwrite(crop_filename, signature_crop)

    # Step 3: Bold/enhance the signature
    kernel = np.ones((3,3),np.uint8)
    bold_signature = cv2.dilate(signature_crop, kernel, iterations=1)

    # Step 4: Resize to fixed size
    desired_size = (400, 200)
    fixed_size = cv2.resize(bold_signature, desired_size, interpolation=cv2.INTER_NEAREST_EXACT)
    output_path = os.path.join(output_dir, f"extracted_signature_{i+1}.png")
    cv2.imwrite(output_path, fixed_size)

    print(f"Saved signature crop: {output_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()
