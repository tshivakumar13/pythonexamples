
import cv2
import os
import supervision as sv
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from huggingface_hub import hf_hub_download, login
from ultralytics import YOLO

# Replace 'your_token_here' with your actual Hugging Face token
login(token="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")

model_path = hf_hub_download(
  repo_id="tech4humans/yolov8s-signature-detector",
  filename="yolov8s.pt"
)
model = YOLO(model_path)

# PDF to image conversion
pdf_path = "C:\\Users\\tshiv\\Downloads\\SpecimenSignatureSheet.pdf"  # Change to your PDF file path
output_dir = "C:\\Users\\tshiv\\Downloads\\op"
os.makedirs(output_dir, exist_ok=True)

images = convert_from_path(pdf_path)

for page_num, pil_image in enumerate(images, start=1):
  # Convert PIL image to OpenCV format
  image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

  # Run signature detection model
  results = model(image)
  detections = sv.Detections.from_ultralytics(results[0])

  box_annotator = sv.BoxAnnotator()
  annotated_image = box_annotator.annotate(scene=image, detections=detections)

  cv2.imshow(f"Detections Page {page_num}", annotated_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # Save each detected signature as a separate PNG file
  for i, box in enumerate(detections.xyxy):
    x1, y1, x2, y2 = map(int, box)
    signature_crop = image[y1:y2, x1:x2]

    # Bold/enhance the signature
    kernel = np.ones((1,1), np.uint8)
    bold_signature = cv2.dilate(signature_crop, kernel, iterations=1)
    gray_image = cv2.cvtColor(bold_signature, cv2.COLOR_BGR2GRAY)

    # Resize to fixed size
    desired_size = (400, 200)
    fixed_size = cv2.resize(gray_image, desired_size, interpolation=cv2.INTER_NEAREST_EXACT)
    output_path = os.path.join(output_dir, f"extracted_signature_page{page_num}_{i+1}.png")
    cv2.imwrite(output_path, fixed_size)
    print(f"Saved signature crop: {output_path}")
