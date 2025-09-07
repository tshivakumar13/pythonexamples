
# Import required libraries
import cv2  # OpenCV for image processing
import os   # For file and directory operations
import supervision as sv  # For annotation and detection utilities
import fitz  # PyMuPDF (not used in this script, but often for PDF handling)
import numpy as np  # For array and matrix operations
from PIL import Image  # For image format conversion

# Import Hugging Face and YOLO model utilities
from huggingface_hub import hf_hub_download, login
from ultralytics import YOLO

# Authenticate with Hugging Face using your token
login(token="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")

# Download the YOLO signature detection model from Hugging Face
model_path = hf_hub_download(
  repo_id="tech4humans/yolov8s-signature-detector",
  filename="yolov8s.pt"
)
model = YOLO(model_path)

# Path to the input document image (PNG or JPG)
image_path = "C:\\Users\\tshiv\\Downloads\\SpecimenSignatureSheet.png"
# Alternative image path (uncomment to use)
# image_path = "C:\Users\tshiv\Downloads\third.jpg"

# Read the image using OpenCV
image = cv2.imread(image_path)

# Run the YOLO model to detect signatures in the image
results = model(image_path)

# Convert YOLO results to supervision Detections object
detections = sv.Detections.from_ultralytics(results[0])

# Annotate the detected signatures on the image
box_annotator = sv.BoxAnnotator()
annotated_image = box_annotator.annotate(scene=image, detections=detections)

# Display the annotated image with detected signatures
cv2.imshow("Detections", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Prepare output directory for saving extracted signatures
output_dir = "C:\\Users\\tshiv\\Downloads\\op"
os.makedirs(output_dir, exist_ok=True)

# Loop through each detected signature bounding box
for i, box in enumerate(detections.xyxy):
  x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
  signature_crop = image[y1:y2, x1:x2]  # Crop the signature from the image

  # Optional: Save the raw crop (uncomment if needed)
  # crop_filename = os.path.join(output_dir, f"signature_{i+1}.png")
  # cv2.imwrite(crop_filename, signature_crop)


  # Step 3: Bold/enhance the signature using dilation
  kernel = np.ones((1,1), np.uint8)
  bold_signature = cv2.dilate(signature_crop, kernel, iterations=1)

# Sharpen the image
  sharpen_kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
  sharpened = cv2.filter2D(bold_signature, -1, sharpen_kernel)
# Convert the signature to grayscale
  gray_image = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)


  # Remove borders from the cropped image using contour detection
  # Threshold to get binary image
  _, thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if contours:
    x, y, w, h = cv2.boundingRect(contours[0])
    borderless_crop = gray_image[y:y+h, x:x+w]
  else:
    borderless_crop = gray_image

  # Step 4: Resize the signature to a fixed size for consistency
  desired_size = (400, 200)
  fixed_size = cv2.resize(borderless_crop, desired_size, interpolation=cv2.INTER_NEAREST_EXACT)
  output_path = os.path.join(output_dir, f"extracted_signature_{i+1}.png")
  cv2.imwrite(output_path, fixed_size)

  print(f"Saved signature crop: {output_path}")

# Final cleanup: Close any OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()
