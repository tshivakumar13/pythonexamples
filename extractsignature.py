import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

# Step 1: Extract images from PDF
pdf_path = "input.pdf"
doc = fitz.open(pdf_path)
for page_num in range(len(doc)):
    for img_index, img in enumerate(doc.get_page_images(page_num)):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n < 5:  # this is GRAY or RGB
            pix.save(f"tmp_page{page_num+1}_img{img_index+1}.png")
        pix = None

# Step 2: Detect and crop signature (assumes only signature present)
image = cv2.imread("tmp_page1_img1.png", cv2.IMREAD_GRAYSCALE)
thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
signature_contour = max(contours, key=cv2.contourArea)
x,y,w,h = cv2.boundingRect(signature_contour)
signature = image[y:y+h, x:x+w]

# Step 3: Bold/enhance the signature
kernel = np.ones((3,3),np.uint8)
bold_signature = cv2.dilate(signature, kernel, iterations=1)

# Step 4: Resize to fixed size
desired_size = (400, 200)
fixed_size = cv2.resize(bold_signature, desired_size, interpolation=cv2.INTER_AREA)
output_path = "extracted_signature.png"
cv2.imwrite(output_path, fixed_size)
