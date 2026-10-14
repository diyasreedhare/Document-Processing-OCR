import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import easyocr
import pandas as pd
import numpy as np
import re
import os
print(" Loading OCR model (first time takes few seconds)...")
reader = easyocr.Reader(['en'], gpu=False)
def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(" Image not found. Check file path!")

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    
    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)[1]

    return image, thresh

def perform_ocr(processed_image):
    results = reader.readtext(processed_image)

    text_list = []
    for (bbox, text, prob) in results:
        text_list.append(text)

    return text_list, results

def extract_fields(text_list):
    text_data = " ".join(text_list)

    date = re.findall(r'\d{2}[/-]\d{2}[/-]\d{4}', text_data)

    total_match = re.search(r'Total\s*([0-9]+\.[0-9]{2})', text_data, re.IGNORECASE)

    amount = total_match.group(1) if total_match else "Not Found"
    

    invoice_match = re.search(r'(SLM[/\-]?\d{4}[/\-]?\d+)', text_data)

    invoice_no = invoice_match.group(1) if invoice_match else "Not Found"

    store = text_list[0] if text_list else "Unknown"

    return {
        "Store": store,
        "Date": date[0] if date else "Not Found",
        "Amount": amount,
        "Invoice_No": invoice_no
    }

def draw_boxes(image, results):
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox

        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        
        cv2.putText(image, text, top_left,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1)

    return image

def main():

    
    image_path = r"c:\\Users\\diyas\\OneDrive\\Desktop\\ocrinvoice4.png"

    print("\n Reading Image...")
    original, processed = preprocess_image(image_path)

    print(" Performing OCR...")
    text_list, results = perform_ocr(processed)

    print("\n Extracted Text:")
    for t in text_list:
        print(t)

    print("\n Extracting Important Data...")
    data = extract_fields(text_list)

    for key, value in data.items():
        print(f"{key}: {value}")

    
    os.makedirs("output", exist_ok=True)

    df = pd.DataFrame([data])
    df.to_csv("output/result.csv", index=False)

    print("\nSaved to output/result.csv")

    boxed_image = draw_boxes(original, results)

    cv2.imshow("Detected Text", boxed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()