from google.cloud import vision
import io
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter  # PIL for image pre-processing
import tempfile
import cv2  # cv for image pre-processing
import numpy as np
import math

def preprocess_image(image_path):

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read {image_path} with OpenCV, falling back to PIL")
            return image_path
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                if abs(angle) < 30:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                
                if abs(median_angle) > 0.5:  
                    (h, w) = gray.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    gray = cv2.warpAffine(gray, M, (w, h), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
                    
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  
            2   
        )

        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        temp_dir = tempfile.gettempdir()
        temp_filename = f"preprocessed_{os.path.basename(image_path)}"
        temp_path = os.path.join(temp_dir, temp_filename)

        cv2.imwrite(temp_path, binary)

        pil_img = Image.open(temp_path)

        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.2)

        pil_img = pil_img.filter(ImageFilter.SHARPEN)

        pil_img.save(temp_path)
        
        return temp_path
    
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return image_path 

def extract_text_from_images(image_folder, output_file, use_preprocessing=True):

    client = vision.ImageAnnotatorClient.from_service_account_file("vision_ocr.json")

    image_files = [f for f in os.listdir(image_folder) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    results = []
    temp_files = []
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)

        if use_preprocessing:
            processed_image_path = preprocess_image(image_path)
            if processed_image_path != image_path:
                temp_files.append(processed_image_path)
        else:
            processed_image_path = image_path

        with io.open(processed_image_path, 'rb') as image_content:
            content = image_content.read()

        image = vision.Image(content=content)
 
        image_context = vision.ImageContext(
            language_hints=['en']  
        )
        
        try:
            doc_response = client.document_text_detection(image=image, image_context=image_context)
 
            if doc_response.error.message or not doc_response.text_annotations:
                response = client.text_detection(image=image, image_context=image_context)
            else:
                response = doc_response
            
            if response.error.message:
                print(f"Error with {image_file}: {response.error.message}")
                extracted_text = ""
            elif response.text_annotations:
                extracted_text = response.text_annotations[0].description

                extracted_text = extracted_text.strip()
 
                extracted_text = '\n'.join(line for line in extracted_text.splitlines() if line.strip())
            else:
                extracted_text = ""

            results.append({
                "filename": image_file,
                "text": extracted_text
            })
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            results.append({
                "filename": image_file,
                "text": ""
            })
    

    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except:
            pass
    

    if output_file.endswith('.tsv'):
        pd.DataFrame(results).to_csv(output_file, index=False, sep='\t')
    else:

        pd.DataFrame(results).to_csv(output_file, index=False)
    
    print(f"Processed {len(results)} images. Results saved to {output_file}")
    return results


if __name__ == "__main__":

    results = extract_text_from_images(
        image_folder="image_folder",
        output_file="extracted_texts.tsv", 
        use_preprocessing=True 
    )
