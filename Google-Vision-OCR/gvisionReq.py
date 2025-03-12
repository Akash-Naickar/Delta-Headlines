from google.cloud import vision
import io
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
import tempfile
import cv2
import numpy as np
import math

def preprocess_image(image_path):
    """
    Preprocess image to improve OCR accuracy using advanced techniques
    
    Args:
        image_path (str): Path to the original image
    
    Returns:
        str: Path to the preprocessed image
    """
    try:
        # Read image with OpenCV for advanced preprocessing
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read {image_path} with OpenCV, falling back to PIL")
            return image_path
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply deskewing - detect and correct skew angle
        # First find edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Skip vertical lines
                if x2 - x1 == 0:
                    continue
                angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                # Only consider angles close to horizontal
                if abs(angle) < 30:
                    angles.append(angle)
            
            if angles:
                # Calculate median angle
                median_angle = np.median(angles)
                
                # Rotate to correct skew if needed
                if abs(median_angle) > 0.5:  # Only correct if skew is noticeable
                    (h, w) = gray.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    gray = cv2.warpAffine(gray, M, (w, h), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
        
        # Apply adaptive thresholding for binarization
        # This works better than global thresholding for varied lighting
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Noise removal using morphological operations
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Create temp file for preprocessed image
        temp_dir = tempfile.gettempdir()
        temp_filename = f"preprocessed_{os.path.basename(image_path)}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Save preprocessed image
        cv2.imwrite(temp_path, binary)
        
        # Further enhance with PIL if needed
        pil_img = Image.open(temp_path)
        
        # Increase contrast further if needed
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.2)
        
        # Apply slight sharpening
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        
        # Save enhanced image
        pil_img.save(temp_path)
        
        return temp_path
    
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return image_path  # Return original path if preprocessing fails

def extract_text_from_images(image_folder, output_file, use_preprocessing=True):
    """
    Process images from a folder, extract text using Google Cloud Vision,
    and store the results in a CSV/TSV file.
    
    Args:
        image_folder (str): Path to folder containing images
        output_file (str): Path to save the extracted text data
        use_preprocessing (bool): Whether to apply image preprocessing
    """
    # Initialize Vision client
    client = vision.ImageAnnotatorClient.from_service_account_file("vision_ocr.json")
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    # Prepare data structure for storing results
    results = []
    temp_files = []
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)
        
        # Preprocess the image if enabled
        if use_preprocessing:
            processed_image_path = preprocess_image(image_path)
            if processed_image_path != image_path:
                temp_files.append(processed_image_path)
        else:
            processed_image_path = image_path
        
        # Read the image
        with io.open(processed_image_path, 'rb') as image_content:
            content = image_content.read()
        
        # Create image object
        image = vision.Image(content=content)
        
        # Create image context with language hints
        image_context = vision.ImageContext(
            language_hints=['en']  # Add more languages if needed
        )
        
        try:
            # Try document text detection first (better for dense text)
            doc_response = client.document_text_detection(image=image, image_context=image_context)
            
            # Fall back to regular text detection if document text fails
            if doc_response.error.message or not doc_response.text_annotations:
                response = client.text_detection(image=image, image_context=image_context)
            else:
                response = doc_response
            
            if response.error.message:
                print(f"Error with {image_file}: {response.error.message}")
                extracted_text = ""
            elif response.text_annotations:
                extracted_text = response.text_annotations[0].description
                
                # Simple post-processing
                extracted_text = extracted_text.strip()
                # Remove excessive newlines
                extracted_text = '\n'.join(line for line in extracted_text.splitlines() if line.strip())
            else:
                extracted_text = ""
                
            # Store the result
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
    
    # Clean up temporary files
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except:
            pass
    
    # Save results based on file extension
    if output_file.endswith('.tsv'):
        pd.DataFrame(results).to_csv(output_file, index=False, sep='\t')
    else:
        # Default to CSV
        pd.DataFrame(results).to_csv(output_file, index=False)
    
    print(f"Processed {len(results)} images. Results saved to {output_file}")
    return results

# Example usage
if __name__ == "__main__":
    # Process a batch of images
    results = extract_text_from_images(
        image_folder="image_folder",
        output_file="extracted_texts.tsv",  # Use .tsv for T5 summarizer compatibility
        use_preprocessing=True  # Enable image preprocessing
    )