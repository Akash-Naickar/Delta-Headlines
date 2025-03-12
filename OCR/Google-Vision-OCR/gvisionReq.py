from google.cloud import vision
import io
import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def process_images_batch(image_folder, output_file, metadata_file=None):
    """
    Process multiple images from a folder, extract text using Google Cloud Vision,
    and store the results with appropriate labels.
    
    Args:
        image_folder (str): Path to folder containing images
        output_file (str): Path to save the extracted text data
        metadata_file (str, optional): Path to JSON file with additional metadata for images
    """
    # Initialize Vision client
    client = vision.ImageAnnotatorClient.from_service_account_file("vision_ocr.json")
    
    # Load metadata if provided
    metadata = {}
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    # Prepare data structure for storing results
    results = []
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)
        
        # Get image metadata or create basic metadata
        image_id = os.path.splitext(image_file)[0]
        image_meta = metadata.get(image_id, {})
        
        # Add basic metadata if not present
        if not image_meta:
            image_meta = {
                "filename": image_file,
                "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "category": "uncategorized"
            }
        
        # Read the image
        with io.open(image_path, 'rb') as image_content:
            content = image_content.read()
        
        # Create image object and get text
        image = vision.Image(content=content)
        
        try:
            response = client.text_detection(image=image)
            
            if response.error.message:
                print(f"Error with {image_file}: {response.error.message}")
                extracted_text = ""
            elif response.text_annotations:
                extracted_text = response.text_annotations[0].description
            else:
                extracted_text = ""
                
            # Store the result with metadata
            result = {
                "image_id": image_id,
                "filename": image_file,
                "text": extracted_text,
                "text_length": len(extracted_text),
                "processing_status": "success" if extracted_text else "no_text_found"
            }
            
            # Add all metadata
            result.update(image_meta)
            
            # Append to results
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            results.append({
                "image_id": image_id,
                "filename": image_file,
                "text": "",
                "processing_status": "error",
                "error_message": str(e)
            })
    
    # Save results based on file extension
    if output_file.endswith('.csv'):
        pd.DataFrame(results).to_csv(output_file, index=False)
    elif output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    elif output_file.endswith('.pkl'):
        pd.DataFrame(results).to_pickle(output_file)
    else:
        # Default to CSV
        pd.DataFrame(results).to_csv(output_file, index=False)
    
    print(f"Processed {len(results)} images. Results saved to {output_file}")
    return results

# Example usage
if __name__ == "__main__":
    # Process a batch of images
    results = process_images_batch(
        image_folder="C:/Users/Akash KG/Google-Vision-OCR/image_folder",
        output_file="extracted_texts.csv",
        metadata_file="image_metadata.json"  # Optional: JSON with additional labels/metadata
    )
    
    # Example of how to load and use the saved data for summarization
    df = pd.read_csv("extracted_texts.csv")
    
    # Group by a label (e.g., category) and summarize
    for category, group in df.groupby("category"):
        print(f"\nCategory: {category}")
        print(f"Number of documents: {len(group)}")
        print(f"Total text length: {group['text_length'].sum()} characters")
        # Here you would add your summarization logic