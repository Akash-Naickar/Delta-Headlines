from google.cloud import vision
import io
import os
import pandas as pd
from tqdm import tqdm

def extract_text_from_images(image_folder, output_file):

    # Initialize Vision client
    client = vision.ImageAnnotatorClient.from_service_account_file("vision_ocr.json")
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    # Prepare data structure for storing results
    results = []
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)
        
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
    
    # Save results to TSV
    pd.DataFrame(results).to_csv(output_file, index=False, sep='\t')
    print(f"Processed {len(results)} images. Results saved to {output_file}")
    return results

# Example usage
if __name__ == "__main__":
    # Process a batch of images
    results = extract_text_from_images(
        image_folder="image_folder",
        output_file="extracted_texts.tsv"
    )