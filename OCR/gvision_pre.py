from google.cloud import vision
import io

client = vision.ImageAnnotatorClient.from_service_account_file("vision_ocr.json") # Creating a client variable for GV

image_path = 'samplenews.jpg'

with io.open(image_path,'rb') as image_file: # Reading the image in binary format using io library
    content = image_file.read()

image = vision.Image(content=content) # Creating an image object 
response = client.text_detection(image=image) 
texts = response.text_annotations

print(texts[0].description)
