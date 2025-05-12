from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

url = 'images0028.png'
image = Image.open(url)
print(image)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

print()