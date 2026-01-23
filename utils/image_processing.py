import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = Image.open(image)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def resize_image(image, max_width=800):
    if not isinstance(image, Image.Image):
        img = Image.open(image)
    else:
        img = image
    
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height))
    
    return img
 
