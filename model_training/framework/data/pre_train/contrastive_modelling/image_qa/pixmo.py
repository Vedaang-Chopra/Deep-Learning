
import requests
from PIL import Image
from io import BytesIO
import numpy as np


def prepare_dataset(raw_data, samples):
    img_dataset = []
    for i in range(0,len(samples)):
        url = samples[i]["image_url"]
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image = np.array(image.resize((224, 224), resample=Image.Resampling.BICUBIC)).astype(np.float32) / 255.0
            img_dataset.append(image)
        except Exception as err:
            print(f"Image Not Found:{samples[i]['image_url']}")
        
    return np.array(img_dataset)
