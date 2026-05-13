import torch
import requests
from PIL import Image
from io import BytesIO

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

url = "https://ultralytics.com/images/zidane.jpg"
img = Image.open(BytesIO(requests.get(url).content))

results = model(img)

df = results.pandas().xyxy[0]
df = df[df['confidence'] >= 0.5]

print("\nFiltered Detections:\n")
print(df)

results.show()