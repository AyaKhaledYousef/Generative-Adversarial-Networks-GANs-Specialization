#https://pysource.com/2019/10/14/ocr-text-recognition-with-python-and-api-ocr-space/#:~:text=from%20an%20image.-,ocr.,return%20us%20the%20text%20scanned.
#https://ocr.space/
import io
import json
import cv2
import numpy as np
import requests
img = cv2.imread("4.jpg")
height, width, _ = img.shape
print(height)
print(width)
# Cutting image
roi = img[0: height, 0: width]
# Ocr
url_api = "https://api.ocr.space/parse/image"
_, compressedimage = cv2.imencode(".jpg", roi, [1, 90])
file_bytes = io.BytesIO(compressedimage)

result = requests.post(url_api,
              files = {"4.jpg": file_bytes},
              data = {"apikey": "K81074602088957",
                      "language": "ara"})


result = result.content.decode()
result = json.loads(result)
print(result)

parsed_results = result.get("ParsedResults")[0]
text_detected = parsed_results.get("ParsedText")

with open('readme.txt', 'w') as f:
    f.write(text_detected)

print(text_detected)