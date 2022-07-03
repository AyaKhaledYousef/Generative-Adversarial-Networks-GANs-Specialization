import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # default
image='test-images/10-461.jpg'
results = model(image, size=640)  # includes NMS

# Results
print(results.print())  
# results.save()  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)
