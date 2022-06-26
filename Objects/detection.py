import torch
import gtts
from pydub import AudioSegment
from pydub.playback import play

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# Image
im = 'yolov5/data/images/bus.jpg'

# Inference
results = model(im)
res=results.pandas().xyxy[0]
for object in range(res.shape[0]):
    object_name=res['name'].values[object]
    print(object_name)
    tts = gtts.gTTS(object_name)
    tts.save("object.mp3")
    play(AudioSegment.from_mp3("object.mp3"))


