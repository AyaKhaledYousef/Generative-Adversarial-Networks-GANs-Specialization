import torch
import gtts
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import cv2


def cam():
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        img_name = "Glasses2/Objects/img.png"
        cv2.imshow('frame', frame)
        k=cv2.waitKey(1)

        if k%256 == 27:
            break
        elif k%256 == 32:
            print("take image")
            cv2.imwrite(img_name, frame)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
results = model('Glasses2/Objects/img.png')
res=results.pandas().xyxy[0]
for object in range(res.shape[0]):
    object_name=res['name'].values[object]
    print(object_name)

    tts = gtts.gTTS(object_name)
    tts.save("Glasses2/Objects/object.mp3")
    play(AudioSegment.from_mp3("Glasses2/Objects/object.mp3"))


