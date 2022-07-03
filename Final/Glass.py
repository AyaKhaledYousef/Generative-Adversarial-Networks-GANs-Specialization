import cv2
import gtts
from pydub import AudioSegment
from pydub.playback import play
#import face_recognition
import pickle
import matplotlib.pyplot as plt
import torch

class Object:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.KNOWN_DISTANCE =45
        self.OBJECT_WIDTH = 16



    def caputer_img (self):
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            cv2.imshow("Take image", frame)
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "1-object/img.png"
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                break
        
        cam.release()
        cv2.destroyAllWindows()

    def focal_length_finder(self,measured_distance, real_width, width_image):
        # finding focal length
        focal_length = (width_image* measured_distance)/real_width
        return focal_length
    def distance_finder(self,focal_Length, real_object_width,width_in_frame):
        distance= (real_object_width * focal_Length)/width_in_frame 
        return distance 

    def calculate_distance(self):

        self.ref_object = cv2.imread('1-object/img.png')
        self.object_data = self.object_detector(self.ref_object)
        self.object_width_in_rf = self.object_data[0][1]
        self.focal_object = self.focal_length_finder (self.KNOWN_DISTANCE ,self.OBJECT_WIDTH,self.object_width_in_rf)
        
        return self.focal_object


    def predict(self):
        self.caputer_img()
        results = self.model('1-object/img.png')
        res=results.pandas().xyxy[0]
        data_list=[]
        if res['name'].empty :
            text="No Object Detected ..."
            print(text)    
            tts = gtts.gTTS(text)
            tts.save("1-object/object.mp3")
            play(AudioSegment.from_mp3("1-object/object.mp3"))
        else:

            for object in range(res.shape[0]):
                object_name=res['name'].values[object]
                print(object_name)
                data_list.append(object_name)
                print(data_list)
                tts = gtts.gTTS(object_name)
                tts.save("1-object/object.mp3")
                play(AudioSegment.from_mp3("1-object/object.mp3"))
                distance = self.distance_finder(self.calculate_distance(),self.OBJECT_WIDTH,data_list[-1])
                distance=distance *2.54
                print('Distance = ' ,distance)



distance=Object()
distance.predict()


