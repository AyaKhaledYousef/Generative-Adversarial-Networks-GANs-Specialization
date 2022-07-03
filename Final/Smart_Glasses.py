    
# Imports
import cv2
import pytesseract
from pytesseract import Output
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
#import face_recognition
import pickle
import matplotlib.pyplot as plt
import requests
import io
import json
import torch
from fer import FER

# =============================================================================
# 
# =============================================================================
class detect_distance_objects:
    def __init__(self):

        self.Conf_threshold = 0.4
        self.NMS_threshold = 0.4
        self.font = cv2.FONT_HERSHEY_SIMPLEX 
        self.COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
                (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        self.class_name = []
        self.net = cv2.dnn.readNet('Detection/yolov4-tiny.weights', 'Detection/yolov4-tiny.cfg')
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True) 

        with open('Detection/classes.txt', 'r') as f:
            self.class_name = [cname.strip() for cname in f.readlines()]

        self.KNOWN_DISTANCE =45
        self.OBJECT_WIDTH = 16
        
           
    def capture_img(self):

        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        img_name = "Detection/img.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        cam.release()

    def object_detector(self,image):
        data_list=[]
        classes, scores, boxes = self.model.detect(image, self.Conf_threshold, self.NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            color = self.COLORS[int(classid) % len(self.COLORS)]
            label = "%s : %f" % (self.class_name[classid], score)
            cv2.rectangle(image, box, color, 2)
            cv2.putText(image, label, (box[0], box[1]-10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        
            data_list.append([str(self.class_name[classid]),box[2],(box[0],box[1]-2)])
            
        return data_list
    def focal_length_finder(self,measured_distance, real_width, width_image):
        # finding focal length
        focal_length = (width_image* measured_distance)/real_width
        return focal_length
    def distance_finder(self,focal_Length, real_object_width,width_in_frame):
        distance= (real_object_width * focal_Length)/width_in_frame 
        return distance 
    def capture_img2(self):
        cam = cv2.VideoCapture(0)
        ret, self.frame = cam.read()
        img_name = "Detection/opencv_frame.png"
        cv2.imwrite(img_name, self.frame)
        print("{} written!".format(img_name))

        cam.release()

    def calculate_distance(self):

        self.ref_object = cv2.imread('Detection/img.png')
        self.object_data = self.object_detector(self.ref_object)
        self.object_width_in_rf = self.object_data[0][1]
        self.focal_object = self.focal_length_finder (self.KNOWN_DISTANCE ,self.OBJECT_WIDTH,self.object_width_in_rf)
        
        return self.focal_object

    def detect (self):
        self.capture_img()
        self.capture_img2()
        data = pickle.loads(open('Detection/encodings.pickle', "rb").read())
        img=cv2.imread("Detection/opencv_frame.png")
        data1 = self.object_detector(img)
        if data1 == []:
            text='No object detector '
            print(text)
            myobj2 = gTTS(text, lang='en', slow=False)
            myobj2.save("Detection/object_name.mp3")
            play(AudioSegment.from_mp3("Detection/object_name.mp3"))
        else:
            for d in data1 :
            
                print(d[0])
                myobj = gTTS(text=d[0], lang='en', slow=False)
                myobj.save("Detection/object_name.mp3")
                play(AudioSegment.from_mp3("Detection/object_name.mp3"))
                distance = self.distance_finder(self.calculate_distance(),self.OBJECT_WIDTH,d[1])
                distance=distance *2.54
                print('Distance = ' ,distance)
                x,y = d[2]
                myobj2 = gTTS(text=f'distance = : {round(distance,1)}cm', lang='en', slow=False)
                myobj2.save("Detection/object_distance.mp3")
                
                play(AudioSegment.from_mp3("Detection/object_distance.mp3"))
                    
        return cv2.imwrite('result.png',img)            

class face:

    def __init__(self):

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('Face/trainer/trainer.yml')
        self.cascadePath = "Face/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath);
        self.names = ['Hadeer','Aya','Huda','Mohamed','Abdelrahman','Dr-Nadia'] 

    def recognize(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)

        ret, img =cam.read()
        img_name = "Face/img.png"
        cv2.imwrite(img_name, img)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale( gray,scaleFactor = 1.2,minNeighbors = 5,minSize = (int(minW), int(minH)))
        
        for(x,y,w,h) in faces:
            id=0
            id, confidence = self.recognizer.predict(gray[y:y+h,x:x+w])
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = self.names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            text="The person name is " + id
            print(text)
            myobj2 = gTTS(text, lang='en', slow=False)
            myobj2.save("Face/Face_name.mp3")
            play(AudioSegment.from_mp3("Face/Face_name.mp3"))
 

        text="Uknown"
        print(text)
        myobj2 = gTTS(text, lang='en', slow=False)
        myobj2.save("Face/Face_name.mp3")
        play(AudioSegment.from_mp3("Face/Face_name.mp3"))
 
        cam.release()

class English_txt:
    def __init__(Self):
        pass
        
    def take_img(self):
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        cv2.imwrite('ocr/img.jpg', frame)
        print("{} written!".format('ocr/img.jpg'))
        cam.release()
        
    def ocr(self):
        self.take_img()
        img=cv2.imread('ocr/img.jpg')
        _,compressedimage = cv2.imencode(".jpg", img)
        file_bytes = io.BytesIO(compressedimage)

        file_bytes = io.BytesIO(compressedimage)
        url_api = "https://api.ocr.space/parse/image"

        result = requests.post(url_api,
              files = {"ocr/img.jpg": file_bytes},
              data = {"apikey": "K81074602088957",
                      "language": "eng"})
        result = result.content.decode()
        result = json.loads(result)

        parsed_results = result.get("ParsedResults")[0]
        text_detected = parsed_results.get("ParsedText")

        with open('ocr/readme.txt', 'w') as f:
            f.write(text_detected)

        file1 = open("ocr/readme.txt","r")
        print(file1)
        print(file1.readlines())


class Arabic_txt:
    def __init__(Self):
        pass
        
    def take_img(self):
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        cv2.imwrite('ocr/img-ara.jpg', frame)
        print("{} written!".format('ocr/img-ara.jpg'))
        cam.release()
        
    def ocr(self):
        self.take_img()
        img=cv2.imread('ocr/img-ara.jpg')
        _,compressedimage = cv2.imencode(".jpg", img)
        file_bytes = io.BytesIO(compressedimage)

        file_bytes = io.BytesIO(compressedimage)
        url_api = "https://api.ocr.space/parse/image"

        result = requests.post(url_api,
              files = {"ocr/img-ara.jpg": file_bytes},
              data = {"apikey": "K81074602088957",
                      "language": "ara"})
        result = result.content.decode()
        result = json.loads(result)

        parsed_results = result.get("ParsedResults")[0]
        text_detected = parsed_results.get("ParsedText")

        with open('ocr/readme-ara.txt', 'w') as f:
            f.write(text_detected)

        file1 = open("ocr/readme-ara.txt","r")
        print(file1)
        print(file1.readlines())



class Emotion:

    def __init__(self):
        self.detector = FER()

    def take_img(self):
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        img_name = "Emotion/img.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        cam.release()

    def emotion_detection(self):
        print("GGG")
        self.take_img()
        img = cv2.imread("Emotion/img.png")
        self.detector.detect_emotions(img)
        emotion, score = self.detector.top_emotion(img)
        print(emotion)
        print(score)
        if emotion == None:

            text= 'Can not detect emotion ' 
            print(text)
            myobj2 = gTTS(text, lang='en', slow=False)
            myobj2.save("Emotion/emotion.mp3")
            play(AudioSegment.from_mp3("Emotion/emotion.mp3"))
        else:
            if score > 0.5:
                print (emotion)
                myobj2 = gTTS(emotion, lang='en', slow=False)
                myobj2.save("Emotion/emotion.mp3")
                play(AudioSegment.from_mp3("Emotion/emotion.mp3"))
            else:
                text= 'The Emotion is not clear ' 
                print(text)
                myobj2 = gTTS(text, lang='en', slow=False)
                myobj2.save("Emotion/emotion.mp3")
                play(AudioSegment.from_mp3("Emotion/emotion.mp3"))

                


class Mask:
    pass
class place:
    pass
class Currency:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # default
        self.image='currency/img.jpg'
        self.results = self.model(self.image, size=640)  # includes NMS
    def take_img(self):
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        img_name = "Detection/img.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        cam.release()
    def currency(self):
        self.take_img()
        print(self.results.print())  
        print(self.results.pandas().xyxy[0])  # img1 predictions (pandas)





        

        


# Distance:
# distance=detect_distance_objects()
# distance.detect()

# reco=face()
# reco.recognize()

# eng=English_txt()
# eng.ocr()

# ara=Arabic_txt()
# ara.ocr()

ee=Emotion()
ee.emotion_detection()