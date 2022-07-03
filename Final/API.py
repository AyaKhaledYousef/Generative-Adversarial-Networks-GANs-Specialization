from locale import currency
from Smart_Glasses import *
from gpiozero import Button
from signal import pause


print("System is Running .... ")


b17 =Button(17)
b27 =Button(27)
b22 =Button(22)
b6  =Button(6)
b23 =Button(23)


b17.when_pressed  = distance.detect
b27.when_pressed = Face_recognize.recognize
b22.when_pressed  = OCR_Ara.ocr
b6.when_pressed  = OCR_Eng.ocr
#b23.when_pressed  = currency.


pause()