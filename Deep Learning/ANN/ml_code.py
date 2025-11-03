# import sys
# print("Hello "+ sys.argv[1])

import sys
from pytesseract import Output
import pytesseract
import cv2
import numpy as np
import pyttsx3
import sys

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

#########################################################################################################
#########################################################################################################
# image_name = sys.argv[1]
image_name = 'D:\Artificial Intelligence\Code Repository\Deep_Learning_Code_Repository\demo.jpeg'
speed = 200
# output_filename = sys.argv[1]+".mp3"
output_filename = 'test.mp3'
# image_name = "public/" + image_name
# output_filename = "Audio/" + output_filename
#########################################################################################################
#########################################################################################################


rgb = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

write_text = ""


min_conf = 90
for i in range(0, len(results["text"])):
    # extract the bounding box coordinates of the text region from
    # the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]
    # extract the OCR text itself along with the confidence of the
    # text localization
    text = results["text"][i]
    conf = int(results["conf"][i])
    if conf > min_conf:
        # display the confidence and text to our terminal
#         print("Confidence: {}".format(conf))
#         print("Text: {}".format(text))
#         print("")
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw a bounding box around the text along
        # with the text itself
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        write_text += text+' '
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)






chapter_name = write_text[10:44]
chapter_no = write_text[:9]
text_to_write = write_text[45:]
text_to_write = chapter_no+'. '+chapter_name+'. '+text_to_write[:3]+'. '+text_to_write[4:]
# text_to_write = write_text

# print(text_to_write)

engine = pyttsx3.init()
# engine.say(write_text)
# engine.runAndWait()


rate = engine.getProperty('rate')   # getting details of current speaking rate
engine.setProperty('rate', speed)     # setting up new voice rate
# print (rate)                        #printing current voice rate


volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
# print (volume)                          #printing current volume level
engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1


voices = engine.getProperty('voices')       #getting details of current voice
#engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female


engine.save_to_file(text_to_write, output_filename)
engine.runAndWait()
engine.stop()
print(output_filename)