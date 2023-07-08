import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import argparse
import imutils

# set tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\pcteste1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

parser = argparse.ArgumentParser(description='Answer sheets')
parser.add_argument('-a', '--answerSheet', help='The official answer sheet', required=False, default='images/Gabarito_ADA.png')
parser.add_argument('-b', '--answers', help='The submitted answers', required=False, default='images/Respostas_ADA.png')

args = vars(parser.parse_args()) # vars returns the __dict__ attribute for a class

answer_sheet = cv2.imread(args['answerSheet'], cv2.IMREAD_GRAYSCALE)
#answers = cv2.imread(args['answers'])

assert answer_sheet is not None, 'Image not loaded'

#answer_sheet = cv2.GaussianBlur(answer_sheet, ksize=(3,3), sigmaX=0)
#answer_sheet = cv2.bilateralFilter(answer_sheet, 13, 60, 40)
thresh_img = cv2.threshold(answer_sheet, 127, 255, cv2.THRESH_BINARY)[1]

contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))

for contour in contours:
    # Get the bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)
     
    middle = (x + (x + w)) // 2
    end = (x + w) - middle

    cropped_image = thresh_img[y:y+h, middle+10:middle+50]

    #cropped_image = cv2.dilate(cropped_image, None)
    cropped_image = cv2.resize(cropped_image, None, fx=2, fy=2)

    cv2.imshow('img', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    text = pytesseract.image_to_string(cropped_image, config="--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    print(text)

#cv2.imshow('img', cropped_image)
#cv2.waitKey(0)
cv2.destroyAllWindows()

'''
#d = pytesseract.image_to_data(answer_sheet, config="--psm 4", output_type=Output.DICT)

#print(d)

text = pytesseract.image_to_string(answer_sheet, config="--psm 4")
print(text)

answer_sheet_dictionary = {}

# remove empty spaces and the dots from numbers
letters = []
for letter in d['text']:
    if letter.strip():
        letters.append(letter.replace('.', ''))


for i in range(0, len(letters)-1, 2):
    answer_sheet_dictionary[letters[i]] = letters[i+1]

for key, value in  answer_sheet_dictionary.items():
    print(key, value)'''

#print(dict(pairwise(letters)))

'''h, w, c = answer_sheet.shape
boxes = pytesseract.image_to_boxes(answer_sheet) 
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(answer_sheet, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
