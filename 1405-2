import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model


img_color = cv.imread('test3.jpg', cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)


ret,img_binary = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

kernel = cv.getStructuringElement( cv.MORPH_RECT, ( 5, 5 ) )
img_binary = cv.morphologyEx(img_binary, cv. MORPH_CLOSE, kernel)

cv.imshow('digit', img_binary)
cv.waitKey(0)

contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, 
                        cv.CHAIN_APPROX_SIMPLE)

for contour in contours:

    x, y, w, h = cv.boundingRect(contour)



    length = max(w, h) + 60
    img_digit = np.zeros((length, length, 1),np.uint8)

    new_x,new_y = x-(length - w)//2, y-(length - h)//2


    img_digit = img_binary[new_y:new_y+length, new_x:new_x+length]

    kernel = np.ones((5, 5), np.uint8)
    img_digit = cv.morphologyEx(img_digit, cv.MORPH_DILATE, kernel)

    cv.imshow('digit', img_digit)
    cv.waitKey(0)

    model = load_model('model.h5')

    img_digit = cv.resize(img_digit, (28, 28), interpolation=cv.INTER_AREA)

    img_digit = img_digit / 255.0

    img_input = img_digit.reshape(1, 28, 28, 1)
    predictions = model.predict(img_input)


    number = np.argmax(predictions)
    print(number)

    cv.rectangle(img_color, (x, y), (x+w, y+h), (255, 255, 0), 2)


    location = (x + int(w *0.5), y - 10)
    font = cv.FONT_HERSHEY_COMPLEX  
    fontScale = 1.2
    cv.putText(img_color, str(number), location, font, fontScale, (0,255,0), 2)
    

    cv.imshow('digit', img_digit)
    cv.waitKey(0)
    

cv.imshow('result', img_color)
cv.waitKey(0)
