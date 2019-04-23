import cv2


cap = cv2.VideoCapture('output.avi')

while True:
    ret,img_color = cap.read()

    if ret == False:
        break

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Color", img_color)
    cv2.imshow("Gray", img_gray)

    if cv2.waitKey(1)&0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

