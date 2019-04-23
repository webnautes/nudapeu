import cv2


cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

while True:
    ret,img_color = cap.read()

    if ret == False:
        continue

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Color", img_color)
    cv2.imshow("Gray", img_gray)

    writer.write(img_color)

    if cv2.waitKey(1)&0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

