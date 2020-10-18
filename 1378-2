import cv2 as cv
import numpy as np
import os



def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects


def removeFaceAra(img, cascade):
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  gray = cv.equalizeHist(gray)
  rects = detect(gray, cascade)

  height,width = img.shape[:2]

  for x1, y1, x2, y2 in rects:
      cv.rectangle(img, (x1-10, 0), (x2+10, height), (0,0,0), -1)

  return img


def make_mask_image(img_bgr):
    
  img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

  #img_h,img_s,img_v = cv.split(img_hsv)

  low = (0, 30, 0)
  high = (15, 255, 255)

  img_mask = cv.inRange(img_hsv, low, high)
  return img_mask

  
def distanceBetweenTwoPoints(start, end):

  x1,y1 = start
  x2,y2 = end
 
  return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))


def calculateAngle(A, B):

  A_norm = np.linalg.norm(A)
  B_norm = np.linalg.norm(B)
  C = np.dot(A,B)

  angle = np.arccos(C/(A_norm*B_norm))*180/np.pi
  return angle


def findMaxArea(contours):
  
  max_contour = None
  max_area = -1


  for contour in contours:
    area = cv.contourArea(contour)

    x,y,w,h = cv.boundingRect(contour)

    if (w*h)*0.4 > area:
        continue

    if w > h:
        continue

    if area > max_area:
      max_area = area
      max_contour = contour
  
  if max_area < 10000:
    max_area = -1

  return max_area, max_contour


def getFingerPosition(max_contour, img_result, debug):
  points1 = []


  # STEP 6-1
  M = cv.moments(max_contour)

  cx = int(M['m10']/M['m00'])
  cy = int(M['m01']/M['m00'])


  max_contour = cv.approxPolyDP(max_contour,0.02*cv.arcLength(max_contour,True),True)
  hull = cv.convexHull(max_contour)

  for point in hull:
    if cy > point[0][1]:
      points1.append(tuple(point[0])) 

  if debug:
    cv.drawContours(img_result, [hull], 0, (0,255,0), 2)
    for point in points1:
      cv.circle(img_result, tuple(point), 15, [ 0, 0, 0], -1)


  # STEP 6-2
  hull = cv.convexHull(max_contour, returnPoints=False)
  defects = cv.convexityDefects(max_contour, hull)

  if defects is None:
    return -1,None

  points2=[]
  for i in range(defects.shape[0]):
    s,e,f,d = defects[i, 0]
    start = tuple(max_contour[s][0])
    end = tuple(max_contour[e][0])
    far = tuple(max_contour[f][0])

    angle = calculateAngle( np.array(start) - np.array(far), np.array(end) - np.array(far))

    if angle < 90:
      if start[1] < cy:
        points2.append(start)
      
      if end[1] < cy:
        points2.append(end)

  if debug:
    cv.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
    for point in points2:
      cv.circle(img_result, tuple(point), 20, [ 0, 255, 0], 5)


  # STEP 6-3
  points = points1 + points2
  points = list(set(points))


  # STEP 6-4
  new_points = []
  for p0 in points:
    
    i = -1
    for index,c0 in enumerate(max_contour):
      c0 = tuple(c0[0])

      if p0 == c0 or distanceBetweenTwoPoints(p0,c0)<20:
        i = index
        break

    if i >= 0:
      pre = i - 1
      if pre < 0:
        pre = max_contour[len(max_contour)-1][0]
      else:
        pre = max_contour[i-1][0]
      
      next = i + 1
      if next > len(max_contour)-1:
        next = max_contour[0][0]
      else:
        next = max_contour[i+1][0]


      if isinstance(pre, np.ndarray):
            pre = tuple(pre.tolist())
      if isinstance(next, np.ndarray):
        next = tuple(next.tolist())

        
      angle = calculateAngle( np.array(pre) - np.array(p0), np.array(next) - np.array(p0))     

      if angle < 90:
        new_points.append(p0)
  
  return 1,new_points


def process(img_bgr, debug):

  img_result = img_bgr.copy()

  # STEP 1
  img_bgr = removeFaceAra(img_bgr, cascade)


  # STEP 2
  img_binary = make_mask_image(img_bgr)


  # STEP 3
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
  img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
  cv.imshow("Binary", img_binary)


  # STEP 4
  contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  if debug:
    for cnt in contours:
      cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)  
    

  # STEP 5
  max_area, max_contour = findMaxArea(contours)  

  if max_area == -1:
    return img_result

  if debug:
    cv.drawContours(img_result, [max_contour], 0, (0, 0, 255), 3)  


  # STEP 6
  ret,points = getFingerPosition(max_contour, img_result, debug)
  

  # STEP 7
  if ret > 0 and len(points) > 0:  
    for point in points:
      cv.circle(img_result, point, 20, [ 255, 0, 255], 5)

  return img_result



current_file_path = os.path.dirname(os.path.realpath(__file__))
cascade = cv.CascadeClassifier(cv.samples.findFile(current_file_path + "\haarcascade_frontalface_alt.xml"))


# cap = cv.VideoCapture('test.avi')

cap = cv.VideoCapture(0)

while True:

  ret,img_bgr = cap.read()
  
  if ret == False:
    break

  img_result = process(img_bgr, debug=False)
  
  key = cv.waitKey(1) 
  if key== 27:
      break


  cv.imshow("Result", img_result)


cap.release()
cv.destroyAllWindows()
