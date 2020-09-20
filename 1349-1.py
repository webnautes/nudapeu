import cv2
import numpy as np


img_color = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
img_original = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

img_template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)

original_h, original_w = img_original.shape[:2]
template_h, template_w = img_template.shape[:2]


best_position_y = 0
best_position_x = 0
best_position_sad = 100000



point = []

# 원본 이미지 스캔
for original_x in range(original_w - template_w):
    for original_y in range(original_h - template_h):
        
        SAD = 0
        # #템플릿 이미지 스캔
        # for template_x in range(template_w):
        #     for template_y in range(template_h):

        #         original_pixel = img_original[template_y + original_y, template_x + original_x]
        #         template_pixel = img_template[template_y, template_x]
        
        #         SAD += abs( original_pixel - template_pixel)
       
        original_pixel = img_original[original_y:original_y + template_h , original_x:original_x + template_w].ravel()
        template_pixel = img_template.ravel()
        
        SAD = np.abs(np.subtract(original_pixel, template_pixel, dtype=np.float))
        SAD = SAD.sum()
   
        # 최소 SAD 지점 찾기  
        if best_position_sad > SAD: 
            
            best_position_y = original_y
            best_position_x = original_x
            best_position_sad = SAD
        point.append((original_x, original_y, SAD))



for p in point:
    if np.abs(p[2] - best_position_sad) < 100:
        print(p[2])
        cv2.rectangle(img_color, (p[0], p[1]), (p[0]+template_w, p[1]+template_h), (255, 0, 0), 3 )

cv2.imshow('result', img_color)
cv2.waitKey(0)
