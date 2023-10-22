from LIME import LIME
import cv2

cap = cv2.VideoCapture('data/train/Jump/Jump_8_1.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 11)
ret, frame = cap.read()
a = LIME(img = frame)
map = a.optimizeIllumMap()
new_img = a.enhance()
cv2.imshow('1',map)
cv2.waitKey(10000)
cv2.imshow('2',new_img)