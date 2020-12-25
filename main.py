import matplotlib.pylab as plt
import cv2
import numpy as np
import high_level
import low_level
cap = cv2.VideoCapture('test2.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = high_level.process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

