import numpy as np
import cv2
import matplotlib.pyplot as plt



cap = cv2.VideoCapture('test2.mp4')

rot = 0
tranIndex = 0
target = 300

trans = cv2.resize(np.array([0,1]), np.array([0,0,0,0,0,0]), 6, 0, 0, cv2.INTER_LINEAR)

while(cap.isOpened()):
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows,cols,color = img.shape



    t1 = np.float32([[1,0,trans[tranIndex]],[0,1,trans[tranIndex]]])
    r1 = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)
    dst1 = cv2.warpAffine(img,t1,(cols,rows))
    dst2 = cv2.warpAffine(dst1,r1,(cols,rows))
    #dst = cv2.addWeighted(dst1, 0.5, dst2, 0.5, 0.0)
    tranIndex += 1
    rot += 1
    rot %= 360

    cv2.imshow('test', dst2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()