import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

random.seed()

def linearSteps(start, end, length=100):
    size = np.zeros(length, np.float32)
    return cv2.resize(np.array([list(start),list(end)], np.float32), (2,length), size, 0, 0, interpolation=cv2.INTER_LINEAR)

def linearMove(cur, path, frameNum, trans, length=100):
    if frameNum % length == 0:
        cur += 1
        trans = linearSteps(path[cur-1], path[cur], length)
        print(trans)
        print(path)
    if cur == len(path):
        return (-1, [], np.float32([[1,0,path[len(path)-1][0]],[0,1,path[len(path)-1][1]]]))
    return (cur, trans, np.float32([[1,0,trans[frameNum%length][0]],[0,1,trans[frameNum%length][1]]]))

def followPath(frameNum, img, path, cur, trans, rot):
    rows, cols = img.shape[:2]
    cur, trans, t1 = linearMove(cur, path, frameNum, trans, 20)
    r1 = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
    rot += 5
    rot %= 360
    dst1 = cv2.warpAffine(img,t1,(cols,rows))
    dst2 = cv2.warpAffine(dst1,r1,(cols,rows))
    dst = cv2.addWeighted(dst1, 0.5, dst2, 0.5, 0.0)
    return (dst, frameNum, cur, trans, rot)


face_cascade = cv2.CascadeClassifier('../opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../opencv/data/haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture('test3.mp4')

rot = 0
trans = []
ranpath = [(0,0)]
ret, img = cap.read()
rows, cols = img.shape[:2]
ranpath += [(random.randint(-cols/10, cols/10), random.randint(-rows/10, rows/10)) for x in range(100)]
cur = 0
frameNum = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (cols,rows))

while(cap.isOpened()):
    #dst, frameNum, cur, trans, rot = followPath(frameNum, img, ranpath, cur, trans, rot)
    dst = img
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        dst = cv2.rectangle(dst,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = dst[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    #cv2.imshow('test', dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    out.write(dst)

    frameNum += 1
    ret, img = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()