import cv2
import numpy as np

def edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 230, 250, cv2.THRESH_BINARY )
    blur = cv2.GaussianBlur(thresh, (3,3), 5)
    return blur

def rotate(l,n):
    return l[n:] + l[:n]

cap = cv2.VideoCapture("Tag0.mp4")
out = cv2.VideoWriter('output3.mp4', -1, 30, (1920, 400))

while True:
    _, frame = cap.read()
    blur = edge(frame)
    lena = cv2.imread("Lena.png")
    lena = cv2.resize(lena, (512,512))
    #EDGE DETECTION
    contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for j,cnt in zip(hierarchy[0],contours):
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            if j[3] != -1:
                squares.append(cnt)          
    cv2.drawContours(frame,squares,0,(0,0,255),2)
    
    #perform perspective transform on the images and video
    for sq in squares:
        pts1 = np.float32([list(i) for i in sq])
        pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, M, (800, 800))
        result = cv2.resize(warped, (1920, 400))
        orientationIndex = [[0,0], [800, 0], [800, 800], [0, 800]]
        lenaCor = [[0,0],[lena.shape[0],0],[lena.shape[0],lena.shape[1]],[0,lena.shape[1]]]
        for i,j,ori in zip(orientationIndex,\
                           ["LeftBottom","RightBottom","RightTop","LeftTop"],\
                           [0,1,2,3]):
            orientationCor = 0
            print(ori)
            break
        lenaOr = rotate(lenaCor,orientationCor)
        M = cv2.getPerspectiveTransform(np.float32(lenaCor),np.float32(lenaOr))
        lenaOriented = cv2.warpPerspective(lena,M,(512,512))
        M = cv2.getPerspectiveTransform(pts1,np.float32(lenaOr))
        linaOnImage = cv2.warpPerspective(lena, np.linalg.inv(M), (frame.shape[1],frame.shape[0]))

        #mask and join the images
        _, mask = cv2.threshold(linaOnImage, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        mask_inv = cv2.cvtColor(mask_inv,cv2.COLOR_BGR2GRAY)
        frame_bg = cv2.bitwise_and(frame,frame,mask = mask_inv)
        linaOn = cv2.add(linaOnImage,frame_bg)
        frameCopy = linaOn
        mask = np.zeros(frameCopy.shape, np.uint8)

    #output
    mass = cv2.hconcat([frame, linaOnImage, frame_bg, frameCopy])
    bangam = cv2.resize(mass, (1920, 400))
    out.write(bangam)
    cv2.imshow('Imposing image on Tag0',bangam)
    cv2.waitKey(1)
