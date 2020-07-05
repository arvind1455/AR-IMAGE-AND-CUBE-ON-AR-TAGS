import cv2
import numpy as np

def edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 230, 250, cv2.THRESH_BINARY )
    blur = cv2.GaussianBlur(thresh, (3,3), 5)
    return blur

def projectionMatrix(h, K):
    rotation = 0.001 * np.matmul(np.linalg.inv(K), h)
    row1 = rotation[:, 0]
    row2 = rotation[:, 1]
    row3 = np.cross(row1, row2)
    row4 = rotation[:, 2]
    Rt = np.column_stack((row1, row2, row3, row4))
    P = np.matmul(K, Rt)
    return P

cap = cv2.VideoCapture("Tag2.mp4")
out = cv2.VideoWriter('cube2.mp4', -1, 30, (1280, 720))

while True:
    _, frame = cap.read()
    blur = edge(frame)
    lena = cv2.imread("Lena.png")
    lena = cv2.resize(lena, (512,512))
    alpha = 0.8
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
    cordinates = np.array(squares)       
    cv2.fillPoly(frame,cordinates,(228,187,37))

    for sq in squares:
        pts1 = np.float32([list(i) for i in sq]).reshape(-1,2)
        pts2 = np.float32([[0,0], [0,200], [200, 200], [200,0]]).reshape(-1,2)
        matrix, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5)
        
        K =([[1406.08415449821,0,0],
        [2.20679787308599, 1417.99930662800,0],
        [1014.13643417416, 566.347754321696,1]])
        P = projectionMatrix(matrix, K)

        #MULTIPLY THE PROJECTION MATRIX WITH THE POINTS TO GET THE CORDINATES OF THE CUBE
        x1,y1,z1 = np.matmul(P,[0,0,0,1])
        x2,y2,z2 = np.matmul(P,[200,0,0,1])
        x3,y3,z3 = np.matmul(P,[0,200,0,1])
        x4,y4,z4 = np.matmul(P,[200,200,0,1])
        x5,y5,z5 = np.matmul(2*P,[0,0,-200,1])
        x6,y6,z6 = np.matmul(2*P,[0,200,-200,1])
        x7,y7,z7 = np.matmul(2*P,[200,0,-200,1])
        x8,y8,z8 = np.matmul(P*2,[200,200,-200,1])
                    
        #JOIN THE CORDINATES BY THE CV2.LINE FUNCTION AND NORMALIZE BY Z TERMS TO FIT THE CUBE      
        cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (0,0,255), 2)
        cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x7/z7),int(y7/z7)), (0,0,255), 2)
        cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x6/z6),int(y6/z6)), (0,0,255), 2)
        cv2.line(frame,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)

        cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (228,187,37), 2)
        cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (228,187,37), 2)
        cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (228,187,37), 2)
        cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (228,187,37), 2)

        cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (255,0,255), 2)
        cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (255,0,255), 2)
        cv2.line(frame,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (255,0,255), 2)
        cv2.line(frame,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (255,0,255), 2)

    a = frame
    
    frame = cv2.resize(frame,(1280,720))
    out.write(frame)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)