import numpy as np
import cv2 as cv

img = cv.imread('map.bmp', cv.IMREAD_GRAYSCALE)

height, width = img.shape

colored = np.zeros((height, width, 3), dtype=np.uint8)

lines = cv.imread('map.bmp', cv.IMREAD_COLOR)

new_lines_img = np.zeros((height, width), dtype=np.uint8)

canny = None

sigma = 0.33

def procImage(cannyMin, cannyMax):
    #_,gray = cv.threshold(img, 100, 255, cv.THRESH_TRUNC)
    _,gray = cv.threshold(img, 100, 255, cv.THRESH_BINARY)

    cv.imshow('gray', gray)

    v = np.median(gray)

    #---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    canny = cv.Canny(gray, lower, upper)

    new_lines = cv.HoughLinesP(canny,1,np.pi/180,15,minLineLength=1,maxLineGap=2)
    
    for line in new_lines:
        for x1,y1,x2,y2 in line:
            cv.line(new_lines_img,(x1,y1),(x2,y2),(255,255,255),2)

    cv.imshow('newLines', new_lines_img)

    contours,_ = cv.findContours(new_lines_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    r = 1
    g = 1
    b = 1
    for cnt in contours:
        area = cv.contourArea(cnt)

        approx = cv.approxPolyDP(cnt, 0.000001 * cv.arcLength(cnt, True), True)

        cv.fillPoly(colored, pts = [approx], color = (b, g, r))
        cv.drawContours(lines, [approx], 0, (0, 0, 255), 1)

        r = r + 1
        if (r >= 255):
            g = g + 1
            r = 1
            if (g >= 255):
                b = b + 1
                g = 1
            
    cv.imshow('canny', canny)
    cv.imshow('map', colored)
    cv.imshow('lines', lines)

global cmin
global cmax

cmin = 500
cmax = 1000

procImage(cmin, cmax)

def on_canny_change(value):
    global cmax 
    cmax = value

def on_canny_change_min(value):
    global cmin
    cmin = value

def refresh(value):
    if (value):
        procImage(cmin, cmax)

cv.createTrackbar('canThresh', 'map', 0, 1000, on_canny_change)
cv.createTrackbar('canThreshMin', 'map', 0, 1000, on_canny_change_min)
cv.createTrackbar('refresh', 'map', 0, 1, refresh)

if cv.waitKey(0) & 0xFF == ord('q'):
    cv.destroyAllWindows()