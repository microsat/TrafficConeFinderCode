# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:54:27 2016

@author: nielsen8
"""

import numpy as np
import cv2


def convexHullIsPointingUp(hull):
    print'convexHullIsPointingUp'
    print 'checking if are we point up'
    x, y, w, h = cv2.boundingRect(hull)

    print 'hull', hull
    print 'x ,y ,w ,h '
    print x, y, w, h 

    aspectRatio = float(w) / h

    if aspectRatio > 0.8:
        print 'nope ->aspectRatio is > 0.8 it =', aspectRatio
        return False

    listOfPointsAboveCenter = []
    listOfPointsBelowCenter = []

    intYcenter = y + h / 2

    print ' intYcenter ', intYcenter
    print' find above or below vertical center '
    for point in hull:
        print 'point', point
        print 'point.x', point[0][0]
        print 'point.y', point[0][1]
        current_y = point[0][1]
        print 'y,intYcenter', current_y, intYcenter
        if current_y < intYcenter:
            print ' are above Center '
            listOfPointsAboveCenter.append(point)

        if current_y > intYcenter:
            print ' are below Center '
            listOfPointsBelowCenter.append(point)

            print ' intYcenter ', intYcenter

    print 'listOfPointsAboveCenter', listOfPointsAboveCenter
    print 'listOfPointsBelowCenter', listOfPointsBelowCenter

    intLeftMostPointBelowCenter = listOfPointsBelowCenter[0][0][0]
    intRightMostPointBelowCenter = listOfPointsBelowCenter[0][0][0]

    print 'intLeftMostPointBelowCenter', intLeftMostPointBelowCenter
    print 'intRightMostPointBelowCenter', intRightMostPointBelowCenter


    # determine left most point below center
    for point in listOfPointsBelowCenter:
            print 'point', point
            pnt = point[0][0]
            if pnt < intLeftMostPointBelowCenter:
                intLeftMostPointBelowCenter = pnt

    # determine right most point below center
    for point in listOfPointsBelowCenter:
            if point[0][0] >= intRightMostPointBelowCenter:
                intRightMostPointBelowCenter = point[0][0]

    print 'intLeftMostPointBelowCenter', intLeftMostPointBelowCenter
    print 'intRightMostPointBelowCenter', intRightMostPointBelowCenter

    for point in listOfPointsAboveCenter:
        if point[0][0] < intLeftMostPointBelowCenter or  \
                     point[0][0] > intRightMostPointBelowCenter:
            print 'not point up'
            return False

    print 'we are pointing up'
    return True

print 'step one'
print 'looking for cones'
print 'Load an color image'
# load in a image
img = cv2.imread('./images/14.jpg', -1)

print 'Image shape ', img.shape
cv2.imshow('image', img)

imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# threshold on low range of HSV red
low_redl = np.array([0, 135, 135])
low_redh = np.array([15, 255, 255])
imgThreshLow = cv2.inRange(imgHSV, low_redl, low_redh)

# threshold on high range of HSV red
high_redl = np.array([159, 135, 135])
high_redh = np.array([179, 255, 255])
imgThreshHigh = cv2.inRange(imgHSV, high_redl, high_redh)

# combine low range red thresh and high range red thresh
imgThresh = cv2.bitwise_or(imgThreshLow, imgThreshHigh)
cv2.imshow('imgThresh ', imgThresh)

# clone/copy thresh image before smoothing
imgThreshSmoothed = imgThresh.copy()

# open image (erode, then dilate)
kernel = np.ones((3, 3), np.uint8)
imgThreshSmoothed = cv2.erode(imgThresh, kernel, iterations=1)
imgThreshSmoothed = cv2.dilate(imgThreshSmoothed, kernel, iterations=1)
# Gaussian blur
imgThreshSmoothed = cv2.GaussianBlur(imgThreshSmoothed, (5, 5), 0)
cv2.imshow('imgThreshSmoothed ', imgThreshSmoothed)
print 'get Canny edges'

imgCanny = cv2.Canny(imgThreshSmoothed, 160, 80)
cv2.imshow('imgCanny ', imgCanny)
image, contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('findContours ', image)

print 'findContours'
print 'len(contours)', len(contours)
index = 0
listOfContours = []
if len(contours) != 0:
    for cnt in contours:
        print 'contours[%d] len = %d', index, len(cnt)
        epsilon = 0.1*cv2.arcLength(cnt, True)
        print'epsilon', epsilon
        dat = cv2.approxPolyDP(cnt, 6.7, True)
        print 'dat ', len(dat)
        listOfContours.append(dat)
        index = index + 1


print 'listOfContours', len(listOfContours)
index = 0
listOfhull = []
print 'convexHull'
for contour in listOfContours:
        hull = cv2.convexHull(contour)
        print ' index of listOfContours, len(hull) ', index, len(hull)
        index = index + 1
        # print 'convexHull',len(temp)
        if (len(hull) >= 3 and len(hull) <= 10):
            print 'hull', len(hull)
        else:
            print'not  checking'
            continue

        if convexHullIsPointingUp(hull):
            print '-Point up-'
            listOfhull.append(hull)


print 'listOfhull', len(listOfhull)

imghull2 = cv2.drawContours(img, listOfContours, -1, (0, 0, 255), 3)

imghull = cv2.drawContours(img, listOfhull, -1, (0, 255, 0), 3)

cv2.imshow('hull ', imghull)
cv2.imshow('hull2 ', imghull2)


print '- press any key to end -'
cv2.waitKey(0)
cv2.destroyAllWindows()
