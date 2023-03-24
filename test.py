import cv2
import numpy as np

# Load the image and convert it to grayscale
img = cv2.imread('test2-1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the outermost contour
outer_contour = None
for i in range(len(hierarchy[0])):
    if hierarchy[0][i][3] == -1:
        outer_contour = i

# Draw a circle at the center of each contour
order = []
def draw_contours(image, contours, hierarchy, idx):
    global order
    M = cv2.moments(contours[idx])
    if M['m00'] == 0:
        return
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    order.append(idx)
    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][3] == idx:
            draw_contours(image, contours, hierarchy, i)

# Draw the contours on the image
draw_contours(img, contours, hierarchy, outer_contour)

# Draw the contour numbers on the image
for i, idx in enumerate(order):
    M = cv2.moments(contours[idx])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(img, str(i+1), (cx+10, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Identify the order of the contours for a robot to pass through them
order = [i+1 for i in range(len(order))]
print('Contour order:', order)
