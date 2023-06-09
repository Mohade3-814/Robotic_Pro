import cv2
import numpy as np

# Load the image
img = cv2.imread('test2-1.png')
#img = cv2.resize(cv2.imread('test2-3.JPG') , (400,400) , 500 , 500)

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds of the red color in HSV
#lower_red = np.array([0, 50, 50])
#upper_red = np.array([10, 255, 255])
lower_red = np.array([170, 50, 50])
upper_red = np.array([180, 255, 255])

# Create a mask for the red color
mask = cv2.inRange(hsv, lower_red, upper_red)

# Apply morphological operation to remove noise and small objects
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# Find contours in the mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
for i in range(len(contours)):
    cv2.drawContours(img, contours, i, (0, 255, 0), 2)

# Find the center of each contour
for contour in contours:
    moments = cv2.moments(contour)
    if moments['m00'] != 0 :
        center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    else :
        center = (0 , 0)

    cv2.circle(img , center, 5, (0, 0, 255), -1)

# Display the image with the contours and circles
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find the first gate
first_contour = None
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        first_contour = i
        break

# Find the last gate 
last_contour = first_contour
while hierarchy[0][last_contour][2] != -1:
    last_contour = hierarchy[0][last_contour][2]

# Find the centers of the first and last
moments_first = cv2.moments(first_contour)
center_first = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
print(center_first)
#cv2.circle(img , center_first , 5 , (255, 0, 0) , -1)

moments_last = cv2.moments(last_contour)
center_last = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
print(center_last)
#cv2.circle(img , center_last , 5 , (255, 0, 0) , -1)
