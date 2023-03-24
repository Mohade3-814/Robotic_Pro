# Importing librariers
import cv2
import numpy as np

# Address of the image
address = 'test2-3.JPG'

# Load the image and convert it to HSV format and resize it to check the output better
image = cv2.resize(cv2.imread(address) , (300,300) , 500 , 500)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

"""
The lower and upper bounds of each shade of red/orange
lower_red_1 = np.array([0, 50, 50])      darkest red
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([170, 50, 50])    lightest red
upper_red_2 = np.array([180, 255, 255])

lower_orange = np.array([5, 50, 50])     pure orange
upper_orange = np.array([15, 255, 255])
"""

# merge all the lower and upper bounds
lower_bounds = np.array([0,50,50])
upper_bounds = np.array([180,255,255])

# Create a mask of the image only showing pixels within the specified color range
mask = cv2.inRange(hsv_image, lower_bounds, upper_bounds)

# Apply morphological operation to remove noise and small objects
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# Find contours in the mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the outermost contour
outer_contour = None
for i in range(len(hierarchy[0])):
    if hierarchy[0][i][3] == -1:
        outer_contour = i

# Filter out contours that are too small or too large
min_area = 500
max_area = 100000
valid_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area and area < max_area:
        valid_contours.append(contour)

# Draw the contour of the gate with a green color and a thickness of 2
cv2.drawContours(image , valid_contours, -1, (0, 255, 0), 2)

# Get the center point of the contour and draw a circle at the center
for contour in valid_contours:
    moments = cv2.moments(contour)
    center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    cv2.circle(image , center, 5, (0, 0, 255), -1)

# Display the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
