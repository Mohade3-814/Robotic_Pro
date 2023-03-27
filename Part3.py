import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Initialize ROS node
rospy.init_node('gate_detecter')

# Initialize CvBridge
bridge = CvBridge()

# Create image publisher
image_pub = rospy.Publisher('/gate_detecter/image', Image, queue_size=1)

# Define HSV threshold for red color
lower_red = (0, 50, 50)
upper_red = (10, 255, 255)
 
# Create a kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Create video capture object
cap = cv2.VideoCapture('test3.mp4')

while not rospy.is_shutdown():

    # Read frame from video
    ret, frame = cap.read()

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask to filter out all colors except for red
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Apply morphological operations to mask
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours in mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Draw circle at the center of contours
    for contour in contours:
        moments = cv2.moments(contour)
        if moments['m00'] != 0 :
           center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
        else :
           center = (0 , 0)

        cv2.circle(frame , center, 5, (0, 0, 255), -1)

    # Publish resulting image
    try:
        image_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
    except CvBridgeError as e:
        print(e)

    # Sleep for a short time
    rospy.sleep(0.01)

# Release video capture object
cap.release()
