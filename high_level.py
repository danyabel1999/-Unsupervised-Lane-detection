import cv2
import numpy as np
import low_level
lower_limit = 100
upper_limit = 200
gamma_weight = 0.0
rho = 2
theta = np.pi/180
threshold = 50
mini = 40
maxi = 25

def process(image):
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
        (0, height), (width/2, height/2),  (width, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, lower_limit, upper_limit)


    cropped_image = low_level.region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    lines = cv2.HoughLinesP(cropped_image,
                           rho,
                           theta,
                           threshold,
                           lines=np.array([]),
                           minLineLength=mini,
                           maxLineGap=maxi)
    if lines is None:
        image_with_lines = image
    else:
        image_with_lines = low_level.drow_the_lines(image, lines)
    return image_with_lines