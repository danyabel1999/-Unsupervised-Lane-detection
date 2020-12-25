import cv2
import numpy as np
lower_limit = 100
upper_limit = 200
max_value = 255
number_of_channels = 3
thickness = 5
alpha_weight = 0.8
beta_weight = 1
gamma_weight = 0.0
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = max_value
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], number_of_channels), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, max_value, 0), thickness)

    img = cv2.addWeighted(img, alpha_weight, blank_image, beta_weight, gamma_weight)
    return img