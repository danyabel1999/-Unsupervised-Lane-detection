import numpy as np
import matplotlib.pyplot as plt
import cv2

def gamma_correction(RGBimage, correct_param=2, equalizeHist=False):
    red = RGBimage[:, :, 2]
    green = RGBimage[:, :, 1]
    blue = RGBimage[:, :, 0]
    # gamma correction in the red channel
    red = red / 255.0
    red = cv2.pow(red, correct_param)
    red = np.uint8(red * 255)
    if equalizeHist:
        red = cv2.equalizeHist(red)
    #gamma correction in the green channel
    green = green / 255.0
    green = cv2.pow(green, correct_param)
    green = np.uint8(green * 255)
    if equalizeHist:
        green = cv2.equalizeHist(green)
    #gamma correction in the blue channel
    blue = blue / 255.0
    blue = cv2.pow(blue, correct_param)
    blue = np.uint8(blue * 255)
    if equalizeHist:
        blue = cv2.equalizeHist(blue)
    #combining gamma corrected channels
    output = cv2.merge((blue, green, red))
    #returning the gamma corrected output
    return output

def color_threshold(image, sthresh=(0,255), vthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    #thersholding in s channel
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    #thresholding in v channel
    v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1
    
    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary) == 1] = 1
    # Return the combined s_channel & v_channel binary image
    return output

def sobel(img):
    #converting the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    thresh_min = 20  # setting the minimum threshold
    thresh_max = 100 # setting the maximum thereshold
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= thresh_min) & (scaled_sobely <= thresh_max)] = 1  #Sobel thresholding in y direction
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max)] = 1  #Sobel thresholding in x direction
    c_binary = color_threshold(img, sthresh=(100, 255), vthresh=(50, 255))
    #return the combined thresholded output
    return (sybinary + sxbinary+c_binary)

def warp(img):
    img_size=(img.shape[1],img.shape[0])
    #defining the source points 
    src = np.float32(
        [[850, 500],
         [1000, 658],
         [23, 658],
         [270, 500]])
    #defining the destination points
    dst = np.float32(
        [[1000, 500],
         [1000, 658],
         [23, 658],
         [10, 500]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    #return perpective transformed image
    return warped
def inv_warp(img):
    img_size=(img.shape[1],img.shape[0])
    #define source points
    src = np.float32(
        [[850, 500],
         [1000, 658],
         [23, 658],
         [270, 500]])
    #define destination points
    dst = np.float32(
        [[1000, 500],
         [1000, 658],
         [23, 658],
         [10, 500]])
    Minv = cv2.getPerspectiveTransform(dst, src)
    invwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    #return inverse perspective trasnformed image
    return invwarped

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 30
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    #print(leftx)

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 3)
    right_fit = np.polyfit(righty, rightx, 3)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    try:
        left_fitx = left_fit[0] * ploty ** 3 + left_fit[1] * ploty ** 2 + left_fit[2] * ploty + left_fit[3]
        right_fitx = right_fit[0] * ploty ** 3 + right_fit[1] * ploty ** 2 + right_fit[2] * ploty + right_fit[3]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 3 + 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 3 + 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    return color_warp

cap = cv2.VideoCapture('test1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi', fourcc, 20.0, (1280, 720))

while (cap.isOpened()):
    ret, frame = cap.read()
    corrected_img = gamma_correction(frame)
    binary_out = sobel(corrected_img)
    binary_warped = warp(binary_out)
    color_warp = fit_polynomial(binary_warped)
    ##inv_pers = inv_warp(out_img)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = inv_warp(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
    cv2.imshow("result", result)
    out.write(result)
    #result = cv2.addWeighted(frame,1.0,inv_pers,1.0,1.0)
    #cv2.imshow("result", result)
    #plt.imshow(binary_warped,cmap ='gray')
    #plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()