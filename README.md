# Unsupervised Lane Detection 

## The following the steps of image processing and analysis involved in 'detection.py',
1. The image captured by the camera is subjected to camera calibration.
![](/images/5.jpg) 

2. Apply Gamma Correction to the calibrated images.
![](/images/6.jpg) 

3. Sobel and HLS thresholding are executed simultaneously for edge detection, and then combined using a combined thresholding technique. 

4.  A perspective transform is applied on the image frame (birds-eye view). 

5. Sliding window algorithm is applied to find the lane pixels and boundary.

![](/images/1.png)

6. Higher order polyfit function is used to fit the detected road lane. 

7. Return to the original image and project the detected lane boundaries.
 
![](/images/3.png)
## Required Python Libraries
1. cv2 (opencv)
2. numpy
3. matplotlib

## Code Execution 
`<cd Sliding-window-algorithm---Lane-Detection>` 
`cd Sliding-window-algorithm---Lane-Detection` 

`<python lane_detection.py>` 
`python lane_detection.py` 
