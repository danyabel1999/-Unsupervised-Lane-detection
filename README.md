![Screenshot (244)](https://user-images.githubusercontent.com/63741335/126901487-9d09a6b8-1d9d-4cc0-b4d5-72bd219858a4.png)
# Unsupervised Techniques For Lane Detection 

## The following the steps of image processing and analysis involved in 'sliding_window_code_new.py',
1. The image captured by the camera is subjected to camera calibration.
![](/images/5.jpg) 

2. Apply Gamma Correction to the calibrated images.
![](/images/6.jpg) 

3. Sobel and HLS thresholding are executed simultaneously for edge detection, and then combined using a combined thresholding technique. 

4.  A perspective transform is applied on the image frame (birds-eye view). 

5. Sliding window algorithm is applied to find the lane pixels and boundary.

![slidingwindow search](https://user-images.githubusercontent.com/63741335/126900401-94cd6a75-8247-4dba-94c9-ae2149711cd9.png)

6. Higher order polyfit function is used to fit the detected road lane. 

7. Return to the original image and project the detected lane boundaries.
![output](https://user-images.githubusercontent.com/63741335/126900340-d502c51f-4ba5-4aa1-b572-a1efc584f5e1.jpg)
 
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
