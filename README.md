# Unsupervised Techniques For Lane Detection 

## The following the steps of image processing and analysis involved in 'sliding_window_code_new.py',
1. The image captured by the camera is subjected to camera calibration.

![Screenshot (244)](https://user-images.githubusercontent.com/63741335/126901487-9d09a6b8-1d9d-4cc0-b4d5-72bd219858a4.png)

2. Apply Gamma Correction to the calibrated images.

![gamma0 5](https://user-images.githubusercontent.com/63741335/126901677-3883546c-cc14-402d-8642-fbf9c39dc723.jpeg)
![gamma2](https://user-images.githubusercontent.com/63741335/126901679-75c49b55-1b11-4224-9063-4931e82f6322.jpeg)

3. Sobel and HLS thresholding are executed simultaneously for edge detection, and then combined using a combined thresholding technique.

![Screenshot (246)](https://user-images.githubusercontent.com/63741335/126901988-061fe0d2-1e6a-40aa-a4c7-24f5235477c5.png)
 

4.  A perspective transform is applied on the image frame (birds-eye view).
![perspectivetransform](https://user-images.githubusercontent.com/63741335/126901757-eb5525a7-c534-4eab-ad2f-e2b5d08c8737.png)
 

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

