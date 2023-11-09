# Canny Edge Detection

**Process**

1. Given a Random Image

![Alt text](docs/image.png)

2. Convert the image to grayscale with `cvtColor()` of opencv

![Alt text](docs/image-1.png)

3. Apply a Gaussian filter with a 5x5 kernel and $\sigma=1.5$

![Alt text](docs/image-2.png)

4. Equalize the image with the function `equalizeHist` of opencv

5. Apply the Sobel border detection, you will get the image $G$, and the border angles $\theta$

![Alt text](docs/image-3.png)

6. Apply the Canny Edge Detection and apply the threshold to get the final borders.

![Alt text](docs/image-4.png)