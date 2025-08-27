# Computer_Vision_Skills

pip install opencv-python

pip install ipykernel

**OpenCV - imread, imshow, waitkey, destroyAllWindows, imwrite, VideoCapture, videoread, cv2.VideoWriter_fourcc**


**Table of Contents:**

**A) Reading and Writing Images**

**B) Working with Video Files**

**C) Exploring Colour Space**

**D) Colour Thresholding**

**E) Image Resizing, Scaling and Interpolation**

**F) Flip, Rotate and Crop Images**

**G) Drawing Lines and Shapes using OpenCV**

**H) Adding Text to Images**

**I) Affine and Perspective Transformation**

**J) Image Filters**

**K) Applying Blur Filters - Average, Gaussian, Median**

**L) Edge Detection Using Sobel, Canny & Laplacian**

**M) Calculating and Plotting Histogram**

**N) Image Segmentation**

**O) Haar Cascade for Face Detection**

### (II) Pytorch

**A) Introduction to Pytorch**

**B) Introduction to Tensors**

**C) Indexing Tensors**

**D) Using Random Number to create Noise Image**

**E) Tensors of Zero's and One's**

**F) Tensor Data Types**

**G) Tensor Manipulation**

**H) Matrix Aggregation**

### **A) Reading and Writing Images:**

**1. Read the Image - image = cv2.imread("./mountain.jpg")**

cv2.imread("./mountain.jpg") - Returns a NumPy Array; Loads image into Matrix grad of Pixel  

**2. Display the Image:**

# Display the image

**cv2.imshow("Image window", image)** - cv2 Opens in a Window, the window name is "Image Window"

**cv2.waitKey(0)** - It pauses the program until a key is pressed;We want to keep the window open until an key is pressed; We don't want the window to be opened and closed instantly; It will wait until a key is pressed

**cv2.destroyAllWindows()** - If we press any key that window will be closed and this will release the memory; This will release that particular window; 

**It will show a output of 14s in VS Code terminal because, we kept the window open for 14 seconds**

**3. Write the Image:**

# Write the image

**image_new = image + 30** - Creating a new NumPy array and add 30 in all of the pixels

**cv2.imwrite("./output.jpg", image_new)**- Path and Name of Image; NumPy array name is image_new

### **B) Working with Video Files**

**4. Load the Video** - video = cv2.VideoCapture("./mountain.mp4") (It will provide a object, to save it, we assign it to "video") (It helps to capture video, frame by frame)

**5. Display the Video:**

while True:
    ret, frame = video.read() (Returns true/false as it reads the frame; If all frames are read, it gives out False)
    if not ret:
        break
    cv2.imshow("Video Frame", frame)  (Same code because video  is agin the set of frames moving one another)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

**cv2.waitKey(1) & 0xFF == ord('q') - Waits for 1 Milliseconds for Frames; i.e, at T0 displays F1 and waits for 1 Millisecond and displays F2 frame at T1 **

**By default waitkey will return -1 if we didn't press any key on keyboard; If we press anything it will return bit component of that ASCII Value;**

**& 0xFF - This will trim the bit component of it to the 8 digits; == ord('q')**

**If I press Q, ASCII of Q will come and we will compare with actual Q value and then only it will break; If I press A it will not match and it won't break out of the while loop**

**Once video is fully played, ret will become false,it will come out of while loop, we will release it and will destroy all the windows**

**If we didn't write video.release and destroy code, system will break the kernel and we have to start the kernel again**

**6. Write a Video:**

**While writing we need to capture the Width and height of original video, so that it won't get corrupted due to any issues**

video = cv2.VideoCapture("./mountain.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("./output.mp4", fourcc, 30, (width, height)) [30 - FPS]

**Made some changes in Pixels like we did for Images**

while True:
    ret, frame = video.read()
    if not ret:
        break
    frame = frame + 20
    out.write(frame)

**We shouldn't add it before if, in the while condition, because in the last condition ret will be false and we will be adding none + 20; Frame will be none at the last**

**So we should do all the working with frames only after if not ret condition**

**It is not a corrupter image; It is how pixels lie after another**

### **C) Exploring Colour Space**

**7. Exploring Color Spaces:**

### BGR is the color space that OpenCV uses by default

**LAB and HSV Looks weird to us, because as Humans we are not used to see the world in that way**

**RGB and BGR has no much difference; It is just a swap of one channel**

## When OpenCV by default, reads the image as BGR, why does the default image (RGB) and the image read by BGR looks the same? Refer the notes for this

**(i) Convert to GrayScale**

# Convert to grayscale

gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("./gray_image.jpg", gray_image)

**Convert to HSV:**

hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

cv2.imwrite("./hsv_image.jpg", hsv_image)

**Convert to LAB**

lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)

cv2.imwrite("./lab_image.jpg", lab_image)

### **D) Colour Thresholding**

**8. Color Thresholding:**

**Converted to RGB to load first** - image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

**Converted to HSV to use for Color Thresholding** - image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

**Targeting the Giraffe Outline**

**Targeting the brightness** 

lower_white = np.array([0,0,190])

upper_white = np.array([30,80,255]) **It is not pure white it has some kind of Brown Orangish; 0 to 60 in Terms of Original HSV; Here it is scaled; 80 because saturation is not too gray; It has some level of brown as well**

**Mask Plotting ** - mask = cv2.inRange(image_hsv, lower_white, upper_white) [Not Black and white because we are using Matplotlib and it uses for a single channel, that purple and yellow] **With OpenCV we can get Black and White**

**Colour Thresholding:**

black_hsv = np.uint8([[[120, 255, 0]]]) - **Hue is 120, Saturation is 255, Brightness is 0; When 0 only we will get black**

black_bgr = cv2.cvtColor(black_hsv, cv2.COLOR_HSV2BGR)[0][0] **Converting to BGR because OpenCV considers BGR in default**

**[0][0] - Added because we want only 120,255,0**

image_result = image.copy()

image_result[mask>0] = black_bgr **[If Mask>0, Colour Thresholding is true and we want to change it**

**Converting the final Image to RGB** - image_result_rgb = cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB)

### **E) Image Resizing, Scaling and Interpolation**

**9. Image Resizing, Scaling and Interpolation:**

**Converted from BGR to RGB**

**image_rgb.shape - Gives Height, Width, Number of Channels**

**Image Resizing:**

**new_height, new_width = 400, 400**

**1. resized_image = cv2.resize(image_rgb, (new_width, new_height))**

**Neck of the Giraffee looks squeezed**

**Image Scaling:**

**1. Scale to 50%**

scale_percentage = 50

**s_width = int(image_rgb.shape[1] * scale_percentage / 100)** (shape[1] --> Width is at the index 1)

**s_height = int(image_rgb.shape[0] * scale_percentage / 100)** (shape[0] --> Height is at the index 0)

**scaled_image = cv2.resize(image_rgb, (s_width, s_height), interpolation=cv2.INTER_AREA)**

**Scale looks similar; We will see loss in details when upsample/downsample, because the algorithm knows to remove and keep the important ones**

**But when it is time to fill new pixels, it gets confused**

**Generative Algorithms like "GAN are for creating new images using Neural Networks"**

**Interpolation:**

**Did interpolation using different methods such as Nearest, Linear, Cubic, Area, Lanczos4** (Refer Notebook for the code)

**Lanczos4 - Showed Blurry Effect**

**Original, Nearest, Area - Shows similar outputs**

**Linear - Looks Blurry**

**Cubic - Doesn't look much blurry like Linear; Cubic is better than Linear; Lanczos4 is better than Cubic**

But, Original, Linear and Area are of different sizes; Original - 100%; Linear and Area - 600 x 600

### **F) Flip, Rotate and Crop Images:**

**1. Flipping Images:**

**1 - Horizontal Flip**

**0 - Vertical Flip**

**-1 - Flip Both Axes**

flip_horizontal = cv2.flip(image, 1)

flip_vertical = cv2.flip(image, 0)

flip_both = cv2.flip(image, -1)

**2. Printing the Image**

image = cv2.imread("../images/giraffe-Kenya.png")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

**Flipping Images and Converting to RGB ** - images = [image_rgb,cv2.cvtColor(flip_horizontal, cv2.COLOR_BGR2RGB),cv2.cvtColor(flip_vertical, cv2.COLOR_BGR2RGB),cv2.cvtColor(flip_both, cv2.COLOR_BGR2RGB)]

**2. Rotating Images:**

rotate_90_clockwise = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)

rotate_90_counterclockwise = cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

rotate_180 = cv2.rotate(image_rgb, cv2.ROTATE_180)

images = [image_rgb,rotate_90_clockwise,rotate_90_counterclockwise,rotate_180]

**3. Cropping an Image:**

print(image_rgb.shape) - Gives shape of Image in Height, Width, Channles; type(image_rgb) -> numpy.ndarray

x_start, y_start = 400,300 --> (x1,y1)

x_end, y_end = 800, 900 --> (x2,y2)

cropped_image = image_rgb[y_start:y_end, x_start:x_end]

**In OpenCV, the co-ordinates are not like what we used to see; Co-Ordinates (0,0) starts from Top Left of the Image - Left to Right goes for X-Axis and Top to Bottom goes for Y-Axis**

**G) Drawing Lines and Shapes using OpenCV:**

**1. We need a empty Canvas to draw lines; So we will create it first**

height, width = 500, 700; blue = (255, 127, 0) **(BGR Blue; Highest Intensity is for Blue and Red has nothing; No Intensity of R)**

canvas = np.full((height, width, 3), blue, dtype=np.uint8) (All the integers that we are going to create on the canvas are going to be integer as always); np.full will fill it

**To print the Canvas** - plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

**Reason for converting to RGB, because Matplotlib expects to be in RGB Image format**

**If we didn't mention the color conversion, It showed a Orange Canvas, because Matplotlib expects in RGB, and we have BGR; So we have to convert it first**

**2. To Draw a Line:**

**cv2.line(canvas, (50, 50), (650, 450), (255, 255, 255), 5)** 

**(50,50) - Starting value for drawing the line; (650,450) - Ending point of the line; (255,255,255) - We want a full white color line; 5 - Thickness of the line**

**3. To Draw a Rectangle:**

**cv2.rectangle(canvas, (100, 100), (300, 300), (0, 255, 255), -1) - (100,100) Starting Co-ordinate; (300,300) - Ending Co-Ordinate; It is a Square; (0, 255, 255) - Colour of Rectangle (Yellow); -1 - To fill the whole thickness**

**4. To Draw a Circle:**

**cv2.circle(canvas, (500, 150), 100, (0, 0, 255), 5) - (500,150) - Centre Point (We need only Centre Point to draw a cirle); 100 - Radius; (0,0,255) - Color; 5 - Thickness **

**5. To Draw an Ellipse:**

**cv2.ellipse(canvas, (350, 350), (100, 50), 45, 0, 360, (0, 255, 0), 3) - (350,350) - Starting Co-Ordinate; (100,50) - Axis of Ellipse (Length of Major and Minor Axis); 45 - Angle of Ellipse; 0 - Starting Angle; 360 - Ending Angle; (45 Degree will be considered in the Starting and ending angle); (0,255,0) - Green Color; 3 - Thickness of Ellipse**

**6. To Draw a Polygon:**

points = np.array([[150, 400], [300, 400], [225, 300]], np.int32)

new_reshaped_point = points.reshape((-1, 1, 2))

print(new_reshaped_point)

cv2.polylines(canvas, [new_reshaped_point], False, (255, 0, 255), 3)

**Polygon has many points (Has many pairs); If we have two pairs of x and y, we can get a rectangle; If we have a single x and y, we can get an ellipse, or a circle; But not with Polygon; It consits of Multiple x and y**

**It can be Xn, Yn; Where n can be greater than 2,3 or 10; If n=2, we will be getting a line; We need more than 2 points to get a polygon**

**We mentioned multiple pairs with help of NumPy**

**[150,400] - First Pair of points; [300,400] - Second pair; [225,300] - Third Pair; np.int32 -On the Image it is not an image and just an Int Value**

**points.reshape((-1, 1, 2)** - These are new reshaped points; (-1,1,2) - In a single row I want two values, -1 means auto; 1,2 - We want one row and two columns in each row of data [We have pair of two-dimensional matrix]

**We will use the Reshaped Matrix to plot the Polygon**

**cv2.polylines(canvas, [new_reshaped_point], False, (255, 0, 255), 3)** - False - IsClosed (We want it to be closed or not) - We don't want it to be closed (we will get three points, but it won't get automatically close by itself, we have to mentioned in that argument to close or not); If True it will select the first and last point and close it (Loops of X and Y will get closed and we get a Triangle); (255,0,255) - Colour of the Polygon; 3 - Thickness of Polygon

**H) Adding Text to Images:**

**1. For creating Canvas:**

height, width = 500, 700

sky_blue = (255, 127, 0)

**canvas = np.full((height, width, 3), sky_blue, dtype=np.uint8)** - 3 - Need the Height and Width in all 3 channels; datatype - int

**canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)** - Need to convert to RGB, or else we get canvas as Orange Color

**2. To add the Text:**

**text = "Hello, OpenCV"** - Text we would like to add

**font = cv2.FONT_HERSHEY_COMPLEX** - Text Font

**org = (200, 100)** - Coordinate for the bottom left corner of the text to start from 

**font_scale = 1.5** - If suppose by default font size is 14, Font scale is 1.5 then, 

**color = (255, 255, 255)** - Colour of Font - White

thickness = 2

**cv2.putText(canvas, text, org, font, font_scale, color, thickness, cv2.LINE_AA)** - cv2.LINE_AA - The type of Line which we want to use (type of line we use for rendering the text)

**3. Trying with Different Fonts:**

fonts = [cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,cv2.FONT_HERSHEY_SCRIPT_COMPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL]

**Trying different fonts in same x, but making y to differ**

y_offset = 50
for i, font in enumerate(fonts):
    text = f"Font {i+1}"
    cv2.putText(canvas, text, (50,y_offset), font, 1, (0,0, 255), 2, cv2.LINE_AA)
    y_offset+=50

**(50,y_offset) - We will start from x=50;y=50; 1- Scale; (0,0,255) - Colour (Blue); y_offset+=50 - Every Loop y will change**

**4. Trying with different font size and thickness:**

styles = [(1,2),(2,2),(1,4),(2,4),] (Font Size/Scale, Thickness)

y_offset = 50

for font_scale, thickness in styles:
    text = f"Font {i+1}"
    cv2.putText(canvas, text, (50,y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0, 255), thickness, cv2.LINE_AA)
    y_offset+=50

**I) Affine and Perspective Transformation**

**1. Defining 4 Points to do the Perspective Transformation**

rows, cols, _ = image_rgb.shape

**For Perspective Transformation we need 4 points**

input_points = np.float32([[50, 50], [cols - 50, 50], [50, rows-50], [cols-50, rows-50]])

output_points = np.float32([[10, 100], [cols - 100, 50], [100, rows-10], [cols-50, rows-100]])

**The Input Points to get transformed to the Ouput points after Transformation (We can map any values)**

**Affine Transformation requires a Transformation Matrix, 2x3 Matrix; Perspective Transformation also needs it, concepts remains same, it requires a different matrix**

**It is 3x3 Matrix and not 2x3 Matrix**

**2. Perspective Transformation Components:**

Trasformation matrix for Perspective trasformation

3x3

a, b, c

d, e, f

g, h, 1

a,b,c is responsioble for horizontal scaling, rotation and tranlation

d,e,f is responsioble for vertical scaling, rotation and tranlation

g,h  : Perspective components that account for depth and skew

**Perspective Transformation needs g,h for depth and skewness; Affine Transformation can use 2x3, but Perspective needs some more information**

**3. We can get that Transformation Matrix with help of OpenCV; We have only Input and Output Points**

**M = cv2.getPerspectiveTransform(input_points, output_points)**

**This is the 3x3 Transformation Matrix**

**4. Getting the Perspective Image:**

perspective_image = cv2.warpPerspective(image_rgb, M, (cols, rows))

**Image will be rotated and we won't have any information over there**

**5. Affine Transformation:**

**For Affine Transformation we need only 3 Points**

input_points = np.float32([[50,50], [200, 50], [50, 200]])

output_points = np.float32([[10,100], [200, 50], [100, 250]])

**To get the Transformation Matrix:**

M = cv2.getAffineTransform(input_points, output_points)

**To do Affine Transformation:**

affine_image = cv2.warpAffine(image_rgb, M, (cols, rows))

**It will get rotated and we can also see shearing and body of Girraffe looks weird, stretched (It is like Rotated, Shearing, Translation)**

**6. Rotation using Affine Transformation:**

**Rotation Matrix - 2D**

M = cv2.getRotationMatrix2D(center,angle, scale)

**Then we use the same WrapAffine**

rotated_image = cv2.warpAffine(image_rgb, M, (cols, rows))

**Previously we did base version of Affine Transformation, now we did more rotation**

**7. Shearing using Affine:**

shear_x, shear_y = 0.5, 0 [Shearing for Horizontal, Vertical]

**Custom Transformation Matrix (2x3)**

M = np.float32([[1, shear_x, 0], 
           [shear_y, 1, 0]])

**We don't use any rotation matrix; Because in shearing we don't have any rotation**

**Shearing using Affine Transformation**

sheared_image = cv2.warpAffine(image_rgb, M, (cols*2, rows))

**We are changing the width; We are passing new width; We are increasing new width; We have taken shear_x and multiplied with rows to increase the column; We will get the whatever column value multiplied by 2**

**8. Translattion using OpenCV:**

tx, ty = 200, 300

**Creating Transformation Matrix**

M = np.float32([[1, 0, tx], 
           [0, 1, ty]])   (Made Shearing Values zero)

**Creating Translated Image:**

translated_image = cv2.warpAffine(image_rgb, M, (cols, rows))

**In X-Axis, we have shifted 200 Pixels and 300 Pixels in Y-Axis**

**Affine Transformation requires 3 pairs of x,y; Perspective Transformation requires 4 points**

**Parallel lines remain parallel in Affine Transformation; But can converge in Perspective; Straight Lines remains same in both of them**

## Affine we can use for Scaling, Rotation, Shearing, Translation; Perspective we can use to change the view point or to simulate some kind of depth in our image

**J) Image Filters:**

**1. We are reading as Grayscale Image; We can also mention as 0,0**

image = cv2.imread("./filter.jpg", cv2.IMREAD_GRAYSCALE)  

plt.imshow(image, cmap='gray') **Cmap should be gray because we have converted it as a Grayscale Image; In the course while he had not mentioned, he got like a Yellow Image**

### We are creating a custom filter

**2. We started passing the Horizontal kernel, but we still had horizontal lines in the images because, the intensity that we are passing might not be that strong; We passed np.array([[-1, -1, -1], 
                              [0, 0, 0], 
                              [1, 1, 1]]); Difference wouldn't have been that much, so we had some horizontal lines in the Images**

**More the intensity, better it would be able to extract from**

### We want to extract only Horizontal Lines; Impact is more when we kept 10; We can see Horizontal Lines more clearly and vertical lines are almost gone; Because of the calculation we were able to get some diagnols alone

**3. We transposed and created for Vertical and Diagonal too**

**For horizontal x is in horizontal row**

horizontal_kernel = np.array([[-1, -1, -1], 
                              [x, x, x], 
                              [1, 1, 1]])

**For vertical x is in Vertical column**

vertical_kernel = np.array([[-1, x, 1], 
                            [-1, x, 1], 
                            [-1, x, 1]])

**For Diagonal x should be in Diagonal; Number above and below x should also follow a diagonal effect**

diagonal_kernel = np.array([[x, -1, -1], 
                            [1, x, -1], 
                            [1, 1, x]])

**From visual output itself we can understand which filter it is; Each filter will filter out it's most prominent features**

**We created own custom filters and changed intensities upto us; We can have as -5, -10, -15; Difference between them should be high enough so that they can detect the edges**

**We learnt about how to extract features like Horizontal, Vertical and Diagonal Separately**

**K) Applying Blur Filters - Average, Gaussian, Median**

**1. Average Blur:**

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

### The requirement for any kernel is OpenCV is we need to pass as Grayscale

kernel_s = 10

avergae_blur = cv2.blur(image_grey, (kernel_s, kernel_s))

**Higher the kernel size, more the bluriness effect**

plt.imshow(avergae_blur, cmap='grey') **Need to mention cmap=grey too**

**2. Gaussian Blur:**

kernel_s = 31 **This requires an odd number to work; In that video he got an error**

**Thir argument that takes below is SigmaX, it is the Standard Deviation of the Gaussian in the X Direction; Higher the Sigma X, better the smoothing effect**

guassian_blur = cv2.GaussianBlur(image_grey, (kernel_s, kernel_s), 0)

plt.imshow(guassian_blur, cmap='grey')

**3. Median Blur**

median_blur = cv2.medianBlur(image_grey, 11) **Need only one kernel value**

**3 is the kernel size; Stronger noise reduction may distort final results; Bigger the value, stronger blur we will get**

plt.imshow(guassian_blur, cmap='grey')

**We the compared all the three blurs with original image**

**Original Grayscale image looks ver sharp; Average Blur looks average; Gaussian Blur does lot of blur; Median blur changes the image itself, it is better when we want to reduce the noise with help of median blur because in a particular kernel it picks median of it and when we try to pick median of it, lower the possible or the higher intensityor the lowest intensity might be lost, because of how median works**

**We can go with average or median blurring; Or with Gaussian blurring when we want to smoothen; In terms of noise we should go with Median Blurring**

**L) Edge Detection Using Sobel, Canny & Laplacian**

1. Reading the Image - image = cv2.imread("../images/giraffe-Kenya.png", cv2.IMREAD_GRAYSCALE)

2. **Sobel Edge Detector**

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) **Output depth of images; 1 is derivative in x direction, 0 is derivative in y direction; We need to calculate the sobel in x direction. So that's why we have kept x as one y as zero.We don't want to calculate in both direction; k is kernel size and we kept as 3**

sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) **Similarly we created for y direction too**

### 3. Once we have the derivative in both the direction, we need to calculate the magnitude.

sobel_combined = cv2.magnitude(sobel_x, sobel_y)

**This is how we do edge detection** 

3. **Laplacian Edge Detection:**

We have to mention for Laplacian it will be pretty straightforward; Next is Output Depth; Next is Kernel Size; It already calculates the derivative with the help of single matrix.

laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)

**We also need to do one more operation in Laplacian because since the edges are just changes in intensity, the sign positive and negative does not matter, but in Laplacian you will get points in negative direction as well as higher positive positive direction, but we only want in positive direction. So what we will do, we will make the absolute value of that okay. We will make the absolute. Because when once you plot Laplacian in its natural state, it will look a little weird.**

laplacian_abs = cv2.convertScaleAbs(laplacian)

**We want to do the abs the absolute of that for this Laplacian matrix. So we have two versions of Laplacian one with absolute, one without absolute.This one is having negative as well as positive values.**

**4. Canny Edge Detection:**

For canny there is also one more thing which we have mentioned.We need to mention the threshold one and two, which was mentioned in the last step of canny.So we need to mention first the higher threshold. It starts with lower and upper.So we need to mention the lower.

canny_edges = cv2.Canny(image, 180, 200)

**5. Next we plotted all the Images**

titles = ['Original', 'Sobel', 'Laplacian_all', 'Laplacian_abs', 'Canny']; image_list = [image, sobel_combined, laplacian, laplacian_abs,  canny_edges]

**This is Sobel Laplacian all looks all gray.But if we do the absolute of all because this has negative values as well as positive.So this might break our matplotlib, but when we do the Laplacian absolute it is able to perform better. So this might break our matplotlib, but when we do the Laplacian absolute it is able to perform better. We next increased fig size as 20x16**

**We are able to see lot of noise in Laplacian because we have converted those negative values to positives.And that might create lot of noise as we can see here.
So we can find out lot of different methods, like if you want to normalize the negative values or if you want to make it zero or something like that. But this also does the job. And this is what a canny object detection looks like. Canny edge detection looks like.You can see this is canny.So we can save this image.**

The best one we got is I think that is canny because the edges here are defined.But but there is no noise here because canny takes care of all the noise components.
But in Laplacian we have a lot of noise.In Sobel.Also we have noise.But Sobel.Considering the computation cost of Sobel, it is also given a very good features.

**M) Calculating and Plotting Histogram**

1. We are going to convert this to RGB here.So image underscore RGB because we want to plot a histogram of our RGB image. We will use cv2 dot CVT color and I'm going to pass the image. Then we will mention.From which color space we want to convert.We want to convert from BGR to RGB.And this will be our RGB image.

2. The next thing we have to do is to get histogram.We need to plot all the channels separately.So we have to first split our channel.So we are going to use cv2 dot split to split our channels.That is cv2 dot split.We want to split my RGB image.So it will be split into three sections.We are going to store into channels.
This is channel has total number of three different dimensions. And we're going to use it one by one. And the colors I am going to represent here as blue.Green and then red.And then for channel comma color in zip of channels comma color.So it will zip it and it will iterate.It will give me one value one by one.But if your if our image is RGB we need to have red first.It will have red first, Then we will have blue.Later this will be red.Channel will come first with the red color here.So channel color will come in single value.This should be channel colors.It is color.And let me mention here colors.

channels = cv2.split(image_rgb)
colors = ['red', 'green', 'blue']

3. We are going to plot histogram. We are going to get histogram values.We will calculate the histogram with calc hist.This is the function which we have to mention.
And here we have to pass the channel.So our channel is going to be coming here.So first channel will come red.And then we need to pass some values.We need to pass values like mask.So this will be zero.So it's an optional one.And then we have to pass.Total number of channels.Total number of images okay.So this will be our image because we are passing one by one right.So we have to pass the first channel the first R.So we are passing R here.Second thing we are going to pass is channels.List of channels which we want to use.We want to use only channel zero for grayscale because we are going to consider this as a grayscalebecause it is only one channel, whereas we are plotting histogram for r g b.So R is a single channel.So we want to use zero here because we are going to treat it as grayscale image.The next thing we are going to do is we have to mention none for mask.It is an optional one.It is used to calculate the histogram for a specific region.So we are going to use none.It means we are going to calculate histogram for the entire image.And the next argument is going to be histogram size.Our histogram size is going to be 256.Why?Because the pixel intensity values ranges from 0 to 256.And we want to see those intensity values.Next thing is range.The range of our histogram is going to be starting from 0 to 256.This is the range.And once this is done we have to store this histogram I will store this histogram into high SD and thenI will use plt dot plot.Here plt dot plot.And then I have to mention this histogram and then color I will mention in color.So it means when I get a histogram it will be plotted with a red color.The red means we are plotting for red channel.And we are getting this color here because we have mentioned already the color here, right?I will mention color here and that's all. So once this is done I want to show the plot.
So I will mention the title PLT dot title. I am mentioning it as color histogram and then plt dot x label as pixel intensity. In x we have pixel intensity.In y axis we have total number of those pixels.It is a frequency intensity of those pixels.I will mention plt dot y label and then it should be frequency.
This is frequency.And then I will mention plt dot show to show the plot.

for channel, color in zip(channels, colors):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)

plt.title("Color Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequence")
plt.show()

**Observations from Image:**

Starting with the blue channel, we can see that there are a lot of blue pixels present. Most of these blues fall in the lighter shades, as indicated by the high spike in the histogram. The highest spike for blue appears in the intensity range of 230 to 250, which corresponds to brighter hues of blue. This tells us that a large number of pixels in the image have high blue intensity values. Additionally, there are also lower-intensity blues present, which means darker shades of blue are also being used in the image.

Moving to the green channel, we notice that green is the second most dominant color. While we don’t see a strong presence of pure green, it mostly appears in combination with the other RGB components. The histogram shows a spike for green in the range of 170 to 200, indicating that these intensity values are frequently occurring in the image.

For the red channel, we can observe that red is present but to a lesser extent compared to blue and green. The intensity of red is not very high, generally falling in the range of 100 to 150, which corresponds to darker shades of red. These reds can be seen in areas like the horse, the giraffe’s patterns, and the grass.

Overall, this is the RGB analysis of the image. 

### However, if we want to understand more about the brightness and overall value of the image, we need to move beyond RGB and look at the HSV (Hue, Saturation, Value) representation, where the “Value” component specifically represents brightness.

4. **HSV Codes:**

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV); 
h, s, v = cv2.split(image_hsv)

h_hist = cv2.calcHist([h], [0], None, [180], [0, 180])

s_hist = cv2.calcHist([s], [0], None, [256], [0, 256])

v_hist = cv2.calcHist([v], [0], None, [256], [0, 256])

**Code Explanations:**

Now, instead of converting the image from BGR to RGB, we are going to convert it from BGR to HSV. This will give us the HSV version of the image.

The next step is to split this HSV image into its three separate channels: H (Hue), S (Saturation), and V (Value). Once we have these channels, we will calculate the histogram for each one separately. Unlike the RGB case, we are not going to use the red, green, and blue colors for plotting. Instead, we will assign specific colors for each histogram: orange for Hue, green for Saturation, and blue for Value.

For histogram calculation, the function cv2.calcHist() is the same, but the ranges differ.

For Hue (H):

The histogram requires 180 bins, since Hue in OpenCV is represented in the range 0 to 180 degrees.

Hue essentially corresponds to the type of color (e.g., red, green, blue, etc.), so its range is smaller.

For Saturation (S) and Value (V):

Both use 256 bins with a range of 0 to 256, just like pixel intensity values in RGB.

Saturation represents the intensity or purity of the color.

Value represents the brightness of the image.

Once we compute these histograms, we plot them using subplots: a single row with three columns. The figure size is set to 15 × 5, so all three histograms appear side by side. Each histogram is drawn on its corresponding subplot:

Hue histogram (plotted in orange, labeled “Hue Histogram”)

Saturation histogram (plotted in green, labeled “Saturation Histogram”)

Value histogram (plotted in blue, labeled “Value Histogram”)

The axes are labeled properly:

The x-axis represents Bins (intensity or range values).

The y-axis represents Frequency (number of pixels for each bin).

Finally, we use plt.tight_layout() to adjust the spacing and plt.show() to display the three histograms together.

This way, instead of analyzing only the RGB color intensities, we can now also understand the image in terms of Hue (color type), Saturation (color strength), and Value (brightness), which often gives a clearer understanding of the overall image properties.

**Observations:**

So, we are able to see this separately for the Hue histogram. In this, we can see that my hue range is showing a spike in the degree from 0 to 225. This indicates the presence of some kind of yellow and red color. We need to check on the color wheel to understand what this 100-degree hue represents, but we have observed that a lot of colors are coming from this 100-degree hue value.

If I talk about Saturation, which represents the saturation of that particular color, we can see that a lot of saturation is in the range of around 100. So, a saturation of around 100 is being used a lot in this picture — in this giraffe picture. We can also see many spikes in the range of 70 or 80 up to around 200. This means that the saturation values are concentrated in this range. They are not using saturation values like 0, 20, 30, 40, or around 250, but rather values within this specific middle range.

Now, if I talk about the Value channel, we can conclude that the image is not dark. Many of the pixels in our image are not dark. In the Value histogram, we see a spike starting from 100, and a lot of values lie in the range of around 220 to 240.

This provides important information about our image: the values are not evenly distributed, and the intensities are not spread out completely. The color saturation that is being used is not too bright and not too dark, but it still leans slightly toward the darker range. If the Value were consistently around 200 or 250, we would see higher saturation. But here, the bins show that we are not using too many different colors — instead, we are mainly using just two different kinds of colors within the hue wheel of 0 to 180 degrees.

So, this gives us a lot of information. We can directly conclude from this whether our image is dark or not. In this case, our image is not dark, because the majority of the pixel values are in the higher range. As a rule: if the majority of pixel values are greater than 150, the image is considered bright. If the majority of pixel values are smaller than 150, the image is considered dark.

Therefore, we can derive a lot of useful information just from a histogram. Histograms are incredibly versatile, as we can see, and they serve as the foundation of many advanced image processing techniques, such as contrast enhancement and thresholding, which we are going to study in future lectures. In fact, we will later study a technique called CLAHE (Contrast Limited Adaptive Histogram Equalization), which we will explore in more detail in the upcoming sessions.

**N) Image Segmentation**

The first image segmentation technique we are going to implement is thresholding. To begin, we start with importing the necessary libraries: cv2 for computer vision, numpy as np for numerical operations, and matplotlib.pyplot as plt for visualization. Once the imports are successful, we first implement simple thresholding.

We begin by reading an image using cv2.imread(). The path provided points one directory back inside an images folder, and the specific image we are using is handwritten.jpg. Since we want to process the image in grayscale, we pass the flag 0 while reading, so that the image is directly loaded as grayscale instead of converting it later. After loading, we apply the thresholding operation using cv2.threshold(). This function requires the grayscale image, a threshold value (in our case 150), and a maximum value (255). What this means is that all pixel values above 150 will be reassigned to 255, while values below 150 will remain unchanged. The function returns two values: a return code and the binary mask, but since we are only interested in the binary mask, we store that and ignore the other return value.

Next, we visualize both the original image and the thresholded binary mask using matplotlib. We create a figure with a size of 12x7, plot the grayscale input image using cmap="gray", and then plot the binary output image similarly. This helps us see the difference between the raw image and the mask generated by thresholding. At first, if we forget to mention the type argument in cv2.threshold(), we encounter an error. This argument specifies the kind of thresholding. For example, cv2.THRESH_BINARY applies the rule that values above the threshold become 255, while cv2.THRESH_BINARY_INV inverts this behavior so that values below the threshold become 255 instead. Once we include cv2.THRESH_BINARY, the mask is correctly generated, where all pixel values above 150 are assigned white (255), while the darker pixels remain unchanged.

After understanding simple thresholding, we implement adaptive thresholding to see the difference between the two approaches. Since we already have a grayscale image, we directly apply cv2.adaptiveThreshold(). This function requires the grayscale image, a maximum value (255), the type of adaptive method (either cv2.ADAPTIVE_THRESH_MEAN_C or cv2.ADAPTIVE_THRESH_GAUSSIAN_C), the threshold type (we use cv2.THRESH_BINARY), and two parameters: a block size and a constant C. The block size (e.g., 11) defines the neighborhood region used for threshold calculation, while the constant (e.g., 9) is subtracted from the mean or weighted mean. We implement both mean and Gaussian methods, storing the outputs as adaptive_m and adaptive_g respectively.

To compare results, we plot both outputs using matplotlib with titles “Adaptive Mean Thresholding” and “Adaptive Gaussian Thresholding”. When visualizing, we notice that the mean thresholding output still shows some unwanted background noise, such as faint marks from pages behind the current one. In contrast, Gaussian adaptive thresholding produces a cleaner result with fewer artifacts. By experimenting with the constant C (e.g., changing it to 2, 9, or 15), we can observe how noise and edge sharpness vary. For example, using C=15 results in a much cleaner image with less visible background artifacts, although the edges may look slightly softer compared to mean thresholding. Thus, adaptive thresholding, especially the Gaussian method, often provides a better segmentation result compared to simple thresholding.

Once thresholding is complete, we move to the next segmentation technique: K-means clustering. Unlike thresholding, which works on grayscale, we will apply K-means on an RGB image. For this, we read the image shapes.jpg using cv2.imread(), then convert it from OpenCV’s default BGR format to RGB using cv2.cvtColor(image, cv2.COLOR_BGR2RGB). Next, we reshape the image so that every pixel is represented as a row with three columns (R, G, B values). This flattening process transforms the image into a 2D array of pixel values which we store as pixel_values. Since OpenCV’s K-means algorithm requires floating point input, we convert the pixel values from uint8 to float32 using np.float32(pixel_values). Printing this confirms that each pixel is now represented as floating-point RGB values.

To clarify, the reshaping ensures that the image is flattened into rows of [R, G, B] triplets, where each triplet corresponds to a pixel’s color. With this prepared data, we define our K-means parameters, including the termination criteria: a combination of cv2.TERM_CRITERIA_EPS and cv2.TERM_CRITERIA_MAX_ITER, with a maximum of 100 iterations and an epsilon of 0.2. This defines when the algorithm should stop iterating. Finally, we feed the pixel values into K-means clustering, which will allow us to segment the image based on color similarities.

Once the pixel values have been prepared, the next step is to define the criteria for the K-means clustering algorithm. The criteria define when the algorithm should stop running. We enclose them inside a tuple: the first element specifies the type of stopping rule (a combination of maximum iterations and epsilon), the second is the maximum number of iterations, and the third is epsilon (the minimum accuracy change required between two iterations). In this case, the stopping criteria were defined as (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2). This means the K-means clustering will stop if it either reaches 100 iterations or if the difference in cluster accuracy between two iterations is ≤ 0.2.

Next, the number of clusters (k) was defined as 3. This ensures the image will be divided into three distinct categories. With both the criteria and cluster count set, the K-means function was applied using cv2.kmeans(). The function takes as input the pixel values, number of clusters, placeholders (set to None in this case), the stopping criteria, the number of attempts (set to 10, meaning it will run 10 times with different initializations and return the best result), and finally the initialization method (cv2.KMEANS_RANDOM_CENTERS, which randomly chooses initial cluster centers). Running this function returned two key outputs: labels (which assign each pixel to a cluster) and centers (the RGB values of the cluster centroids).

After printing the outputs, it was confirmed that each pixel in the image had been assigned a cluster label, and the centers contained three distinct RGB values representing the cluster centroids. To reconstruct the segmented image, the cluster centers were first converted to np.uint8, ensuring proper pixel value format. Then, using the labels, each pixel was reassigned the RGB value of its corresponding centroid. The resulting flat array of segmented pixel values was reshaped back to the original image dimensions, forming the segmented image. Finally, both the original and segmented images were displayed side by side using Matplotlib. The result showed the input image segmented into three categories. Although visually two dominant colors (orange and blue) were visible, the third cluster corresponded to white regions, making it an important part of the segmentation.

The next algorithm implemented was the Watershed algorithm, particularly effective for separating overlapping objects such as coins. The process began by reading the input coin image and converting it from BGR to RGB. A copy of the image was also stored separately for later use. After visualizing the image, it was converted to grayscale using cv2.cvtColor().

The first preprocessing step was to apply binary thresholding. Using cv2.threshold(), pixels above a chosen value (130) were set to white (255), and those below were set to black (0). Both the normal binary and inverse binary thresholding were tested, with the inverse being chosen for better clarity. This produced a base mask of the image, highlighting the coins against the background.

Next, morphological operations were applied to remove noise. A kernel (np.ones((3,3), np.uint8)) was defined, and cv2.morphologyEx() with the cv2.MORPH_OPEN operation was used to clean up the image. Running the operation with different iterations showed how increasing iterations removed more noise but could also erase useful details. A balanced value (around 4 iterations) produced a clean mask where the coins appeared distinct without losing detail.

With noise removed, the sure background was extracted by applying cv2.dilate(), which expanded the white areas. This provided regions that were confidently background. For the sure foreground, a distance transform was applied using cv2.distanceTransform(). This computed the distance of each pixel from the nearest background pixel, effectively highlighting the centers of objects. The distance map was then thresholded at 70% of the maximum value, ensuring only strong foreground regions were kept. This step guaranteed the algorithm could differentiate between definite objects and background.

Finally, the sure foreground and sure background were combined to identify the unknown regions (areas not confidently classified as either). These markers were passed into the Watershed algorithm, which then treated the grayscale image like a topographic surface, “flooding” it until boundaries were clearly established between objects. The result was an image where overlapping coins were separated by distinct boundaries, demonstrating the effectiveness of the Watershed approach in complex segmentation tasks.

After calculating the sure background, the next step was to determine the sure foreground. This was achieved by applying a distance transform (cv2.distanceTransform) on the cleaned binary mask, followed by thresholding it. By experimenting with different threshold values (0.7, 0.8, 0.4), it was observed how the regions of the sure foreground expanded or shrank. At higher values, only the most certain areas were included, while lowering the threshold revealed more parts of the objects. The key observation was that the foreground (black coins) and background (white regions) followed the expected pattern, confirming that the process correctly isolated the sure foreground.

Once the sure foreground was obtained, the unknown region was computed. This represents the ambiguous areas where the algorithm is uncertain whether pixels belong to the foreground or background. It was calculated by subtracting the sure foreground from the sure background (cv2.subtract). When plotted, the unknown region appeared as gray areas between foreground and background, visually highlighting the uncertain boundaries. These unknown areas later played a critical role in marker generation for the watershed algorithm.

The next step was to create markers for the watershed process. Using cv2.connectedComponents on the sure foreground, connected objects were assigned unique integer labels. These labels were then incremented by one (markers = markers + 1) to avoid confusion with background pixels. In addition, wherever the unknown region contained white (255), the corresponding marker value was set to zero. The result was a labeled marker map, where different colors represented different connected components, ready to be passed into the watershed algorithm.

The watershed algorithm was then applied using cv2.watershed(image, markers). The algorithm treated the grayscale image as a topographic map, “flooding” it from the markers until all boundaries were clearly defined. The output highlighted catchment basins and distinct separating lines between objects, making the overlapping coins clearly segmented.

At this point, although segmentation was successful, additional filtering was required because not all detected components were coins. To refine results, a circularity check was introduced. First, a blank mask was created using np.zeros_like() with the same shape as the grayscale image. The unique labels from the markers were extracted, and for each label, a separate binary mask was generated. Using cv2.findContours, contours of each labeled region were identified. For each contour, both area (cv2.contourArea) and perimeter (cv2.arcLength) were calculated. Circularity was then computed using the formula:

Circularity
=
4
𝜋
×
Area
Perimeter
2
Circularity=
Perimeter
2
4π×Area
	​


If the perimeter was zero, the contour was skipped. Otherwise, circularity values close to 1 indicated near-perfect circles. A filtering condition was applied: circularity between 0.7 and 1.2 and area greater than 1000. This ensured only contours that were sufficiently circular and of meaningful size were kept, effectively filtering out non-coin regions.

For each valid coin, the following steps were applied:

The contour was drawn on the original image (cv2.drawContours).

A bounding rectangle was added (cv2.boundingRect).

A label such as “Coin 1”, “Coin 2”, etc. was written above each coin using cv2.putText.

A final coin mask was created using bitwise operations, highlighting only the detected coins.

Finally, the results were visualized in three plots side by side:

The original RGB image.

The labeled image with bounding boxes and coin numbers.

The final coin mask containing only the segmented coins.

The output confirmed that six coins were correctly identified and labeled, while other non-circular objects were excluded. This demonstrated that with careful preprocessing, OpenCV’s watershed algorithm combined with circularity checks could achieve instance-level segmentation without requiring deep learning.

**O) Haar Cascade for Face Detection**




### (II) Pytorch

**A) Introduction to Pytorch**

To install PyTorch, the first step is to visit the official PyTorch website at https://pytorch.org
. This page provides all the latest versions, tutorials, and installation instructions. On the installation page, you need to select your preferences carefully: choose the version (stable or preview), your operating system (Windows, Linux, or Mac), the package manager you want to use (Conda, Pip—which is recommended by the mentor—or LibTorch/Source), the programming language (Python, C++, or Java), and the compute platform. For CPU systems, no extra steps are needed, but if you have a GPU, you must select the correct CUDA version that is compatible with your GPU.

To check your GPU and CUDA version, open the terminal and run the command nvidia-smi (for NVIDIA GPUs). This will display GPU processes, available GPU RAM, driver version, and the installed CUDA version. It is important to ensure that the CUDA version is compatible with the PyTorch build. For example, PyTorch may require CUDA 12.4, while your system has 12.7, in which case you may need to downgrade CUDA if PyTorch cannot access the GPU. Once all selections are made, PyTorch generates a command for installation. For example, a typical pip command might be pip3 install torch torchvision. You can remove torchaudio if it is not required. Note that the GPU version of PyTorch is around 2.5 GB in size, while the CPU version is smaller and does not include CUDA.

For local installation, it is recommended to create a dedicated conda environment. This can be done with the command conda create -n <env_name> python=3.11, then activating it using conda activate <env_name>. After activating the environment, run the PyTorch installation command. To verify the installation, open a Python shell within the environment and run import torch. If there are no errors, PyTorch is installed successfully. If you installed the GPU version, you should also check GPU availability using print(torch.cuda.is_available()), which returns True if the GPU is accessible. If it returns False, it usually indicates either the CPU version is installed or there is a CUDA mismatch.

If setting up PyTorch locally is complicated, you can use Google Colab as an alternative. After signing in to your Google account, create a new notebook and click Connect. You can check the available RAM, disk space, and compute units. To access GPU, go to Change Runtime Type and select a GPU such as Tesla T4. PyTorch is pre-installed in Colab, so you don’t need to install it manually. To verify GPU access in Colab, run import torch and torch.cuda.is_available(). If running in a CPU instance, it will print False, but after switching to a GPU runtime, it should print True. Note that changing the runtime refreshes the session, so RAM and disk space may change, but your code remains saved in the notebook.

### import torch - This shows if Pytorch is already installed in Colab

In summary, PyTorch can be installed locally or used via Colab. For local installation, pay attention to OS, package manager, Python version, and CUDA version if using GPU. GPU setup requires checking CUDA compatibility. Verification is done via import torch and torch.cuda.is_available(). Google Colab provides a simpler alternative with pre-installed PyTorch and free GPU access, making it easier to get started without local setup hassles. 

**B) Introduction to Tensors**

In this lecture, we will understand one of the most fundamental concepts in PyTorch, which is tensors. Tensors are multidimensional arrays similar to NumPy arrays but with additional capabilities, such as running computations on GPUs. Tensors are used to represent data in PyTorch, ranging from simple numbers to complex multi-dimensional datasets. We start by importing PyTorch using "import torch". First, we create a zero-dimensional array, which is called a scalar tensor in PyTorch. A scalar tensor is analogous to a single variable in Python, for example, "a = 2". To create a scalar tensor in PyTorch, we write "scalar = torch.tensor(42)", and printing "scalar" shows the tensor value. The dimension can be checked using "scalar.dim()", which will return 0, and the shape using "scalar.shape", which is empty because it is zero-dimensional.

Next, we create a one-dimensional tensor, also called a vector. To do this, we pass a list of values to "torch.tensor", for example, "vector = torch.tensor([1, 2, 3])". Printing "vector", its dimension using "vector.dim()", and shape using "vector.shape" shows that it is a one-dimensional tensor with size three. Moving on, we create a two-dimensional tensor, also known as a matrix, by passing nested lists to "torch.tensor", such as "matrix = torch.tensor([[1, 2], [3, 4]])". Printing "matrix", checking its dimension using "matrix.dim()", and shape using "matrix.shape" reveals a 2x2 matrix. For tensors with more than two dimensions, there is no specific name, and they are simply called tensors. For example, a three-dimensional tensor can be created using nested lists: "tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])", which results in a tensor of shape 2x2x2 and dimension 3.

**One of the key features of PyTorch compared to NumPy is the ability to perform computations on GPUs. To use GPU acceleration, we first check if CUDA is available using "torch.cuda.is_available()". If available, we can move a tensor to the GPU using "gpu_tensor = vector.to('cuda')". Printing "gpu_tensor" will show an additional argument "device='cuda:0'", indicating that the tensor is stored on the GPU. The device number reflects which GPU the tensor is on, such as cuda:0, cuda:1, etc.**

PyTorch tensors have several properties we can inspect. The data type of a tensor can be checked using "matrix.dtype", which will be torch.int64 if the tensor contains integers. If we create a tensor with floating-point values, for example, "matrix = torch.tensor([[1.2, 2.3], [3.4, 4.5]])", its dtype changes to torch.float32. The device where a tensor resides can be checked using "matrix.device", which returns cpu for CPU tensors and cuda:0 for GPU tensors. To verify if a tensor is on the GPU, we can use "matrix.is_cuda", which returns a boolean value. The shape of a tensor can also be checked using "matrix.shape", which internally calls "matrix.size()". Finally, to find the total number of elements in a tensor, we use "matrix.numel()", which in the case of a 2x2 matrix returns 4.

In summary, this lecture covered an in-depth exploration of tensors in PyTorch, starting from scalar tensors, moving to vectors, matrices, and higher-dimensional tensors, demonstrating how to check their dimensions, shape, dtype, and device. We also explored GPU acceleration, showing how to move tensors to CUDA-enabled devices and verify their presence on the GPU. This forms the foundation for working with data and computations in PyTorch.

**Tensor operations exist in both PyTorch and TensorFlow — they are not exclusive to one. Both frameworks provide powerful tensor libraries that allow you to create, manipulate, and perform computations on tensors; Pytorch uses torch.tensor and Tensorflow uses tf.tensor**

**C) Indexing Tensors**

## Accessing Elements in a Tensor - tensor[row, column]

In this lecture, we explore indexing and slicing in PyTorch tensors, which refers to selecting specific elements or a range of elements from a tensor. PyTorch tensors allow for indexing and slicing using syntax very similar to Python lists or NumPy arrays, but with additional flexibility for multi-dimensional data. To access elements in a tensor, we first import PyTorch using "import torch", and then we create a 2D tensor, for example, "tensor = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])". The dimension of the tensor can be checked using "tensor.dim()", and its shape using "tensor.shape", which in this case is 3x3. To access a specific element, the syntax is "tensor[row, column]", for instance, "tensor[0, 1]" accesses the element in the zeroth row and first column, returning the scalar value 20. To access an entire row, we can use "tensor[0]", giving the first row [10, 20, 30], and to access a column, for example the first column, we can use "tensor[:, 0]", which returns all rows of the zeroth column [10, 40, 70].

ndim = 2 → 2D tensor (like a matrix) (2 - 1 Row and 1 Column)

shape = (3,3) → 3 rows × 3 columns

print("Element at (0, 1): ", tensor[0,1]) - Output 20; Accessing First Row - tensor[0] - [10,20,30]; Accessing first column - tensor[:, 0] - tensor([10, 40, 70])

**Slicing:**

For slicing, we can select a range of rows or columns. For example, "tensor[:2]" returns the first two rows, giving [[10, 20, 30], [40, 50, 60]], and to get the first two columns, we write "tensor[:, :2]", which outputs [[10, 20], [40, 50], [70, 80]]. To select a specific range of elements, such as row 1 and columns 1 and 2, we write "tensor[1, 1:3]", which returns [50, 60]. Beyond sequential access, we can perform fancy indexing to access non-sequential elements. For example, to extract 20 and 90 from the tensor, we use "tensor[[0, 2], [1, 2]]", which returns the desired elements. PyTorch also supports Boolean indexing, where we can create a mask, for instance, "mask = tensor > 50", which gives a tensor of True and False values depending on whether the elements satisfy the condition. To retrieve elements meeting the condition, we can use "tensor[mask]", returning [60, 70, 80, 90].

print("First two rows : \n", tensor[:2]) --> First two rows : tensor([[10, 20, 30],[40, 50, 60]])

print("First two columns : \n", tensor[:, :2]) -->  tensor([[10, 20],[40, 50],[70, 80]])

Middle elements (row 1,column 1 and 2 only) - print(tensor[1, 1:3])

**Fancy Indexing - Getting 20,90 ((0,1) --> 20; (2,2) --> 90)** - tensor[[0,2],[1,2]]

**Boolean Indexing - mask = tensor > 50; print(mask); print(tensor[mask]) - Values >50 will alone be True (Where condition is True)**

Changing Tensor Value via Indexing - tensor[0,1] = 25

**Keep all the rows, but only change column values - tensor[:,0] = torch.tensor([100,200,300])**

**Indexing and selecting columns from a Tensor - selected_rows = torch.index_select(tensor,dim=1,index=indices)**

**Prints all rows and columns of step 2 - print(tensor[:, ::2])**

We can also change tensor values via indexing. For example, to update a single element, we write "tensor[0, 1] = 25", replacing 20 with 25. To update multiple values at once, for example, all rows of the zeroth column, we can write "tensor[:, 0] = torch.tensor([100, 200, 300])", resulting in the first column being updated to [100, 200, 300]. For advanced slicing, PyTorch provides "torch.index_select". We first define an index tensor, e.g., "indices = torch.tensor([0, 2])", and then select rows using "torch.index_select(tensor, dim=0, index=indices)", returning rows 0 and 2. This method can also be applied to columns by setting dim=1. Additionally, tensors can be accessed with steps using slicing syntax like "tensor[:, ::2]", which selects every second column. To reverse a tensor, unlike NumPy where "tensor[::-1]" works, PyTorch requires "torch.flip(tensor, dims=[0])" to flip rows or "torch.flip(tensor, dims=[1])" to flip columns. This flips the tensor along the specified dimension while preserving the rest of the structure.

### Reverse a Tensor: [::-1] works in NumPy but Tensorflow we need to use - torch.flip(tensor, dims=(0,))

In summary, indexing and slicing in PyTorch tensors allows for flexible selection, extraction, and updating of elements, whether through sequential access, ranges, fancy indexing, Boolean masks, or advanced operations like torch.index_select and torch.flip. These operations provide powerful ways to manipulate multi-dimensional data efficiently, forming the foundation for deeper tensor manipulations and computations in PyTorch. This concludes the session on indexing tensors in PyTorch.

**D) Using Random Numbers to create Noise Image**

In this lecture, we are going to learn how to get random numbers with the help of PyTorch. Randomness is a critical aspect of many machine learning workflows, such as initializing neural network weights or parameters, or simulating real-world variability since real-world features are generally random in nature. In this lecture, we will do something fun with random numbers: we will create a noisy image from random numbers. First, we start by importing PyTorch using "import torch". To create a random vector of values in the range 0 to 1, we use "torch.rand(5)", which generates five random values and stores them in "random_tensor". We can print this tensor using "print(random_tensor)" and check its dimension using "random_tensor.dim()", which will return 1, meaning it is a vector, not a scalar.

**Creates 2 of 5x3 Matrix - random_tensor = torch.rand(2, 5, 3) (5 Rows, 3 Columns)**

**For Reproducibility - torch.manual_seed(1)**

# Generating Random Numbers formet - torch.randint(low, high, size)

**Generating Random Numbers - random_integers = torch.randint(-10,10,(5,5))**

# Generate random numbers from a normal distribution - torch.normal(mean, std, size)

**Generate Random Numbers from a Normal Distribution - random_numbers_normal = torch.normal(0.5,2, (5,))**

To create a higher-dimensional tensor, such as a 3D matrix, we can pass a shape instead of a single number, for example: "random_tensor = torch.rand(2, 5, 3)", which generates a tensor with two 5x3 matrices. Checking the shape with "random_tensor.shape" confirms it is (2, 5, 3). To ensure reproducibility of random numbers, we set a seed using "torch.manual_seed(42)". With the seed set, rerunning "torch.rand(5)" will always produce the same vector. Changing the seed will produce a different vector, but consistently the same vector each time with that seed. This is especially useful when working with multiple teams or repositories to ensure reproducible results.

To generate random integers, we use "torch.randint(low=0, high=10, size=(5,))", which gives a vector of five integers from 0 to 9 (high is exclusive). For a 2D tensor, we can use "torch.randint(low=0, high=10, size=(5, 5))", which creates a 5x5 matrix. The range can also include negative numbers, e.g., "torch.randint(low=-10, high=10, size=(5, 5))", giving values from -10 to 9. For generating random numbers from a normal distribution, we use "torch.normal(mean=0.5, std=2, size=(5,))", which produces a tensor of five values with mean 0.5 and standard deviation 2, stored in "random_numbers_normal".

**Using Random Number to create Noise Image:** 

uniform_noise = torch.rand((256,256)) **h,w of image 256x256**

plt.imshow(uniform_noise.numpy(), cmap='grey')
plt.title("Uniform Noise Image")
plt.axis('off')
plt.show()


Next, we use these random numbers to create noisy images. We start with matplotlib using "import matplotlib.pyplot as plt". To create a grayscale image, we define the image dimensions, e.g., 256x256, and generate a random tensor using "uniform_noise = torch.rand(256, 256)". To display the image, we use "plt.imshow(uniform_noise, cmap='gray')", add a title with "plt.title('Uniform Noise Image')", remove the axis with "plt.axis('off')", and show the plot using "plt.show()". This produces a black-and-white noisy image similar to static on old televisions. Running "torch.rand" repeatedly generates different noise, which can be used to create animated noise videos.

To create an RGB noise image, we modify the tensor to have three channels: "uniform_noise_RGB = torch.rand(256, 256, 3)". We can convert this PyTorch tensor to NumPy using "uniform_noise_RGB.numpy()" before plotting, as many libraries like Matplotlib and OpenCV work with NumPy arrays. RGB images do not require the cmap parameter when plotting.

**Creating an RGB Noise Image** - uniform_noise_rgb = torch.rand((256,256,3))  (h,w,c of image 256x256X3)

plt.imshow(uniform_noise_rgb.numpy())
plt.title("Uniform Noise Image RGB")
plt.axis('off')
plt.show()

We can also inspect properties of these tensors. To check the shape, use "uniform_noise_RGB.shape", which will return (256, 256, 3). To check the data type, use "uniform_noise_RGB.dtype". The minimum and maximum values of the tensor can be obtained with "torch.min(uniform_noise_RGB)" and "torch.max(uniform_noise_RGB)", which for uniform random numbers range between 0 and 1. Since "torch.rand" generates values from a uniform distribution in the interval [0,1), the tensor values will always lie between 0 and 1.

This concludes the session on generating random numbers and creating noisy images using PyTorch, covering vector, 2D, 3D random tensors, reproducibility with seeds, random integers, normal distribution, and creating both grayscale and RGB random noise images.

**E) Tensors of Zero's and One's**

In this lecture, we are going to learn how to create tensors filled with zeros and ones. These tensors are the backbone for initializing models, placeholders, or testing data in PyTorch. Let's learn how to create them. We will start by creating a tensor filled with zeros. Let's get started. I will write "import torch". And the next thing is we need to mention the size of zeros which we want. So I will write "torch.zeros". This is the method which we have to use and need to mention five because we want a total number of five values in our 1D vector. So this will give us a 1D vector. And 1D vector is a tensor. If I mention here as zeros tensor and let's try to print the same. So I will mention "zeros_tensor" and we will be able to see all the values are filled with zero. All the values in our 1D tensor are filled with zeros.

Now what if I want to create a multidimensional tensor of zeros? Same, but we have to mention a different shape. So if I want to create multiple dimensions, I will mention "torch.zeros" and then I will mention "256, 256". And I can store again in zeros tensor and we can print it. So let me print "zeros_tensor" and we can also print the dimension, or we can print the shape. And here we can see the torch shape is "256, 256". And if you want to see this tensor visually, we can use our matplotlib because this is anyways just a matrix and it's of dimension two. You can see it is of dimension two. If you want to see the dimension we can also print that here "zeros_tensor.ndim". And we can see the dimension of this particular tensor is two. So if it is a two-dimensional tensor, the least we can do is we can print, we can show the grayscale image out of it.

plt.imshow(zeros_tensor.numpy(), cmap='grey',vmin=0, vmax=1) - Expected Minimum and Maximum values we need to give

So we can first import matplotlib. We will mention "import matplotlib.pyplot as plt" and we will use "plt.imshow". Then our tensor which is "zeros_tensor.numpy()" and then we have to mention our "cmap='gray'". One thing we need to make sure before using matplotlib for visualization and using tensor is when using matplotlib, we have to mention the minimum and maximum value out of it. So if you don't mention it, by default your matplotlib will try to normalize in the range of zero and one. And for this image, because the value is very low, it is zero. Even if the value is one, the image will look black. But if you don't want to normalize automatically, if we want to give static values by ourselves, we need to mention a few more arguments like "vmin" and "vmax". What is the maximum value? What is the minimum value we are expecting from this tensor, and what is the maximum value we are expecting from this particular tensor? That is a numpy array. First it will be tensor, then we will convert to numpy and we are mentioning the value. The value we are getting is in the range of zero and one. So this is already the normalized range which we are giving. It means if it is creating zero, it means it is black. If you are passing one here, it will be all white because the maximum value it can go to is one. So this image will be all white. Right now we are only doing zero, so we will be able to see that particular plot. If I want to run it, I will mention "plt.title('Dark Image')". Then I will mention "plt.axis('off')" and then "plt.show()". And if you do that, you will be able to see this dark image. And the title will show as "Dark Image" as well.

**Tensors of One's** - ones_tensor = torch.ones(5); ones_tensor = torch.ones(256,256)

So now let's try to create tensors of ones. Once we have created a tensor of zeros with a different dimension, we have created a vector, we have also created a 2D matrix, and now we have also visualized our 2D matrix as well. Now let's try to create tensors of ones. The same syntax will follow, but instead of "torch.zeros" it will just follow "torch.ones". "Torch.ones". And I will store into "ones_tensor", just like we were storing in zeros tensor before. And we can print the same. And we can see all the 1D vector which we have here has all the values of one.

Now we can do the same just like we did for zeros. But instead of creating values with the help of "torch.zeros", we will be creating these values with the help of "torch.ones". We can store these values in "ones_tensor". We can check the attributes, the properties of our tensor. And we can see all the values are filled with one, the dimension is two, and the shape is "256, 256". You can copy this same code and then try to plot this particular 2D matrix. Instead of zero tensors, we will be mentioning "ones_tensor". And because again it's a grayscale image and the dimension is two, we are mentioning "vmin=0" and "vmax=1". So if our "vmax=1", it will show us everything as white because white is the maximum it can go. For an image, the brightest pixel is the white, or the brightest intensity in a grayscale image is one when we normalize the value from zero to one. Let's try to run this and let me mention "plt.title('White Image')", "plt.axis('off')" and then "plt.show()". And that's it. We are able to see this white image. There is no boundary here because everything looks white. The background is by default white. Previously we were trying to plot black there. Now we are trying to plot white on a white background. That's why this whole image, apart from this title, looks white.

And that is all for this lecture on tensors of zeros and ones.

**F) Tensor Data Types:**

1. floating -- for most deep learning tasks ---> float32, float64, float16

2. integers - for categorical data and indices --> int32, int64, int8

3. booleans - mask or logical operation

4. Complex number - advanced computation --> complex64, complex128

(i) memory consumption : float16 << float32 << float64

(ii) computation : lower precision will be faster on gpu

(iii) numerical precision : float64 is more precise than float32

In this lecture, we are going to understand different data types available in PyTorch and see their impact on performance, precision, and memory usage as well. Let's get started. So there are different kinds of data types available for PyTorch for tensors which we can see here. Let me write data types for your tensors. So I'm going to write base data type, base data type types like floating, integer, boolean, and complex numbers. For floating, if I write about floating, floating is used for most deep learning tasks. So let me write most of the deep learning task. For most deep learning task. If I talk about integer, so integer is used for categorical data or if I want to mention indices or indices of my tensor, or you can say list, there we use integers. So we use integers for categorical data and indices. Let me mention all of this in a docstring. Now this looks better. So I'm going to write all of this information here so that we can refer this notebook later.

If I talk about boolean values, these are used to create mask or logical operations. And the final one is complex numbers. So these are used for advanced computation. Okay, so just like complex64, complex128 I will talk about it later. So choosing the right data types impacts your performance. So if you're going to choose in that floating itself. Now let's deep dive into floating and what are the different values available for torch in PyTorch, okay, for tensor in PyTorch. So I'm going to write we have float32 as well, that is a precision of your float. And we also have float64. But we also have lower categories like float16 as well. And for our tensors we have different categories. We will talk about it when we are going to write code for that.

For integers we have int32, we have int64, we have int8 as well. For Boolean it is just boolean consisting of true and false. And for complex also we have multiple categories like complex64, complex128 as well. So if I talk about float32, float64, and float16, the memory consumption out of all of these three present here, the memory consumption for float16 will be less than float32 and float64. So that will also impact your performance. Because if you're going for a bigger precision, the calculation it will take, the memory it will take will be larger than float16 and other values. Right. Because if I talk about 32, 32 will be taking more memory consumption than float16. And also the speed: lower the precision, faster it will be on speed.

Let me write a few key points here. For memory consumption, float16 will use less memory than float32, which will use less memory than float64. And if I talk about computation, we will write lower precision will be faster on GPUs. And the final one is numerical precision. So if I talk about float64 here from our example, float64 is more precise than float32 because it can carry more numbers after decimal. So based on all of these factors, if you have to choose for our algorithm, for our operation, which one to pick, either to pick float32, float16, int32, based on the limitations of memory consumption, computation, and the precision we also need. These are the main two factors, but based on the precision we have to select, and we have to do the tradeoff whether we want a higher memory consumption or higher computation or a less precision model.

So let's start by creating tensors with different data types. So first we will start by importing torch. And we will create our basic tensor. So I'm going to write "torch.tensor([1.5, 2.5, 3.5])". This is our dummy tensor. And instead of this, there should be a comma here. And we are just mentioning it as default tensor. The default data type for PyTorch tensor is float32. If you check the data type for this particular tensor, it will be float32. Let me print "default_tensor.dtype" and it should print float32. You can see this is the default data type for our tensor.

### Default Type of Pytorch Tensor is float32 - default_tensor = torch.tensor([1.5, 2.5, 3.5]) (We got output as Float32)

### To specify specifically float64 - float_tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)

### Converting Float to int - int_tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.int32); It will remove the decimal points - tensor([1, 2, 3], dtype=torch.int32)

### Default for int is int64 - int_tensor = torch.tensor([1, 2, 3])

And if you want to specify our data explicitly, if you want to explicitly specify our dtype, that is data type of our tensor specifically, so what we have to do is copy this code and mention float tensor. So we are going to specifically mention that this is a float tensor. We will not depend on the default data type. So we will just mention here that the tensor which we are creating is of type float64 because the default is 32. So I'm just going to mention float64. So we can see what is happening here. Float tensor. If I run it, you will be able to see that now the data type of this particular tensor is "torch.float64".

Or we can also create multiples. Like let me copy this, paste it here, and instead of torch float64 we can mention int32. So right now you can see the value is 1.5, 2.5, 3.5. Let's see if we get any error if we are trying to convert these floating values. Because these are decimal values and we are trying to convert to int32. So let's try to see that. So we are going to mention int tensor. If it gives any error we will convert it back to one, two, and three. You can see this is now converted to "torch.int32". And if you want to see the values that are stored in this int tensor now, yeah, so it has removed our decimal points and kept only the digits, that is one, two, and three.

But we can also mention a different data type. So if we mention just let me write it here. So if we mention one, two, and three and don't mention any data type, you can see the default for all the integer values is "torch.int64". So it is storing one, two, and three, but it is the default one here. Let's try to create a boolean one. So if I want to create a boolean one I will just mention "torch.tensor([True, False, True])". And if you want to mention specifically, otherwise it will be taken according to the value which is present in our tensor, it will be "torch.bool", that is a boolean. And we will store this into bool tensor. And we'll see the same components here which we were writing for previous tensors. This is bool tensor. That is a data type we want to print. And we also want to print the tensor itself. And we can see this is "torch.bool" and this is the tensor. So if we remove this "dtype", by default it should also take the boolean. You can see this is by default taking as boolean. But if you want to explicitly mention that what kind of data type you want, this is the format we should go with.

### Creating a Boolean Value - bool_tensor = torch.tensor([True, False, True], dtype=torch.bool); print(bool_tensor.dtype) --> torch.bool

### Converting Flot to int - float_tensor = torch.tensor([1.5, 2.5, 3.5]); int_tensor = float_tensor.to(torch.int64)

### Converting int to boolean - int_tensor = torch.tensor([0,1,2,0, -1]); bool_tensor = int_tensor.to(torch.bool) [Any value 0 - False; Other than 0 - True]

Now let's talk about how to convert data from one data type to another data type. Okay. So what we are going to do first, we are going to take this tensor. This is our tensor. And I'm going to remove this value. So it will be float32 values. If you want to check that what we will do, we will copy this. And it will be a float tensor because the value inside it is float. But the precision of that particular tensor, either it is 32, 64, 16, will be decided by default. You can see that is 1.500000… This has been added after this value and the default is float32. So this is my original tensor. And I want to convert this tensor to integer tensor. So what I will write, I will mention float tensor. Then I have to mention just simple command that is ".to()" and the data type which I want to convert to. So I want to convert this to "torch.int64". Let's convert to int64. And we will store into int tensor. And let's not forget to print the details of this tensor. We will mention int tensor. And we will also mention int tensor dtype. So we can see the original is float32 and these are the values when we use this ".to()" function which will convert our original data type to the data type which we have mentioned here, and it will convert to this particular tensor.

We can also convert this to boolean as well. That is our int tensor and we want to convert to boolean. So let's see what will happen here. It will not make sense but it will provide us all the zero values. So anything that is greater than zero should be true. Let's create a new int tensor here with different values. So we will write "torch.tensor([0, 1, 2, 0])". So it is already an integer. So if you want to convert this to boolean I will mention here that we want to convert this int tensor ".to(torch.bool)", and we will convert it to bool tensor. Let's try to print the details of this new tensor, which we have just converted. And we'll be able to see whenever there is zero, it is false. Whenever the value is greater than zero, that is one, two, it can be thousand as well, it will say true. Let's try to see for minus one as well, what will happen to the minus one. And it is also true. So only when the value is zero, it is giving us false, and any value other than zero will give you true. When you try to convert from integer or float to boolean, because zero is the only value for false, and every other value is being considered as true.

### Impact of Data Type on Memory:

float32_tensor = torch.ones(1000, dtype=torch.float32)

float64_tensor = torch.ones(1000, dtype=torch.float64)

# tensor.element_size()

# element_size() --> gives you the size of one element in bytes

# nelement() --> gives the total number of elements in tensor

print(float32_tensor.element_size()) - 4

print(float32_tensor.nelement()) - 1000 

print("Memory used by float32 tensor : ", float32_tensor.element_size() * float32_tensor.nelement(), " bytes") - 4000 Bytes

print("Memory used by float64 tensor : ", float64_tensor.element_size() * float64_tensor.nelement(), " bytes") - 8000 Bytes

**It is like Size of One Element x Total Number of Elements in Tensor**

So now let's try to see the impact of data type on memory. We are going to check impact of data type on memory. We are going to first create "torch.ones(1000, dtype=torch.float32)". That is a vector or tensor full of ones. And we are going to mention value of 1000. And first we will mention the dtype here. So we are going to create this with data type of float32. It is "torch.float32". We are going to create the same but with a different precision. Instead of 32 we will mention 64 and we will store in float32 tensor, and for this we will store into float64 tensor. Now with the help of "tensor.element_size()", that is a function which we are going to use. So the function which we are going to use is: we have to mention the tensor that is the variable we need to mention, not just tensor. And then we will mention the element size. So what this function will give you? This function gives the size of one element in bytes, okay, that is one element in bytes.

So what we will do we will first print the memory. So "The memory used by Float32 tensor is:", then we will mention "float32_tensor.element_size() * float32_tensor.nelement()". Because this only gives you one value, right. "element_size" gives size of one element, but how many of that particular element we have? We have "float32_tensor.nelement()". And that's it. So this covers it. So what this "nelement" gives: "nelement" gives total number of elements in the tensor. Let me write this function names here. So if I talk about "element_size", this gives the size of one element in bytes. So this is size of one element. If it is a float32 tensor, it will take the data type for the data, type it will calculate the total number of bytes that data type takes. If I talk about float32, this will check for "torch.float32", what is the size it takes in bytes, and how many elements we have we need to multiply with that. So suppose we have total number of 1000 elements here. So 1000 multiplied by the size of one float32 tensor element in bytes. So that will give us total number of memory used by float32 tensor.

And "nelement", this gives the total number of elements in tensor. If you want to see this particular value one by one, instead of writing all in a single statement, you can just check this first value. And the second one we will just see total number of elements. And then we will do the calculation in the third print statement. So let's see. So each float32 tensor takes four bytes. We have total number of 1000 values. So what we will do we will multiply four with 1000 and we will get 4000 bytes used for that particular tensor which we are storing here. And we will do the same for float64. We will mention float64 here and total number of elements present in float64. And let us see how much element size it takes. So let me mention that at the end we want to write bytes. You can also use f-string like f-string is better, but for the simplicity I am writing it in print statement. So memory used by float32 tensor is 4000 bytes, while memory used by float64, because it is double in precision, it is taking double the size which is 8000 bytes.

So key tensor data types and their use cases: when we talk about float32, it is a 32-bit floating point and it is the default for most models, default for most neural network models. And if I talk about float64, this is for high precision computation. If I talk about int32, this is a general-purpose integer. And if I talk about int64, these are used for tensor indices. If you want to check the index or do slicing and other operations, it is basically int64. The values which we provided are int64. For bool we already know these are again used for mask and logical operation. And if I talk about float16 which is half of your floating point, that is a default floating point, this is half precision of your default floating dtype. So this is used to reduce the memory usage on GPU. Okay, so if you want to reduce memory usage then we use float16. We also have int8 and int16. Again both are used to reduce memory usage. And finally we have complex64 which I've already talked about. This is used for advanced computation. So I'm going to write it here so that you can refer it later. So this is complex64. We have complex128 as well. So this is 32-bit complex numbers, okay, and this is used for advanced computations.

# float32 --> default for most NN models

# float64 --> high precision computation

# int32 --> general purpose integer

# int64 --> tensor indices

# int8, int16 --> reduce memory usage

# bool --> mask and logical operation

# float16 --> half precision of your default floating dtype, reduce memory usage

# complex64 --> advanced computation

**G) Tensor Manipulation**

(i) Reshaping

(ii) Slicing

(iii) Joining or splitting

(iv) Transposing and permuting dimension

In this lecture we are going to explore tensor manipulation techniques in PyTorch, which will help us operate tensors better for efficient workflows. Let's get started. So tensor manipulation refers to operation that alters the structure, shape or content of tensors. This includes reshaping. So let me write it here. We will first connect to our instance and I will write tensor manipulation. You can create a text box and that will be better. So we are doing tensor manipulation. And then the first one is we do the tensor manipulation. And this operation includes reshaping the tensor. Second one is we do slicing on tensor. Let me increase the size. We also do slicing on the tensor. Then another operation includes joining or splitting, joining the tensors, joining multiple tensors, or splitting one tensors into multiple sections. So this also includes splitting tensors. And there are a few more like transposing that is changing uh, like m cross n matrix to n cross m matrix. This is what we do in transposing. We will get into more details. And then is permuting dimensions. So this is so in total we have reshaping and reshaping. We have two techniques that is using reshape or view. We will use slicing. Then we have joining and splitting then transposing and permuting. So we will see it one by one and understand what actually they mean.

So we will start by reshaping tensor. So let me write reshaping Tensors. So as I mentioned, we have two methods here in reshaping. One is reshape and another is view. So we are not going to focus on each of this operation that is reshape and view, because that is not the point of this particular lecture. The point is how we can perform tensor manipulation. How can we manipulate our tensors? We are going to have a separate lecture on reshape and view because they have their own concepts, which we need to learn in more depth. So we are just going to use reshape and view to just reshape our tensors. This is the main focus here.

So what I'm going to do first we will start by importing torch. So we will mention "import torch". I think all of this command should go in next line not in the torch. We don't want to initialize our torch again and again. So here I will first create a tensor. So we will mention tensor. Okay. Or let's name it as original tensor. Just a better naming convention. And then we will create a tensor. So we will mention "tensor = torch.arange(12)". So this will give us tensor from particular range which we will mention. So we are giving it 12. So this will give us value from 0 to 11 total number of 12 values. But the range will be 0 to 11. That is what this arange does. So let's try to print it and see this particular operation. You can see this is value starting from 0 to 11. This is our original tensor.

So what we will do with the help of reshape. First we will reshape our tensor okay. First we will reshape our tensor. So here because we have total number of 12 elements. We can also check the length here if you want to see the length. Or we can check any elements. Also, if you can mention "tensor.nelement()". It is nelement in full, so it is "nelement()". So we will know the length of our tensor. So we have total number of 12 elements here. So what we will do because we have 12 values. Now I want to create I want to change this shape. Currently this is a vector the dimension if you want if if I also want to know the dimension let's also print the dimension. So dimension is "tensor.ndim". So right now the dimension is one. It means this particular tensor is of type vector.

**Original Tensor** - original_tensor = torch.arange(12)

print(original_tensor) - tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

print(original_tensor.nelement()) - 12

print(original_tensor.ndim) - 1

**Reshaping Tensor** - reshaped_tensor = original_tensor.reshape(2,6)

Now, print(reshaped_tensor.ndim) --> 2; Because we have reshaped it

**Reshaping to (3,4)** - reshaped_tensor = original_tensor.reshape(3,4); ndim - 2

So I now I want to convert it to matrix I want to change this dimension. But keep in mind that we need to consider that what will be the total number of length. So because we have 12 values what I can do, the first thing I will do is I will be converting this particular vector to a matrix of dimension two cross six. If you check two cross six. If you check two cross six, the total number of values this two cross six will have is 12, which matches our original tensor. So I'm going to create a new tensor. So I will mention this as a reshaped tensor. And here I'm going to write "reshaped_tensor = original_tensor.reshape(2,6)". Let me comment it and then we will print our reshaped tensor. Now you can see this is already looking different. We can also print dimension. And we can also print the number of elements. The number of elements will not change because we are operating on top of another tensor. You can see that now the dimension is two. It is not one anymore. It is not a vector. Now it is a matrix. Now it has total number of two rows. You can see from value 0 to 5 and 6 to 11. We have total number of two rows. And it means we have mentioned here two rows. And we want six columns which is what it has done.

Now if you want to change to something others we can also do that. Let me write it here. So instead of making this reshape now we can also reshape to a different value. Now let's suppose because we need to have total number of 12 values. I can also mention three cross four. That is, four multiplied by three also gives you 12. If I try to print it, it means three rows and four columns. So now you can see we have total number of three rows and four columns. That is zero, one two, three. And all the values are respectively the number of dimensions remains same. And also we have 12 here. That is the total number of elements. This is what we can do with the help of reshape. So this is the first operation for doing tensor manipulation.

# View

print(original_tensor.is_contiguous())

flattened_tensor = original_tensor.view(-1)

print(flattened_tensor)

We can also do a different operation for reshaping our tensor which is view. Let's talk about view. It will everything about view will be same but internally it requires a contiguous memory. Okay. So we will talk about this later. As I mentioned before, this reshape and view will have its own separate lecture in coming classes. But for now it will operate the same. But only thing changes between reshape and view is that view requires contiguous memory. That is in our Ram, in our memory all the data points should be in a sequence. Okay. But for now we will just view on our original tensor. This is our original tensor. So what I will do I am writing here as view. Then I will mention "original_tensor.view(-1)". But if this original tensor is not contiguous, this view operation will give us error. So what we are trying to do here with the help of view, when you mention minus one, minus one means your dimension is going to be automatically calculated by your PyTorch. Let's try to see. So minus one means it will do. It will create a vector. That is what minus one means, because we are only saying to our PyTorch that whatever this value fits here, calculate it automatically. So because we are only giving it one dimension, it will only calculate. The answer will here be 12. That's all. And answer will be 12 here. So we are going to save this as "flattened_tensor = original_tensor.view(-1)". So this is a flattening operation. If you mention minus one here. Let's try to print it. And you can see this is how it is working.

So this is what view does whatever you are doing with reshape can also be done with view. But with view you need to make sure that the memory you are using, the original tensor you are using is contiguous. So if you want to check whether this original tensor is contiguous or not, as I mentioned in the coming lectures, we will have like a lot of details in that particular lecture. But if you want to see right now what we can do, we can check "original_tensor.is_contiguous()". You can see there is a function is contiguous. Let's see whether it is true. So what we will do we will try to print it. If it is true it means the original tensor that is being created by PyTorch is in a sequence in memory. If it is not sequence view will not work, but reshape will work. Yes, this particular tensor, it's contiguous. It means it is in a sequence in our memory. That's why it is working. So that is all about reshaping and doing tensor manipulation.

## Slicing - Extract specific portions of Tensors

tensor_a = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

Dimensions - 2 (tensor_a.ndim)

**Extracting only first row** - print(tensor_a[0])

**Extracting first column of all the rows** - tensor_a[:,1]

**Extracting Sub Tensor from the Main Tensor** - sub_tensor = tensor_a[1:,1:]

Next thing is we are going to perform slicing operation on our tensor. Let me write it here. So what we do inside slicing we extract specific portions of tensor using slicing. What we do here let me write it in short. We extract specific portions of tensors. That's all we have to know. That is what we do in slicing. So in slicing let's create again a tensor. So I will mention "torch.tensor". So this is torch dot tensor. We will create our own tensor. So I'm going to mention a 2D tensor here. And the first row will have 1,2,3. So it means it has total number of three columns. And then I'm going to mention 4,5,6. And then I'm going to mention 7,8,9. So in total we have three rows. As you can see this is row number one, row number two, row number three. And each row has three columns. So in total we have three columns. So I'm going to store into tensor. So let's write it as "tensor_a = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])". So this is our original tensor for this particular operation. Let's try to print it and see the value.

### Joining Tensors: torch.cat() --> Merges tensors along an exisiting dimesion. (Doesn't create any new dimension / increase existing tensor rank)

tensor1 = torch.tensor([[1,2],[3,4]])

tensor2 = torch.tensor([[5,6],[7,8]])

**Row wise Merging** - concat_tensor_rows = torch.cat((tensor1, tensor2), dim=0) **Rows will get added one below the other (2+3 will be 5 rows)**

**Column wise Merging** - concat_tensor_colm = torch.cat((tensor1, tensor2), dim=1) **Columns will be 1+2 = 3 Columns** 

# Stack --> Creates a new dimension, increases the tensor's rank (Creates new dimension and increases existing tensor rank)

stack_tensor_rows = torch.stack((tensor1, tensor2), dim=0) **Shape is [2,2,2]**

stack_tensor_colm = torch.stack((tensor1, tensor2), dim=1) **Shape is [2,2,2]**

**Row wise stacked tensor** - tensor([[[1, 2],
         [3, 4]],
        [[5, 6],
         [7, 8]]])

**Column wise stacked tensor** - tensor([[[1, 2],
         [5, 6]],
        [[3, 4],
         [7, 8]]])

### Stacking does operation on top of Matix and not on top of Rows/Columns

Similarly we have five, six, seven, eight, this is also a 2D tensor, this is tensor one, this is tensor two, so if I want to add these two tensors on rows it means the final tensor will have internally it will have total number of four rows that is 1234 and 5678, it will not have this dimension because we are adding and creating a new one and it will look similarly like this, you can see we have two dimension as a output, but in the two dimension we have four rows and this is how we can achieve by using dimension is equals to zero, but if you mention dimension is equals to one so it will add alongside, it means one and two and right to it will be added a55 comma six of the next tensor, then three comma four will be in new line, then alongside right to it we will have seven cross eight because we are adding on columns, so this is column one two, then this is column one two of the next tensor, this is column one two of the next tensor, this is column one two of the next tensor but a different row, then we will get this particular value, so this is what happens when we are using cat, as mentioned this operates by merging tensors along an existing dimension, it is not creating any new dimension, this is a 2D dimension, this is also a 2D matrix, this is a 2D matrix, this is also a 2D matrix, so everything is happening internally on the internal values of our main dimension, but if I talk about stack let’s try to understand stack, it will be very similar operation, but when we talk about stack, stack creates new dimension, it increases the tensor rank, so it creates a new dimension which increases the tensors rank, so all these tensor ranks are two, that is the dimension, let’s try to use stack and use on the similar tensor one and tensor two, so let me copy this whole code, so instead of concat I will just mention stack here, also stack here, and instead of cat I will mention stack, stack and everything will remain same, I also want to print it, and I also want to print our original dimension as well, because we already have this value, so I am just going to print the shape of it, also the shape here, okay, and then also printing, after printing the shape I am going to print this dimension, so I’m going to mention shape of the new one as well as their actual values, if you try to see the output, we will see that the original tensor is of size two cross two, two cross two, that is, tensor one and tensor two, but if I try to check concat tensor rows, it is of dimension four cross two, so you can see we have total number of four values one, two, three and four, that is one two, three, four, it is having the same exact values like the stack okay, that’s why I was like it has not added any dimension, there is something wrong, now it should give us correct answer, yes, you can see this is one, two three, four, this is a single matrix, this is a matrix, then because we are adding on row wise, it has it picked this value and appended on top of this value, it picked the whole matrix not row by row, you can see this is the first value, this is the first value, this is taken everything as row, then it is appended to a next row, this is what it is trying to do, okay, this is like stacking along rows, and if I try to see that this is stacking along columns okay one two, then we have three four and five six and then seven eight, we have little mistake here because again I copied everything from rows and column, that’s why it looks a little weird, now it will be correct, that is one two with five six, that is one, two is now with five, six is a single matrix, then three four and seven eight is in a new matrix, altogether you can see that is one, two, five, six in a new matrix, then three, four, seven, eight in a new matrix, and you can see on this two cross two matrix we have a new dimension, on top of all of these values you can see this extra square bracket at the start, this is extra square bracket, this is also extra square bracket ending at the start, so this is the main matrix and this is the main operation, but it does by creating a new dimension because it is trying to add do the operation on top of matrix itself, not on particular rows, it tries to add on the dimension, you can see this is one, two, five, six, then three, four, seven eight, so there is little difference when you try to use stack and using torch.cat, 

## Splitting Tensors:

## torch.chunk() --> Divides your tensor into equal-sized chunks

## torch.split() --> Allows eneven splitting based on size

**origin_tensor = torch.arange(12); chunks = torch.chunk(origin_tensor, 5, dim=0)** 

12 was not divided by 5; So it will divide into 4, with equal parts - (tensor([0, 1, 2]), tensor([3, 4, 5]), tensor([6, 7, 8]), tensor([ 9, 10, 11]))

**Split** - splits = torch.split(origin_tensor, 7, dim=0) - (tensor([0, 1, 2, 3, 4, 5, 6]), tensor([ 7,  8,  9, 10, 11]))

### Transposing and premuting

## Transpose() --> Swaps two dimesnion. mxn --> nxm

## Premute() --> Rearranges all dimension in the specified order

**tensor_original = torch.arange(24).reshape(12,2)** - torch.Size([12, 2])

**transposed_tensor = tensor_original.transpose(0,1)** - torch.Size([2, 12])

**Permute Operation:**

tensor_original = torch.arange(24).reshape(2,3,4)

permuted_tensor = tensor_original.permute(2, 0, 1) ([0,1,2])

**Permute - Earlier it was 0,1,2 Now we want it to be (2,0,1)**

So the next thing we will learn is splitting tensors as we can see above, so we are done with reshaping slicing joining, now it’s time to learn splitting how to split our tensors, let me write it here, let me create a new text box, so I will mention splitting, yes, that is splitting, we are doing splitting of tensors, so splitting tensors means we want to split our tensors into smaller chunks, and we are going to use two methods for it, which is torch dot chunk, and the second one is torch dot split, so torch dot chunks actually divides your tensors, let me mention it divides your tensor into equal sized chunks, and if I talk about torch dot splits it allows uneven splitting as well based on size of our tensor, so let’s see how to implement torch dot chunk and torch dot split, so first of all we will start with torch dot chunk, so first we will create torch dot arrange, we will create total number of 12 values as we have mentioned before, and we will store into a tensor, so what we will do next year we will try to use torch dot chunk, and then we have to mention the tensor, this is the tensor which we have created, let me name this tensor as original tensor, because I don’t want to create the same name values as tensor, I’m going to create this original tensor, and I want to divide into a chunk of three, and then I want to divide into a dimension of zero because we have only rows here, so I’m going to create I’m going to perform this operation on dimension equals to zero that is on rows, and then I’m going to create chunks, and this will return us a iterables, so what do you mean by iterables, you will see this answer here, this will not be directly tensors but it will be in a set or a tuple, you can see this is you can check the dtype, and this must throw error because this is not a tensor, that’s why it is saying tuple, yeah, so this is a tuple actually you can see, so it was a tuple and it is returning you tuple that is inside a tuple you have the output of tensors, let’s see, so this is tensor number one, it has divided this into total number of three parts, that is we have mentioned three, so this is tensor zero tensor one and then tensor two, even if you mention five it will not allow it and find the optimal size for it, you can see now this is the optimal size, it is able to find the maximum it is able to find when given the value of five is four, so because 12 cannot be divided by five, but four is 12 is divisible by four, that’s why the maximum it can go is four here, so it has tried to divide this particular tensor, this is index zero, index one, index two and index three, and because this is a retrieval, what we can do, we can also mention for chunk and chunks we will just print chunk, but this does not gives a flexibility, even if I mention five I want five because it was not the optimal value, chunk is automatically deciding the number of splitting it has to do, but in the case of splitting, even if the value is uneven, if I even if I mention five and only one values come at the last answer, it will allow it, and this is the main advantages of using split but chunk advantages, it will try to find the optimal value even if you give some wrong number to it, let’s try to mention torch dot split, I can use this same exact code because we already have our original tensor and split will also return you, let me mention a split here, a value enclosed in tuple, this is split, split, splits, splits, this will also be split, this will be only split, and we will try to print split only, and instead of torch dot chunk, this value will also change, this method is split and everything will remain same, now I want to split into five, this will not automatically calculate values and whatever we have mentioned, it will only mention it will only give you that output, let’s try to run it and see the output, you can see this is our original value, this is all the splits we have in the first, at the first index we have a tensor with five elements, at the first index this is zeroth index, at the first index on the tensor we have five more values, and on the second index that is the second, that is the third tensor, we are having two values, this is total number of five values, total number of five values ten values are covered, then we have two more values, and ten and 11 are covering here, you can also mention seven, and let’s try to see how the split works, seven times two is 14, so it will be the first seven will be here, and whatever left value is left that is total number of five values kept in a second tensor, so this is an uneven splitting, but your torch dot chunk divides your tensor into equal sized chunks, that is the main difference between torch dot chunk and torch dot split, now the next thing is how to transpose your matrix or do the permuting of your matrix, so let me try to write, I am going to do in the same tab itself, so I will mention transposing, and permuting, so what transposing does, let me write what transpose does, this is the method which I’m going to write, it swaps transport swaps two dimension, so think of if the dimension is m cross n, the output will be n cross m, the dimension will be swapped, okay, this is about transpose, now if I talk about permute, so what this permute does, it rearranges, it rearranges all dimension in the specific or the specified order, let’s try to see the implementation and then it will be more clear, so what we are going to do we are again going to create torch dot arrange, and here I’m going to create total number 24 values, so instead of reshaping it later, I’m going to reshape it here on the vector I’m going to perform because the output of this will be a vector, and on this vector I’m performing reshape operation, and for this 24 values I want to reshape to two, cross three, cross four, this is what I want to do, and then I will create a tensor, and if is if this is too complex what we will mention we will just mention 12 cross two, this is better for the understanding, this will give us a tensor, and we will mention again in tensor original, so on this tensor what we are going to perform we are going to perform our transpose operation, we are going to perform transpose operation, and we will mention that we want to transpose dimension zero and dimension one, okay, so we are mentioning here what are the different transforms like which of the dimension, which of the index which we want to transfer, so we want to transpose index zero and zero, we have 12 in 1 we have two, it is going to do the transformation transpose on that, so let me write the operation name here, so we are going to do the transpose, and let’s try to see the output, so we have to store into transposed tensor, transposed tensor, so we will first print transpose tensor and we will also print, tensor original dot shape and also dot shape here, and once the shape is done I want to print transpose tensor, but this is our original tensor, that’s why not print the original tensor as well, it will be more clear, now this is our original tensor, it has total number of 12 rows and each row is having two column the size, the first size is torch dot size is 12 comma two, okay, 12 rows, two columns, 12 rows, two columns, now the next one is, after doing the transpose of, we have mentioned that which of the indexes you have to do the transform like transpose operation and store into transpose tensor, so the shape of transpose tensor is previously the original is 12 cross two, now it is two cross 12, it means in total we have two rows and 12 columns, and in each rows we have 12 columns, so this is how we can do transpose of any matrix, this is like this is called transpose operation, next thing is we are going to do permute operation, so what we will mention in permute operation, it rearranges into a specified order, so like right now here we have very less dimension, so I’m going to increase the complexity to like understand the changes better I’m going to stick with we want to make two cross three cross four, and if you multiply this total number of values it should be, if you multiply two multiplied by three multiplied by four it will give you four as the output, you can see this is 24 right, 24 is matching with the total number of 24 values, it means we are doing the correct reshaping, and on that reshape what we are going to mention in this tensor original, we are going to mention permute, and in the permute we have to mention the indexes at zeroth index, which at zeroth dimension okay, at zeroth index, this is how the original sequence looks like, right, this is how the original sequence is, so if I try to access zero, one and two, let me print it, let me print, tensor original, then I will explain what I’m talking about, there is something wrong here, comment it because we have not satisfied all the arguments, so this is like of the shape of two cross three, cross four, it means we want total number of two tensors of size three cross four, so this is tensor one, this is tensor one, this is tensor two enclosed by an extra bracket because it needs to cover that, so we have total number of two tensor, this is tensor one, tensor two and each tensor is of type, this is total number of three rows and four columns, total number of three rows and four columns, so we have two of three cross four tensor three cross four matrix, so if this is our original the starting index this is if I talk about zero, this is the zeroth dimension, this is the first dimension, this is the second dimension, that is the index, so if I want to change that particular sequence itself, that is we want the first index to be two index to be replaced by the second index, then I want to be replaced, I want to keep zero as it is and one, so whatever the original sequence is zero, one, two, the sequence we want is two, zero, one, think of this as an example at index zero, this is our list, this is our list, L is equals to list and in list we have values like zero, one, two, three, four, so what I’m mentioning here that at index zero at index zero the value is zero right, whatever the value of index zero at zero I want it to be of, this is index zero, so I want this four to be replaced, and wherever there was four, the position of four should be swapped with position of zero, so this is swapping, so whatever the index zero is not swapped with index of two, whatever was at index one is not swapped with zero, whatever was at index two is now swapped with a value of one, if you try to see it and try to store into permute tensor, think of this as we are swapping the dimension itself, the values itself, this will be stored into permuted tensor and we will print our permuted tensor, and the spelling looks wrong here, so this is permute p e r m u t e, that was the error, you can see previously it was 0123, now it is 048, the dimension itself is changed, previously, what was the column, if I try to print the shape here, let’s try to print shape, then it will be clear, dot shape, and now the shapes will be switched, if you check, the shape will be switched, so permute tensor dot shape, the shape was two, cross three, cross four, so what we have mentioned at index zero it is two, at index one it is three at index two it is four, what we have mentioned, whatever was at index zero before switch it with index two, so instead of this two it will be replaced with four, so we can see, let’s see, yes it is replaced with like four, so whatever was at index one replaced with index zero, so at index one we have two, so it will be replaced by what, it will be replaced by two here, so it is two here right, so whatever was at index two replaced with one, so at index two we have four, we want to replace with what, we want to replace with index one and index one, we have total we have three, so it is three here now, so whatever was 234 before has now become four comma two comma three, so it means we have four of this matrix, and in each of this matrix we have a dimension of two cross three, so you can see we have two rows and three columns, this is what permute does, it rearranges all the dimensions in the specified order, so here we are mentioning the order itself, and we have to compare with the original order, the original order is two, cross three, cross four, and we have to count the index index starting from zero, one and two, so we are just mentioning that swap, this indexes in this format to zero and one, and whatever those values are, it will look like for two and three because we have two comma three comma four in our original shape, and now comes the final topic that is cloning and detaching, I have not mentioned about cloning and detaching when I was like writing the agenda of this lecture, that is tensor manipulation, this is not a part of it, but it is an important part of tensor manipulation, if we check in numpy there are concepts like copy, so if you want to create a copy of that numpy like I’m just giving you a Python version, here, let me write a Python version first, let me let’s create our markdown, so we are doing cloning and detaching, so let’s understand cloning with the help of a Python example, so if I try to create a variable let’s see this variable as this I will create a list here, and I will use values like one comma two, comma three, and if I mention another variable and will try to mention b is equals to a, both of a and B are not separate entities, a is having

## Cloning and Detaching:

tensor = torch.ones(3,3, requires_grad=True) # part of computation graph

cloned_tensor = tensor.clone() # # part of computation graph

**This code will detach the tensor from the computation** - graphdetached_tensor = tensor.detach()  

# Not a part of computation graph

# But storage will be same as the original ones

Cloning and detaching are especially important in training deep learning models with PyTorch’s autograd system. When we use "clone()", we create a new tensor that is completely independent of the original one in terms of memory but still requires gradients if the source tensor does, for example "y = x.clone()". On the other hand, when we use "detach()", we create a new view of the tensor that shares the same underlying data with the original tensor but does not track operations for gradient computation, for example "y = x.detach()". This is useful when we want to perform operations on tensors without affecting the computation graph, such as using model outputs for logging or evaluation without interfering with backpropagation.

Another important aspect of tensor manipulation is ensuring that reshaping operations are done correctly. While "reshape()" can handle both contiguous and non-contiguous tensors by making a copy if necessary, "view()" is faster but only works with contiguous tensors. If a tensor is not contiguous, we must call "x.contiguous()" before applying "view()", otherwise it will throw an error. For instance, "y = x.t()" transposes a tensor, and after that "y.view(-1)" will fail unless we use "y.contiguous().view(-1)".

Joining and splitting tensors are very common in mini-batch training, where data is grouped together. For example, "torch.cat([batch1, batch2], dim=0)" will concatenate two batches along the batch dimension to form a bigger batch, while "torch.chunk(batch, 4, dim=0)" will split one batch into four smaller ones. This makes it easier to work with data loaders, parallel computation, and batch processing. Similarly, "torch.stack([a, b, c], dim=0)" is often used to combine multiple tensors of the same shape into a single higher-dimensional tensor, which is useful for preparing input data.

Permutation and transposition are particularly useful when working with image data or RNNs. For example, in computer vision tasks, PyTorch often uses channel-first tensors shaped like "(batch_size, channels, height, width)", but some libraries or pretrained models might expect "(batch_size, height, width, channels)". In such cases, "x.permute(0, 2, 3, 1)" can be used to reorder the dimensions. For recurrent neural networks, where time-steps may need to be the first dimension, "x.permute(1, 0, 2)" can reorder tensors to match the expected format.

Altogether, tensor manipulation in PyTorch—through reshaping, slicing, joining, splitting, transposing, permuting, cloning, and detaching—forms the backbone of preparing data, building deep learning models, and controlling how computations are carried out efficiently during both training and inference.

**H) Matrix Aggregation**

In this lecture, we will explore matrix aggregation. Aggregation refers to extracting meaningful summaries from matrices or tensors, such as their sum, mean, or maximum values. These operations are essential for understanding data patterns and for reducing dimensions. Let’s get started.

We will begin by creating our input tensor. First, we import torch, and then we create a tensor. For this example, I will create a 2D tensor. Even if I write values such as 1.2, they will be treated as floats by default. I want three rows, each containing three elements, so the tensor will form a 3×3 matrix. The values will be arranged as:

[1, 2, 3
4, 5, 6
7, 8, 9]

We will call this tensor matrix. Before performing any operations, let us first print the matrix to verify its contents.

**Basic Matrix Operations:**

print(matrix.sum()); print(matrix.min()); print(matrix.max()); print(matrix.median()); print(matrix.mean())

Now, we start with the most basic aggregation operations. For example, if we want the sum of all values in the matrix, we can simply use matrix.sum(). This will add up every element and print the total, which in our case is 45. Similarly, we can calculate the minimum, maximum, median, and mean values. For instance, matrix.min() will return 1, matrix.max() will return 9, matrix.median() will give 5, and matrix.mean() will also give 5. These are some of the simplest aggregation functions available in PyTorch.

## dim=0 --> rows (column)
## dim=1 --> column (rows)

print(matrix.max(dim=0))
print(matrix.sum(dim=0))
print(matrix.sum(dim=1))

After covering basic aggregation, we move on to aggregation along specific dimensions. Instead of computing over the entire matrix, we may want to compute along rows or columns. In PyTorch, when we specify dim=0, the operation runs across rows (that is, it works column-wise). On the other hand, when we specify dim=1, the operation runs across columns (that is, it works row-wise).

To clarify, when we mention dim=0, the function will consider values across rows for each column index. For example, it will take the first element of row 0, the first element of row 1, and the first element of row 2, and perform the aggregation on them. This is why we say it operates “across rows,” but in practice it processes each column. Similarly, when we specify dim=1, the function takes all values across columns within each row and applies the aggregation.

Let us test this. If we write matrix.sum(dim=0), it will compute the column-wise sums. For example, the first column is 1 + 4 + 7 = 12, the second column is 2 + 5 + 8 = 15, and the third column is 3 + 6 + 9 = 18. The result is [12, 15, 18]. Similarly, if we write matrix.sum(dim=1), it will compute the row-wise sums. For row 0, the result is 1 + 2 + 3 = 6; for row 1, it is 4 + 5 + 6 = 15; and for row 2, it is 7 + 8 + 9 = 24. The result is [6, 15, 24]. The same concept applies for max and min functions as well.

**Cumulative Aggregation** - cumulative_sum = matrix.cumsum(dim=1)

**Cumulative Product** - cumulative_prod = matrix.cumprod(dim=0)

Next, let us look at cumulative aggregation. Cumulative operations keep a running total (or product) as they iterate through values. For example, if we have values [1, 2, 3], the cumulative sum will be: first 1, then 1 + 2 = 3, then 1 + 2 + 3 = 6. This gives us [1, 3, 6]. Similarly, a cumulative product would multiply as we go along: first 1, then 1 × 2 = 2, then 1 × 2 × 3 = 6.

To compute cumulative sum in PyTorch, we use matrix.cumsum(dim=1) if we want to calculate across columns within each row. For the first row [1, 2, 3], the result becomes [1, 3, 6]. For the second row [4, 5, 6], it becomes [4, 9, 15]. For the third row [7, 8, 9], it becomes [7, 15, 24]. Similarly, for cumulative product, we can use matrix.cumprod(dim=0). This will compute down the rows. For example, in the first column [1, 4, 7], the cumulative products will be [1, 4, 28] because 1, then 1×4=4, then 1×4×7=28. This is how cumulative aggregation works.

### Advanced Aggregation - masked_matrix_sum = matrix[matrix>5].sum()

Now, let’s explore advanced aggregation using conditions. Suppose we want to sum only the values in our matrix that are greater than 5. We can create a mask with the condition (matrix > 5), which generates a True/False matrix. Applying this mask to matrix and then summing gives us the sum of all values greater than 5. In our case, those values are [6, 7, 8, 9], and their sum is 30. Similarly, if we specify (matrix > 8).sum(), it will only include the value 9, so the result is 9.

## Non-Zero Values:

matrix = torch.tensor([[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.],])

non_zero = matrix.nonzero()

We can also count non-zero values. PyTorch provides matrix.nonzero() for this. It returns the indices of all non-zero elements. To count how many non-zeros are present, we can check its size using matrix.nonzero().size(0). For our original matrix with 9 non-zero elements, the result is 9. If we modify the matrix by replacing three elements with zeros, the result changes to 6, which matches the number of non-zero elements left.

Finally, let us discuss normalization. Normalization is the process of rescaling values so that they lie between 0 and 1. The formula is:

**Normalized Matrix**

max_v = matrix.max()

min_v = matrix.min()

normalized_value = (value – min) / (max – min)

To apply this, we first compute the maximum value of the matrix (max_val = matrix.max()) and the minimum value (min_val = matrix.min()). Then we apply the formula: (matrix – min_val) / (max_val – min_val). This creates a normalized version of the matrix. In our case, since the minimum is 1 and the maximum is 9, the element 1 becomes (1–1)/(9–1) = 0, the element 9 becomes (9–1)/(9–1) = 1, and everything in between is scaled proportionally. Thus, the new matrix ranges from 0 to 1 instead of 1 to 9.

Through this lecture, we have covered a wide range of aggregation operations in PyTorch. We started with simple functions such as sum, mean, min, max, and median. Then we learned about aggregation along specific dimensions, followed by cumulative operations like cumulative sum and cumulative product. We then moved to advanced aggregation techniques such as conditional sums and non-zero counts. Finally, we explored normalization using min-max scaling. With these tools, we can efficiently summarize and analyze data stored in matrices or tensors.

