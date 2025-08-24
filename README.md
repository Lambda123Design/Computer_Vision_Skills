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
