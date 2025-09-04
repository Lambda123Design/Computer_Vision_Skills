# Computer_Vision_Skills

pip install opencv-python

pip install ipykernel

**OpenCV - imread, imshow, waitkey, destroyAllWindows, imwrite, VideoCapture, videoread, cv2.VideoWriter_fourcc**

## Different ways to create neural network: Function (Functional API - Flexible, harder to interpret), Sequential (Sequential API - nn.Sequential). There are also different ways to define neural networks in PyTorch. The Functional API is a flexible approach where we directly define operations on tensors, allowing for more customization. The Sequential API is a structured approach where layers are stacked in a linear order using torch.nn.Sequential, making it simpler to define straightforward models.

## We need to convert NumPy Arrays to Tensors in order to do prediction for our Linear Regression Model with Pytorch Components Project

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

**I) View and Reshape Operation**

**J) Stack Operation**

**K) Understanding Pytorch Neural Network Components**

**L) Creating Linear Regression Model with Pytorch Components**

**M) Multi-Class Classification with Pytorch using Custom Neural Network**

**N) Understanding Components of Custom Data Loader in Pytorch**

**O) Defining Custom Image Dataset Loader and Usage**

**P) CNN Training using Custom Dataset**

**Q) Understanding Components of an Application**

**R) What is Deployment**

**S) Tools to Create Interactive Demos**

**T) Hosting Platform**

**U) Setting Up Gradio App in Local Space**

**V) Implementing Gradio App Interface Backend**

**W) Setting HuggingFace Space**

**X) Deploying Gradio App on HuggingFace Spaces**

**III) Deep Dive Visualizing CNN's**

**A) Image Understanding with CNNs vs ANNs**

**B) CNN Explainer**

**C) Visualization with Tensorspace**

**D) CNN Filters**

**E) Building our own Custom Filters**


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

**I) View and Reshape Operation**

We begin with torch.arange to create a sequence of values. Suppose we want a total of 12 values starting from 0 to 11, we can generate this sequence and store it into a variable called tensor. Once we have this tensor, we can reshape it using both view as well as reshape. First, we use the view method and specify the new shape. For example, if we want to reshape the tensor into a 2 × 6 matrix, we mention that shape with view. This gives us a reshaped tensor using view. Next, we repeat the same thing with reshape instead of view. When we print both results, we see that there is no visible difference. Both methods rearrange the same data into the desired format. To further confirm, we can print the shape of either one (no need to check both), and we see that the shape has indeed become 2 × 6, exactly what we specified.

Before moving further, it is important to check whether a tensor is contiguous. PyTorch provides a function .is_contiguous() for this purpose. We can use it to determine if the original tensor that we reshaped with view or reshape is contiguous in memory. If the data is not contiguous, using view will throw an error. On the other hand, if the tensor is non-contiguous, reshape will handle it gracefully because it creates a new copy of the data and works with that. This makes reshape more flexible than view.

Now, consider a case where we don’t know one of the dimensions while reshaping. Suppose we have 12 values in total and want PyTorch to calculate one of the dimensions automatically. We can do this using -1 in the shape specification. For instance, we can reshape the tensor to have 3 rows but leave the number of columns unspecified by writing -1. PyTorch will automatically infer the correct number of columns by dividing the total number of elements by 3. Printing the result shows that the shape becomes 3 × 4. This use of -1 is quite practical when we want PyTorch to determine the missing dimension.

This idea becomes very useful in Convolutional Neural Networks (CNNs). After certain operations, CNNs require vector inputs instead of a 2D or 3D matrix. For example, suppose we have an m × n matrix but need a vector of size 1 × (m×n). Here, using -1 helps flatten the matrix into a vector. Let’s take a practical example. If we create 24 values using torch.arange and reshape them into the shape 2 × 3 × 4, we get a 3D tensor. Now, if we want to feed this tensor into a CNN, we might need to flatten it into a 1D vector. We can do this easily by applying view(-1) or reshape(-1). This flattens the tensor, turning the 2 × 3 × 4 shape into a single vector of size 24. Printing it confirms that all values are flattened. We can then check whether this flattened tensor is contiguous by calling .is_contiguous(), which usually returns True.

Next, let’s deliberately create a non-contiguous tensor and see how view and reshape behave. We can start with 24 values reshaped into a 12 × 2 matrix and then transpose it. Transposing makes the tensor non-contiguous. If we check .is_contiguous() on this transposed tensor, it returns False. Now, if we attempt to apply view to reshape this non-contiguous tensor into a shape like 6 × 4, we encounter an error saying that the view size is not compatible with the input tensor size and stride. This shows that view cannot handle non-contiguous data. However, if we instead use reshape, it works perfectly. This is because reshape internally creates a contiguous copy of the tensor and then performs the reshape, so it can handle non-contiguous data as well. Printing the result shows that the tensor has been reshaped into 6 × 4 successfully.

Although reshape handles such cases, operations on contiguous tensors are faster because memory is accessed sequentially. Since view strictly requires contiguous memory, if the tensor is not contiguous we can first call .contiguous() on it. For example, if we take the transposed tensor and apply .contiguous(), it reorganizes the data into a contiguous layout. After this, we can safely use view to reshape it without error. This results in the same outcome as reshape, but now with the efficiency benefits of working on contiguous memory.

Through these examples, we clearly understand the differences between view and reshape. Both can reshape tensors into new dimensions, but view only works with contiguous tensors, whereas reshape is more flexible as it can handle non-contiguous data by making a copy. When performance is critical, it’s better to ensure tensors are contiguous and use view; otherwise, reshape is a safer, more general option.

**J) Stack Operation:**

We will start by importing torch as usual, so we will write "import torch". Next, we will create two tensors as mentioned in the illustration. For this, we use "torch.tensor" and create a simple tensor with the values one, two, and three, which we will name as tensor one. Similarly, we will create another tensor with different values, namely four, five, and six, just to show some difference; otherwise, we won’t be able to distinguish whether we are appending on dimension zero or dimension one. This second tensor will be called tensor two.

Now, we will also print the dimension of the tensor. For this, we write "tensor1.ndim" and also print the shape with "tensor1.shape". Alternatively, we could also use "size" because shape internally uses size. Since tensor one and tensor two are the same in terms of dimension, we only need to print the properties of tensor one. From this, we can see that the tensor is of dimension zero, with a dimension of one and a size of three. This means it has a total of three elements, which gives a shape of one by three.

Next, we will stack the tensors. For this, we use "torch.stack" and mention the tensors we want to stack, i.e., tensor one and tensor two, as arguments. We will specify that we want to stack along dimension zero, and save this into a variable named stacked tensor dimension zero. To verify the result, we will print the shape using "stacked_tensor_dim0.shape" and also print the stacked tensor itself. Then, we will copy and paste this code to create a stacked version along dimension one using the same two tensors, and print the output to see the difference.

When stacking along dimension zero, the shape of the tensor becomes two by three, meaning we have two rows and each row contains three elements. Tensor one (values 1, 2, 3) and tensor two (values 4, 5, 6) are stacked to form a new dimension. Thus, we now have row zero and row one. On the other hand, when stacking along dimension one, it means we are pairing the first index of tensor one with the first index of tensor two, then the second index of tensor one with the second index of tensor two, and so on. This results in a single tensor where one is paired with four, two with five, and three with six. To put all values into a single tensor, they are enclosed within a new dimension. This stacked tensor along dimension one shows values combined column-wise.

We can also stack more than two tensors. For example, if we create another tensor called tensor three with values seven, eight, and nine, and use the same stack function with tensors one, two, and three along dimension zero, we will get a new tensor containing all three. We can label this as new stacked tensor to avoid overwriting the previous variable. Printing it shows the stacked result: 1, 2, 3 from tensor one, 4, 5, 6 from tensor two, and 7, 8, 9 from tensor three, all stacked along a new dimension. The same operation can also be done for dimension one.

Stacking is not limited to vectors (1D tensors); we can also stack 2D tensors. To demonstrate this, we create a 2D tensor, where the first row has two columns and the second row also has two columns. Let’s name it tensor one. Next, we create another 2D tensor, starting with values five and six for the first row, and seven and eight for the second row, naming it tensor two. Using "torch.stack" along dimension zero with these two tensors, we save the result as stacked tensor 2D and print its shape.

The result shows that tensor one (values 1, 2, 3, 4 arranged in a 2D structure) is taken as a single entity, and tensor two (values 5, 6, 7, 8 in 2D) is stacked along a new dimension. This creates a 3D tensor. Printing the dimension before and after stacking reveals that before stacking, the tensor’s dimension was two, but after stacking, we now have three. This is because both 2D tensors are enclosed in another dimension, making the final tensor 3D.

To make this clearer, consider tensor one as [[1, 2], [3, 4]] and tensor two as [[5, 6], [7, 8]]. In this case, dimension zero refers to the outermost structure (rows), dimension one to the inner lists, and stacking along dimension zero means both tensors are combined along this outer level. This results in [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], which is now 3D. The same concept applies to the 1D case, where stacking along dimension zero meant combining full tensors into a new dimension.

We can also stack along dimension one or two. For instance, when stacking along dimension one, the first rows of tensor one and tensor two are combined: (1, 2) with (5, 6), and the second rows are combined as (3, 4) with (7, 8). This produces [[1, 2, 5, 6], [3, 4, 7, 8]]. If we stack along dimension two (the innermost level, the columns), the output pairs elements column-wise: (1, 5), (2, 6), (3, 7), and (4, 8). This is verified by running it in Google Colab. If we attempt stacking along dimension three, it throws an error since the tensor only has up to dimension two, and the error states that the dimension is out of range.

Apart from stack, PyTorch also provides another method for combining tensors, called concatenation (cat). To use it, we write "torch.cat" and pass the tensors we want to concatenate along with the dimension. For example, concatenating tensor one and tensor two along dimension zero results in a 2D tensor where the rows are simply appended: [[1, 2], [3, 4], [5, 6], [7, 8]]. Unlike stack, cat does not create a new dimension; it preserves the number of dimensions. Printing the shape and ndim confirms that the output remains 2D.

If we concatenate along dimension one instead, the columns are appended side-by-side. In this case, the output becomes [[1, 2, 5, 6], [3, 4, 7, 8]]. Thus, concatenation along dimension zero appends rows, while concatenation along dimension one appends columns. The key difference between stack and cat is that stack creates a new dimension to hold the tensors, whereas cat merges them directly while keeping the same dimensionality.

#### **K) Understanding Pytorch Neural Network Components**

In this lecture we will understand PyTorch neural network components. These are building blocks that power deep learning models. Whether you are training a simple classifier or a complex neural network, PyTorch provides additional tools to define, train and optimize models.

So let us understand what are those key components. We will start by writing components of a simple neural network. To get started, we will increase the size and write in a docstring the basic components of a simple neural network. This will be a breakdown of a simple neural network.

## 1. Breakdown of a simple neural network

If we talk about a neural network, it has an input. We denote the input as x. It has weights, denoted as w. The weights can be written as w1, w2, w3, etc. Then we have biases as well, denoted as b. This bias term is added to the linear combination. Next, we have an activation function. We denote the activation function as a. This activation can be sigmoid, ReLU, or any other non-linearity. Finally, we have the output, denoted as y.

To create a forward propagation pass, we start with the input x. The formula for a linear regression is y = mx + c. In neural networks, this can be generalized as y = wx + b, because we have weights w and biases b applied to the input x.

The first step is to compute z = w1·x + b1, which is the linear operation. The output of z then passes through an activation function to introduce non-linearity into the model. So we compute a = activation(z), where the activation could be sigmoid, ReLU, or any other function. The output after activation is denoted as z′ (z dash).

Next, this transformed output is used as input to another node: z2 = w2·z′ + b2. This gives the final output y. Thus, we have two nodes in this simple network: the first node followed by an activation function, and the second node without an activation function, directly producing y.

The components required for this forward propagation are the input x, the weights w, the biases b, the activation function a, and the final output y. This process of computing the output is called forward propagation.

Once forward propagation is complete, we need to compute the loss. The loss is calculated as the difference between the predicted output and the actual target output. After calculating the loss, we need to compute gradients using backpropagation. With backpropagation, gradients of weights and biases are calculated. Finally, an optimizer is required to update the parameters using the computed gradients.

Thus, the key components of a neural network are: inputs, weights and biases, activation functions, forward propagation, loss function, backpropagation, and optimizers.

X --> input; Wx --> Weights; bx --> bias; A --> Activation function; Y ---> Output; Z = W1.X + b1; Z' = A(Z); Y = W2.Z' + b2

# We also use Loss function, Backpropagation and optimizer

## 2. Components of pytorch

Now let us see what components are provided by PyTorch as a library. The first component is the base class for defining custom models, which is torch.nn.Module. This class is always inherited when creating custom models. The second component is the fully connected (or dense) layer, defined using torch.nn.Linear. If we want activation functions, we can use torch.nn.ReLU, torch.nn.Sigmoid, or others. Optimizers for updating weights are available in torch.optim, such as torch.optim.SGD. Loss functions are available in torch.nn, for example torch.nn.CrossEntropyLoss.

For loading data in batches, PyTorch provides torch.utils.data.DataLoader. This class allows loading data efficiently in batches and supports GPU acceleration.

(i) Base class for defining customer models : torch.nn.Module
  
(ii) Fully connected (dense) layers : toch.nn.Linear
  
(iii) Activation fucntion : torch.nn.ReLU
  
(iv) Optimiser : torch.optim
  
(v) Loss function : torch.nn.CrossEntropyLoss
  
(vi) Loads data in batch : torch.utils.data.DataLoader

## 3. Different ways to create neural network

1. Function : Flexible, harder to interpret

2. Sequential : nn.Sequential

There are also different ways to define neural networks in PyTorch. The Functional API is a flexible approach where we directly define operations on tensors, allowing for more customization (flexible and allows custom operations on tensors, making it suitable for more complex architectures). The Sequential API is a structured approach where layers are stacked in a linear order using torch.nn.Sequential, making it simpler to define straightforward models (more structured, where layers are stacked linearly in order, making it easier for simpler models).

## 4. Finally, when building a PyTorch model, we usually start by importing required components:

import torch
import torch.nn as nn
import torch.optim as optim

### Creating Neural Network using Functional API:

Once we have all the imports required, we will start by creating our simple neural network using the Functional API. For the Functional API, we need to create a class, and this will be defined as a model name. Suppose we are creating a model called simple, and this will be our model name. This class needs to inherit nn.Module class. That is the first requirement, as it inherits from nn.Module, which is the base class for all models in PyTorch. Whenever we are creating any custom model, this should be the base class because it has components required to create a neural network.

The second step is that we need to define the __init__ method. We will mention def __init__ and the first argument should be self. This method will initialize model layers. Suppose we are creating a linear network or a simple neural network with only two layers. In the __init__, we can mention a few arguments such as input size, hidden size, and output size. When we initialize our model in Python, we will pass these parameters. The model should have this input size, these many hidden layers (or hidden size), and an output size.

We then have to call the super() function. The first argument to super will be our class name and the second argument is self. We need to call __init__ to initialize the base class. The super function calls the constructor of the parent class, also known as nn.Module. In PyTorch, every custom model inherits from nn.Module, which is the base class. This ensures that the __init__ method of nn.Module is executed before initializing anything else. First, we initialize nn.Module, and then we define our own components.

The first component is self.fc1, which is our first fully connected layer created using nn.Linear. It accepts the input size and the number of output neurons, which in this case is the hidden size. We also define an activation function self.ReLU = nn.ReLU(). Next, we define self.fc2, which will be our second and final fully connected layer using nn.Linear. The input neurons to this will be the hidden size, and the output neurons will be the output size.

So far, we have created the components of our model: input size, hidden size, output size, fully connected layers, and an activation function. To visualize this: suppose our input size is 4. The hidden size is a certain number of neurons, and the output size is, say, 3 neurons. The first fully connected layer fc1 maps from input size to hidden size, and the second fully connected layer fc2 maps from hidden size to output size.

To perform forward propagation, we need to define the forward function. The forward function accepts an input x, which should be a tensor of shape equal to the input size. Inside forward, we first pass x through self.fc1, store the result back in x, apply the ReLU activation function on it, and store it back again in x. Then, we pass it through self.fc2 and again store the result in x. Finally, we return x. This forms the forward propagation pipeline.

import torch
import torch.nn as nn
import torch.optim as optim

class Simple(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Simple, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        return x

### Creating Neural Network using Sequential API:

Now, if we want to create the same neural network using Sequential API, we can define the model differently. With Sequential API, instead of creating each layer separately, we define the sequence of operations directly. We do not define multiple separate components but rather specify them in one sequence.

class SimpleNNSequential(nn.Module):

  def __init__(self, input_size,hidden_size, output_size):
    super(SimpleNN, self).__init__()
    self.network = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )

  def forward(self, x):
    return self.network(x)

In Sequential API, all layers are executed in order, and we cannot change the flow once defined. It is best suited for straightforward architectures where the data flows sequentially from one layer to the next. However, if we need flexibility, such as reusing outputs from previous layers or adding skip connections (as seen in architectures like ResNet), then Sequential cannot handle it. Functional API, on the other hand, provides full flexibility to define custom flows.

## When we learn advanced CNN, we will learn that architecture is not always CNN

## When we move to advanced concepts of Convolutional Neural Networks (CNNs), it becomes clear that architectures are not always purely sequential. Unlike simple feed-forward networks, CNNs often include additional connections where outputs from earlier layers are reused at later stages in the network.

To illustrate this, consider a basic neural network consisting of four hidden layers. The input flows into hidden layer one, its output is passed to hidden layer two, the output of layer two flows into layer three, and so on until hidden layer four produces the final output. In this case, the architecture is strictly sequential. Since each layer passes its output directly to the next one without deviation, a sequential model is sufficient to implement it.

However, in more complex architectures, the flow is not strictly linear. For example, an input may need to be fed directly into an intermediate hidden layer, or the output of one hidden layer might be reused along with the output of another hidden layer. This introduces additional connections that deviate from the standard sequential order. An example of this can be seen in ResNet (Residual Networks), where outputs from previous layers are added to the outputs of deeper layers to enable residual learning.

In such cases, sequential models are no longer flexible enough to handle these additional pathways. With sequential, each layer depends only on the previous one, so reusing intermediate outputs is not possible. On the other hand, functional models provide this flexibility. Each layer is defined separately (e.g., self.fc1, self.fc2, etc.), and their outputs can be stored and reused at any point. This allows operations such as combining outputs from multiple layers, for example, self.fc1 + self.fc2 + self.fc3.

The key difference is that everything possible with sequential can also be implemented using functional, but not everything possible with functional can be replicated in sequential. If the network design is simple and strictly linear, a sequential model is sufficient. For architectures with skip connections, multiple inputs, or reused outputs, the functional API becomes necessary.

Thus, sequential models are best suited for straightforward architectures, while functional models provide the flexibility required for advanced architectures such as ResNet and other modern CNNs.

## To compare: if the architecture is simple and sequential, we can use Sequential API. Otherwise, we should use Functional API. Everything that can be implemented using Sequential can also be implemented using Functional. However, everything that can be done with Functional cannot always be done with Sequential.

### We will now pick the functional network and begin model training. Although this is a dummy network, it accepts an input size, has a hidden size, and produces an output size. We will initialize the network and observe how the training process looks in practice.

**Initializing the Network**

Before creating dummy data, the first step is to initialize the model. Initializing a model means defining its components, such as self.fc1, self.ReLU, and self.fc2. Each layer has arguments such as input size, hidden size, and output size, which must be passed when the model is created.

We create a class called Simple for our model and pass the initial parameters. For this example, the input size is 4, the hidden size is 8, and the output size is 3. This corresponds to the architecture we have seen in the diagram, where the network accepts four input features, passes them through a hidden layer of eight neurons, and produces three output neurons.

model_fun = Simple(input_size=4, hidden_size=8, output_size=3); print(model_fun)

Printing the model displays the layers contained in it. The architecture follows the forward propagation defined in the forward method and shows how the layers are connected.

**Creating Dummy Data:**

Next, we create dummy data for training. The input tensor x will have ten samples, each with four features, corresponding to the input size of the network.

x = torch.rand(10, 4)   # 10 samples, 4 features each

We also define the target labels y. Since we have three output classes, we generate ten random integer labels ranging between 0 and 2.

y = torch.randint(0, 3, (10,))   # 10 random class labels (0, 1, or 2)

Thus, x contains ten rows of input data, and y contains ten corresponding class labels.

**Defining the Loss Function**

We use Cross Entropy Loss as the loss function. In PyTorch, nn.CrossEntropyLoss() automatically combines log softmax and negative log-likelihood loss. This means we do not need to explicitly define a softmax activation function at the output layer. Cross entropy loss inherently applies the softmax operation to the final outputs before computing the loss.

criterion = nn.CrossEntropyLoss()

This loss function is stored in the variable criterion, which is a standard naming convention often seen in PyTorch code repositories.

**Defining the Optimizer**

We now define the optimizer, which updates the weights and biases of the model during training. We use the Adam optimizer with a learning rate of 0.01. All trainable parameters of the model are passed to the optimizer.

optimizer = optim.Adam(model_fun.parameters(), lr=0.01)

**Verifying Input and Labels**

At this stage, we can print x and y to verify that the input tensor contains ten samples and the target tensor contains ten class labels ranging between 0 and 2. The output layer of the model has three neurons, one for each class. Each neuron is responsible for predicting the probability of one of the three classes.

## If a network were required to classify between ten classes, the output layer would need ten neurons. Since our classification problem involves three classes, the output layer has three neurons, with each neuron responsible for predicting one class label.

**Defining the Training Loop**

We now define the training loop. The number of epochs is set to 50, which means the training process will iterate 50 times. For each epoch, the following steps are performed:

(i) Clear gradients using optimizer.zero_grad().

(ii) Forward pass the input x through the model to obtain predictions.

(iii) Compute loss by comparing the predicted output with the actual labels y.

(iv) Backpropagate the loss using loss.backward().

(v) Update parameters using optimizer.step().

(vi) Print loss every 10 epochs to monitor progress.

**Codes for Training:**

epoch = 120

for e in range(epoch):
  optimizer.zero_grad()
  outputs = model_func(X)
  loss = criterion(outputs, Y)
  loss.backward()
  optimizer.step()

  if (e+1) % 10 == 0:
    print(f"Epoch [{e+1}]/50, Loss : {loss.item() :.4f}")

 **Training Results:**

On running the above loop, we observe that the loss decreases over epochs. For example, at epoch 10 the loss is approximately 1.11, at epoch 20 it reduces to 0.93, at epoch 30 it becomes 0.77, and by epoch 50 it drops to around 0.51. Increasing the number of epochs to 120 results in further loss reduction, eventually reaching values close to 0.03 after reinitializing the model and rerunning the training loop.

 **Conclusion:**

This demonstrates that the model is able to learn patterns from the input data. The training process consists of forward propagation, loss calculation, backpropagation, and parameter updates. PyTorch handles the underlying details of backpropagation and optimization automatically.

**L) Creating Linear Regression Model with Pytorch Components**

In this lecture, we are going to implement a linear regression model. Instead of using machine learning libraries like scikit-learn or sklearn, we will use PyTorch. To begin, any machine learning problem statement has multiple steps. 

# They are: Data gathering, Data preprocessing, Feature engineering, Model training, Testing

The first step is data gathering, where we collect data from sources such as the web or multiple SQL tables and combine it in a single space. The second step is data preprocessing, which involves changing data types, removing unnecessary columns, and keeping only the relevant ones for analysis. The third step is feature engineering. For example, if we have a column of a person's date of birth, we can derive additional features such as the person's age or the current day of the week. Once we have the relevant features, the fourth step is model training. After the model is trained, we evaluate it on a test set or production data to see how well it performs.

For this lecture, we use a dataset related to medical costs and personal data. The dataset contains columns such as age, sex, BMI, number of children, smoker status, region, and individual medical charges. Age represents the age of the primary beneficiary. Sex indicates the gender of the insurance holder. BMI represents the body mass index. Children represents the number of dependents. Smoker indicates whether the person smokes or not. Region represents the beneficiary’s residency area in the US, including northeast, southeast, southwest, or northwest. Charges represent the individual medical costs billed by the insurance. The dataset has a total of 1,338 rows.

To start, we load the dataset into Google Colab. Instead of downloading it manually from Kaggle, we install Kaggle in Colab using pip install kaggle. We then import Kaggle, specify the dataset to download using kaggle.datasets.download('dataset-name'), store it in a specified path, and print the path using print(path). We can check the dataset directory using import os and os.listdir(path). Once the file insurance.csv is located, we load it into a DataFrame using import pandas as pd and df = pd.read_csv(os.path.join(path, 'insurance.csv')). We inspect the data using df.head() and df.info(). To understand the statistics of numerical columns such as age, BMI, children, and charges, we use df.describe().

Next, we import PyTorch for model training using import torch, import torch.nn as nn, and import torch.optim as optim. We also import functions from sklearn like from sklearn.model_selection import train_test_split for splitting the data, LabelEncoder for encoding categorical variables, and StandardScaler for feature scaling. We split the dataset into training and test sets using train_df, test_df = train_test_split(df, test_size=0.2, random_state=42).

For encoding categorical variables such as sex, smoker, and region, we initialize a dictionary Labelencoder = {}. For each column, we create a label encoder, fit it on the training set using train_df[col] = L.fit_transform(train_df[col]), and then transform the test set using test_df[col] = L.transform(test_df[col]). The trained label encoders are stored in the dictionary for future use.

To define the features and target, we create X_train by dropping the charges column from the training DataFrame using X_train = train_df.drop(['charges'], axis=1) and y_train as y_train = train_df['charges']. Similarly, for the test set, we define X_test = test_df.drop(['charges'], axis=1) and y_test = test_df['charges'].

Next, we normalize the features to help the model learn faster. We initialize a scaler using scaler = StandardScaler(), normalize the training data with X_train = scaler.fit_transform(X_train), and normalize the test data using X_test = scaler.transform(X_test). 

#### After normalization, we convert all numpy arrays to PyTorch tensors using torch.tensor(X_train, dtype=torch.float32) and torch.tensor(y_train.values, dtype=torch.float32).view(-1,1) for the target, and similarly for X_test and y_test.

# Define Neural Network Model Code:

class SimpleNNRegressionModel(nn.Module):
  def __init__(self, input_dim):
    super(SimpleNNRegressionModel, self).__init__()
    self.network = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

  def forward(self, x):
    return self.network(x)

We then define the neural network model. We create a class class SimpleRegression(nn.Module): which inherits from nn.Module. In the __init__ method, we initialize the neural network with self.network = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64,128), nn.ReLU(), nn.Linear(128,1)). The forward method defines def forward(self, x): return self.network(x). The input dimension input_dim is the number of features, which we extract from X_train_tensor.shape[1]. We then initialize the model using model = SimpleRegression(input_dim) and define the loss function as mean squared error using criterion = nn.MSELoss(). The optimizer is Adam with a learning rate of 0.01 using optimizer = optim.Adam(model.parameters(), lr=0.01).

We then create the training loop. We define the number of epochs, for example, epochs = 10000, and iterate over each epoch using for epoch in range(epochs):. We set the model to training mode using model.train(). We clear gradients using optimizer.zero_grad(), compute predictions y_pred = model(X_train_tensor), calculate the loss loss = criterion(y_pred, y_train_tensor), backpropagate using loss.backward(), and update the weights with optimizer.step(). We print training statistics every 100 epochs using if (epoch+1) % 100 == 0: print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}').

Once training is complete, we set the model to evaluation mode using model.eval(), and predict on the test set using y_pred = model(X_test_tensor).detach().numpy(). We calculate evaluation metrics using from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score. RMSE is computed as rmse = mean_squared_error(y_test, y_pred) ** 0.5, MAE as mae = mean_absolute_error(y_test, y_pred), and R2 as r2 = r2_score(y_test, y_pred). These metrics help assess how well the model predicts the charges.

Finally, we define a prediction function def predict_charges(age, sex, bmi, children, smoker, region): which takes the input parameters, constructs a DataFrame with pd.DataFrame, label encodes categorical values using the stored encoders, scales the features with the trained scaler, converts the input to a tensor torch.tensor(input_data, dtype=torch.float32), passes it to the model predicted_charge = model(input_tensor).item(), and returns the predicted insurance charge. We test this function with sample input such as age=19, sex='female', bmi=27.9, children=0, smoker='yes', region='southwest' and print the predicted insurance charge. By adjusting input features like age and smoker status, we can observe changes in the predicted charge.

In summary, we successfully created a linear regression model using PyTorch. We first gathered and loaded the dataset, explored its statistics, split it into train and test sets, encoded categorical variables, scaled the features, and converted them to tensors. We defined a neural network with input, hidden, and output layers, specified the loss function and optimizer, trained the model, evaluated it using RMSE, MAE, and R2 metrics, and finally created a function for predicting insurance charges based on new input data. This end-to-end process demonstrates building a regression model entirely using PyTorch without relying on sklearn for modeling.

### **M) Multi-Class Classification with Pytorch using Custom Neural Network**

In the previous lecture, we have seen how to perform regression task with the help of neural networks. In this class, we will see how to perform multi-class classification with the help of PyTorch using our own custom neural networks. Let's get started.

So the data set which we are going to use for this task is iris data set. You can see this is archive.ix.uci.edu. And you can see in this dataset we have iris data set. So let's talk about the characteristics of this data set. It is a tabular subject area is biology. It is a classification task. We are going to do classification task. And there are total number of features here is four. It means we have total number of four columns and 150 rows in total.

So here if we talk about multiple values you can see it has variable called sepal length which has continuous value. It means it is a numerical and the unit is centimeter. Then another is sepal width and petal length and petal width. And the class so given sepal width and length and the petal width and length. We have to predict what is the class. And there are total number of three classes here which we can see. All of them are a class of iris plant, one is setosa, another is versicolor and then virginica. So in total we have total number of three classes.

Given all of these features that is this will be our x variable. This will be our y variable. We will be creating a neural network which will predict three neurons not just single. Because this is not a regression task. This is a classification task. We will see how to do that. And we will be loading this dataset with the help of sklearn. Okay. We are not going to download any CSV file here because sklearn directly provides it, because our main task is to how to perform this with the help of PyTorch.

Total number of classes. This is an sklearn page. You can check load Iris. When we use load Iris, we will be able to load this data set Sklearn.datasets load iris. And then this is the iris data set. It is classic and very easy. Multi-Class classification data set Samples per class. Each class has a 50 like total number of rows. It means this data set is pretty balanced. It means in total we have three classes. In each class we have 50 rows. It means 150 total. So each class is having 50, which is pretty balanced. Dimensionality is four. It means number of features. Total number of features is either it is real or positive. These are the features.

And we can see all of this information. This is not the information of setosa but this is the information of like how to load it. So this is how we load it data or target and other things. Okay. We will see how to do this in PyTorch. So first of all we will start by loading lot of libraries which we need.

First of all I am going to start with ;import torch; then ;import torch.nn; And we are going to use this ;torch.nn; while we are creating our own custom neural network architecture. Then ;import torch.optim; This is for optimizer ;torch.optim; As Optim and then ;import pandas as pd;. ;Import numpy as np;. From ;sklearn.datasets; We are going to ;import load_iris;. We have just seen this in the documentation. Then from ;sklearn.model_selection; We are going to ;import train_test_split;. We don't want to waste time writing our own train test split. We will see that in later classes. But for now we are going to stick with PyTorch and going to create the whole pipeline. Then from ;sklearn.preprocessing; we are going to ;import StandardScaler; and then ;LabelEncoder;.

And let's try to run it. Okay so I've made this spelling wrong. It is sklearn. And now everything is loaded successfully. So we are going to use this method ;load_iris;. And we are going to store this in iris variable. Then we are going to create from iris. We are going to create data frame. So I am going to write ;pd.DataFrame;. And in the data frame I'm going to pass ;iris.data;. And then columns I'm going to store in. This is already available in iris under features name. Feature names. So when you mention ;iris.feature_names; all the feature names that is sepal width, sepal length, petal width, petal length. All of these feature names will come under column and all the data will by default go to ;iris.data;.

We will be creating ;pd.DataFrame; from here and create under df. Then we are going to mention ;df["target"] = iris.target;. So all the target that is the class labels will come under ;iris.target;. And let's run it. Now we will see this data frame as well. That is ;df.head();. And it will give us information about our pandas data frame. So you can see this is sepal width sepal length petal width petal length. All of those information was present in ;iris.data;. And these are the feature names right. But this particular that is a target column was not available. It is available in ;iris.target;, which we will store in our target column.

Next thing is, we are going to separate this data set and split into train and test split. I am going to mention ;train_test_split;. This is from again sklearn. And we are going to pass our data frame. And then we will mention our test size. I want test size to be on a 20%. So all the training data set will have 80% of the data set random state. I am going to mention it as 42 and then stratify. That is the how we want to split it. I am going to mention ;stratify=df["target"];. And we have to store this train test split into. We will mention ;train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"]);.

If you zoom in a bit. Now this looks better. Now that we have splitted our data set into train and test df, we will split train df into X train and Y train and we will split test df into x test and y test. Let's see how to do that. So I will mention train df which is our data frame. And I'm going to drop column. That is ;train_df.drop(columns=["target"]);. And I'm going to mention ;train_df["target"];. Same thing. What I will do is, uh, this particular drop, I want to drop this here, and then I want to store this particular split into X train, and then this will be Y train. So this particular portion will be X train. This particular part will be our Y train.

There is no need of bracket here. Similarly I will repeat the same step for test df. This will be X test. This will be y test and column name will be same. But instead of train df I will be using test df and here test df. So here in x train I have everything except target column. In y train I have all the target values. Okay, so in y train I have all the target values. Again In x test I have everything except target column. In y test I only have column target of the test data frame. I will run it and it is saying column column C, o, L, u, m and s. It should be columns. So here again that is the same issue. Now it is correct. So we need to mention columns because there can be more than one columns here.

Now we will be using standard scalar. And we are going to perform scaling on our both Xtrain and Xtest. I am going to create this object and store it into scalar then ;scalar.fit_transform(X_train);. We are going to fit transform on our X train. And store again in X train. And then we are going to just transform whatever we have learned from our X train data. We are going to do transformation based on that. To from let me mention this is going to be X test and we are going to store again, we are going to do our transformation on X test and store again into X test. Now this part is also done.

### So now we are going to because this is still a data frame object pandas data frame or numpy. What we are going to do we are going to convert this into tensor. I'm going to mention ;torch.tensor;. And here I will use X train. You have to convert everything X train, Y train, X test and Y test all of them to tensor.

I’m going to mention ;dtype=torch.float32;. And I’m going to store into x train tensor. I’m going to copy and paste this for all and for y test expressed. Let me mention X test. This is going to be same but here it will be X test. So for x test also this is done for Y train. This is going to be long. ;torch.long;. So long for classification. And we are going to store into y train tensor. Similarly we will repeat it and this will be y test. And this is y test tensor. This should also be ;torch.long;. We have to make it values. And also for y test we have to mention values. Now it should work. So this is completed. This is successful.

### Now the next thing we are going to create is we need a model. We will start by creating our model. So we will mention iris classifier. We have to ;import nn.Module;. That is the first step. Second step is we need to define the init class. I didn’t mean to run it and ;def init;. Then we have to mention self. Let me first remove this error. This does not look good. And in ;nn.Module; it is capital. Now this is better.

So in the init we need few things. We need input dimension which we are going to pass when we are creating this model. I will mention input dimension then hidden dimension which we are also going to pass, and also output dimension. Output dimension in our case will be three because we have total number of three classes. And then we need to call super class to initialize an inner module with some correct parameters. The first argument is our class name, second is self. Then we need to initialize. And all this is done.

We are going to create our network using ;nn.Sequential;. So I will mention ;self.network = nn.Sequential(...);. And in that one we will start creating our network. So first one is not linear. We want to create an ;nn.Linear;. And we will mention input dimension. Then hidden dimension. This part is done. Next is we want to append ;nn.ReLU();. Next is I’m going to copy this and ;nn.Linear;. So this will start with hidden dimension. And I will still connect to hidden dimension. That will be same number of neurons in the middle hidden layer. And then an ;nn.ReLU();. And at the end I will repeat hidden layer with output layer the number of output dimensions. So now our model is ready.

Next thing is we have to create forward function. Forward method. It will take x as input. And we need to pass that particular x to ;self.network; which is our model. And we will return whatever the output is. And that is it.

Now what we will do, we will create input dimension. Input dimension. And before creating input dimension we can check the shape ;X_train.shape;. And let me print. This should be ;X_train.shape;. This is a tuple. We cannot print it. This is not callable. Yes. Let us see. This is ;X_train.shape;. You can see total number of values we have in x is 120. But total number of features in each of these values is four. So if I mention ;X_train.shape[1]; then I will just get four. And this I want to be my input dimension. I will save this into input dimension as ;X_train.shape[1];.

And hidden dimension. We can mention by ourself because this is something that we will select. I’m going to mention our hidden dimension as 16. You can choose any number. Our output dimension. You already know how many number of classes we have in our data set. I’m going to mention three. Input dimension will be four. Hidden dimensions will be. That is, total number of neurons in the middle layer is going to be 16. Then that 16 is going to connect it with another 16. That is this layer that is four will be connected to 16, 16 will be connected to another 16. Then that 16 will be connected to your output dimension. And in between we have ReLU and let’s run it. So this is completed.

Now we are going to create our iris classifier. We are going to initialize this neural network. So I’m going to mention input dimension. That is the first argument. Second dimension is hidden dimension. Third is output dimension. I’m going to store this as a model. Now we have to define our loss function and optimizer. I will write ;nn.CrossEntropyLoss();. So it should give us yes cross entropy loss. And we are going to store in a variable called as criterion. And then we will mention our optimizer. And we are going to use ;optim.Adam;. And here in Adam we want to optimize what we want to optimize our model parameters and then learning rate. We also need to mention here. So I’m going to start with 0.01. And I will store this into optimizer. And let’s run it. Okay. Spelling is wrong. It should be parameters. Now it is able to load all the parameters and will store into optimizer.

Now we will start our model training. Here I will write the process of model training. So we will write train the model. First of all, I’m going to mention how many epochs we want to train. Suppose I want to start with 500 and see how the output looks like. Then ;for epoch in range(epochs):;. First of all, I will start and mention that I want to train our model. So first argument is this should be epochs and first argument should be ;model.train();. I am setting our model to training mode. Second is optimizer. We have to refresh the optimizer. We have to clean the optimizer. So I will mention ;optimizer.zero_grad();. Next is we have to create a forward pass. It means in the model we have to pass our data. Our full data is x train tensor. So we are passing all the 120 rows of X tensor and model will predict and we will store into predictions. Predictions.

And from this prediction we are going to calculate the loss. For that we are going to use criterion. It will take predictions and the actual value. The actual value is y train tensor. And it will give us loss. With using this loss we have to calculate the gradients. So we will do ;loss.backward();. This will calculate the gradients. And now with the help of optimizer we will update those gradients in our model or parameter. I’m going to write ;optimizer.step();.

Once this is done I’m going to write a loop to check the progress. I’m going to mention ;if (epoch+1) % 50 == 0:;. So every 50 step I want to check the progress because we have total number of 500 epochs. I want to print the stats. What is my current accuracy. What is my current loss and other things. So for simplicity we will just mention epoch and what is the current loss? I will mention ;f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}";. And that is it. Let’s try to run it and see how the loss looks like. It is giving us issue of the forward. There is some issue with the forward. Okay. There is an indentation which is not correct. Now everything should work correctly. Former precision. Missing precision. Okay. Okay. Okay. So now we are able to decrease the loss which is from 0.03. And it is going to 0.02. So the loss was already low. Our model was pretty good at the start because of the number of epochs. If you want to initialize things again and see on every 10th epoch, then we will be able to see the progress better.

Now you can see the model started with 0.5, and at the end of this iteration that is epoch 500, we are able to see 0.0003. That is from 0.5. We went all the way till 0.0003. It means our model is learning okay. Our model has learned the patterns. Now what we will do, we will write a prediction script and we will check the accuracy. But before that, we will predict on our test set. We already have a test set. And let’s see how our model performs on that. I’m going to write ;model.eval();. And here we have to set that. We don’t want to perform any update. We will mention ;torch.no_grad();. We don’t want to update any gradient. And within it we will mention model and we will pass our X test tensor. And it will give us predictions which we are going to store in y prediction.

Then with the help of torch max that is ;torch.argmax();. And we will get the class with the highest probability using ;torch.argmax(y_pred, dim=1);. We will store into y prediction labels. Because if you just see the prediction, it will have total number of three classes. But we only want the maximum. That is the total highest probability out of that. And we have to use torch.argmax to get a single value from this particular prediction. Okay. Because each prediction will be a tensor of size three because we have total number of three classes. So torch using torch.argmax, we are going to get the class with the highest probability and we are going to store into y labels.

So once this is done what we are going to do we are going to calculate the accuracy. And we will be using y labels just y prediction labels. We will be using y prediction labels. And we’ll compare it with y test tensor. And whenever it is equal we are going to do the sum. And at the end we will calculate the total number of that sum it will give an output, and we have to compare that with total number of y test values. ;Y_test_tensor.size(0);. The accuracy is total number of correct predictions divided by total number of predictions. So this will give us total number of correct predictions by total number of prediction. Total number of values. Okay. So if you have hundred value here in total we have hundred rows in y test tensor. But our model predicted only 50 of them correctly. It means model is 50% accurate. This is what we are trying to do here. Then I will store this into a variable called accuracy. And ;print(f"Test Accuracy: {accuracy:.4f}");.

Let’s try to run it. So our model is predicting 96% accurately on test set. Okay. So now we will write we will create a function to do inference on our model not on the data set but on a single set on a single row. We will be creating ;predict_iris;. This is a function which we are creating. It will take few things like sepal length, sepal width, then petal length, then petal width. So what we’re going to do we are first going to create an array. And I’m going to mention all these values, that is the argument of this function and paste it here. That will be our numpy array. And this will be our input data.

But we have to convert this. We have to transform this data just like we have done for original data that is X train and Y train. Also X test and Y test. What we are going to do, we are going to load our scaler and scaler. We have already done the fit transform. So we are going to use that scaler and do transform. Transform on what? Transform on input data and store again into input data. So once our transformation is done the next thing is we are going to convert this into tensor because our model expects tensor data type. We are going to use ;torch.tensor(input_data, dtype=torch.float32);. And we will store into input data tensor or input tensor.

So once this is ready we have to set our model to evaluation mode. Now I will copy this code this particular code. To get the indentation correct. I will use our model and pass input tensor and we will get prediction. For that prediction we are going to use torch.argmax. And at the end we directly want item. Previously we were taking the item at the end because there were a lot of values. Now we only have single value. So we can use item here itself and then it will give us predicted label. Or you can say predicted class. And whatever it gives we are going to return because this is this will give us 0, 1 or 2. That is the predicted class.

And we want to convert this particular class that is 0, 1 or 2 to the data set available here. That is, we want to either say setosa or versicolor or virginica. But this data is present in where this data is present in the iris data itself. So what we are going to mention ;iris.target_names[predicted_class];. Target name is a list okay. If you want to see before doing this you can check ;iris.target_names;. You can see this is an array. And we are going to use predicted class as our index. So if predicted class is 0 it means model is predicting setosa. If predicted class is 2, that is virginica. We are going to use this particular list and pass the index. Where do we have index? We have index in predicted class and this is what we are going to return.

Let’s try to run it. Our function is ready. We just have to call this function. We just have to call this function ;predict_iris(5.1, 3.5, 1.4, 0.2);. If you want to check the range of this data set, we are passing the same thing. Okay, 5.1, 3.5, 1.4 and 0.2. It should be target should be predicted as zero and zero is setosa. We have also printed the target here. It should predict setosa. Given all these arguments it will return us the predicted class, which I’m going to store here and predicted class and print at the end. ;print("Predicted Species:", predicted_class);.

So let’s see. And it should give us setosa. So that is correct because our model has very good accuracy. This prediction also looks like it is part of the training data set, which is correct. So our predicted class given this kind of data that is 5.1, 3.5, 1.4 and 0.2 is giving us that this particular flower, this particular given this configuration of flower, it is giving us setosa species and that’s it.

In this class we have learned how to perform multiclass classification using our own custom neural networks using PyTorch. Thanks for watching.

## **N) Understanding Components of Custom Data Loader in Pytorch**

In this lecture we are going to learn about Dataloader in PyTorch.

Let's get started.

So before moving forward and understand the components of custom Dataloader in PyTorch, we need to understand the problem with our current way of loading data. This is recap this we have already performed while learning about creating linear regression model using PyTorch.

So we have used this dataset from Kaggle that is insurance data set. And we have used pip install Kaggle which is by default library available in our Google Colab. As you can see, this is already satisfied. What we have done earlier is that this is just a recap, and we will understand the problem with our current way of loading data. Then I will talk about the components of data loader and what exactly data loader is.

This is Kaggle hub. We are going to download this data set and it will be downloaded to the path. It will also show the path. And in this path we will check whether we have any file downloaded or not. We are going to import OS and pandas dot pd. Then we are going to paste this path and mention that in OS dot list directory, and it will mention that what files we have in this particular directory and it is showing insurance dot csv. So we are going to load insurance dot csv. Then df dot head to know the information about data frame. It will show the first five rows of our data frame.

Then what we are trying to do is we are trying to load torch torch dot n to create our own custom model optimizer Labelencoder Standardscaler and train test split to split our data set. So here what we're trying to do, we are trying to mention our data frame which has all the information. So if we check this data frame and let's try to print information of the data frame as well df.info. And we can see in total we have 1338 entries. Right.

So what we are trying to do next is we are trying to split our data set into train DF and Testdf then we are performing something like Labelencoder on the columns which are object type right. Text is object type, Smoker is object type and region is also object type. And we are performing labelencoder on those object columns. And then to create Xtrain Ytrain x test and y test from the trained df, we are dropping only the charges column and keep all the columns exactly as it is in X train in the train DF for Y train taking the train DF and only considering the column charges. This will make us Xtrain and Ytrain. Similarly we will do for x test and Y test.

Let me run it. This cell we already ran. Let me run this as well. So now if we check and let me print x train dot shape. So this is in train. We have 1070 rows and six features. Similarly I can do for. Let me mention print statement because in Colab without print it will only show the last code that is x train dot shape. And I'm going to print for White Rain. So it is showing us again 1070. But because the feature length is only one it is showing us that. Okay, now if I mention x test because y test will be same but without that six. So we are able to see 1070 rows is present in x train and 268 rows is present in our X test.

What we are trying to do next is we are going to initialize our standard scalar and do the fit transform on X train and transform on the x test. So once this is done we are going to convert our Xtrain Ytrain x test, y test to tensors so that we can use into our model. And then this is our model which we have already learned before. And then our input dimension is going to be six because we have total number of six features. So we are going to pass the shape of one. That is this is zero and this is one. So six will be the number of features which we are going to pass into our model as an input dimension and create our model. Then we are going to create Criterion Optimizer.

And then the real thing comes the main problem with this approach of loading data. You are able to see that we are going to initialize. We are going to first set our model to train mode. This is useful when you have batch normalization and other layers. But for now we are just going to follow this practice that is model train. So we are going to set our model to train. And in our model we are passing x train tensor. Let me try where we have initialized x train tensor. Here I am going to print the shape of x train tensor dot shape. It is going to be 1070 comma six which is same. But now it is in a tensor format. So in a single iteration we are passing all of 1070 rows in a single iteration.

Let me define it here. Not here. Selective mentioning docstring. So suppose we have hundred epochs, and in a single epoch we are going to pass all of the data that is 1070 rows. Okay. Now suppose we have. This is just an example okay. In Xtrain we have 1070 and x train tensor. E n s o r. We have 1070 rows here. So now suppose we have we are working with another data set which has 1 million rows 123456 okay. So we have total of 1 million rows in our data set. Okay. I'm just going to like mention with this like separate with the help of commas. So suppose now we have 1 million rows in our x train tensor. So in a single iteration, using this method of loading data, we need to send all of this information into our model at a time.

And what is the issue? Issue is memory. Now suppose this 1 million rows is taking up two GB of data. Or suppose ten GB of data in your system. It might be possible that you don't have this much of Ram in your GPU or in Vram. That is your local Ram. You don't have this much enough space. So what you will be facing, you will be facing out of memory issue, which we call as Om. Okay. You will see this error which is out of memory okay.

So this is the first issue which we will get. And now one more thing is suppose you are working with a lower end CPU. Your CPU is slow. You have enough memory to take all of this data in a single go. So how learning will happen in single epoch? You will get all of this data that is total number of 1 million rows and your model will learn. Your model will do backward propagation and gradient descent optimization. Only when you have iterated over all of this 1 million rows, that is a single epoch. Only then you will get an updated updation on weights and bias. Okay. Only when you are done with iterating over 1 million rows of data.

So that is an issue, right? It's like suppose this is human okay. Suppose we are teaching human. We are teaching human and we have a book and we have a book of, suppose 1000 pages and we are teaching it to a student. And this student is learning all the 1000 pages. And only after learning 1000 pages, student is saying that I am not able to understand. And this is an issue that it might be possible that students have faced issue on the fifth page, not on the thousand page, but on the fifth page. But student can only say that I'm not able to understand because mention a scenario that only when the thousand pages are completed, then you can say whether you have understood or not.

So this is an issue. We want a feedback. We want a faster mechanism. We want a faster updation. So what you want suppose we can ask student that whenever you are completed with ten pages of this book, you can mention whether you are able to understand or not. And this is feedback okay. This will give us a faster feedback. So we are able to update our weights and biases faster now. And but even though we have ten pages, our one epoch will be completed. Okay. So it means that suppose we have 1000 pages. We are saying we will complete your feedback. We will take your feedback in ten pages. So how many pages are left. Total number of hundred pages. So ten pages one feedback. Total number of feedback will be hundred, so 100 feedback. Our one epoch will be completed. Because handed feedback means here hundred feedback means.

Hundred feedback is equals to 1000 pages of book, and thousand pages equals to one epoch. But we have divided the feedback. So that's why our original data set is thousand pages. We have mentioned that we want to keep we want to give the feedback that is weights updation okay. Using back propagation and gradient descent on 1010 pages. That's why in single epoch. Now suppose we have total number of hundred epochs. We have set hundred epochs for the model we have said that we want to give. We want to take the feedback on each ten pages. So how many iterations we will have in total, how many iterations we will have? If you have ten pages. 1000 pages total and epochs and ten pages feedback how many iterations we will have.

So to complete one book that is of 1000 pages, ten pages feedback we need to we will have hundred iterations. That is for one book. And in each iteration we have single epoch will have 100 iteration. So hundred iteration will have 100, epoch will have hundred iteration. So now we will have hundred thousand iterations in hundred epoch okay. So now the advantage is we are able to update our weights and biases more frequently as well as we will not run out of memory because we are going to load a chunk of data instead of passing all the data. That is total number of 1 million data, 2 million data, or it can be 10 million data into our memory, into our model we will be passing. We can pass chunks of data using data loader.

And this is what data loader is about. We are going to create our own custom data loader. We are going to use this same data. Although this data is not much it is only 1338 entries. But we are going to see the demo of how to create our own custom data set and pass data in batches to get things like memory utilization, as well as faster updation on our weights and biases.

So let's understand the components of data loader in PyTorch. So let me write here I'm going to mention in text. The first component is if I mention the first component the first component is data set. That is data set. And we use this data set from torch dot utils dot data dot data set okay. So a data set class is responsible for loading and processing data. That is if you are if you have a data frame, if we have a CSV file, we can mention things like how to load that is, read CSV transform, do standard scalar, do one hot encoding and transform into tensor. All these three processing we can do in data set. This is what we do in data set. Okay so it mentions how data is stored and accessed.

Second thing is how data is transformed. That is preprocessing. Third is how to retrieve a single sample. All those things we need to mention in data set. We will create a custom class. Second thing is data loader. Let me write data loader. So we import data loader from torch dot utils dot data dot data loader. So data loader is already available in PyTorch. What we create is we create our own custom data set. We tell our PyTorch that how we want to retrieve the data set, how we want to process the data set and data loaders help us by data loader helps wrap the data set. Okay data loader wraps the data set into an iterable object that loads data in batches.

Okay, so if I want 1000 rows in a single iteration, data loader will wrap the data set and ask data set to provide 1000 rows. Or if I mention only provide 16 rows in a single iteration, data loader will wrap data set and ask data set that I want 16 rows in a single iteration. Okay, so it handles batching and parallel processing using multiple workers, which I will talk about later. Okay.

So what we will do we will first create our own custom data set. Okay. We have to create the data set of how we want to like load the data set. Then we will wrap it with the help of data loader to ask our data in multiple batches. So let me write creating. Our Custom data set in PyTorch. So it also has multiple components components like in it. Okay. So this is asked in interviews a lot that how do you create a custom data set in PyTorch. And what are the multiple components of it. So keep this in mind that this is very important for interview point of view.

So this init initializes. The data set. It loads data and applies transformation if needed. That is pre-processing. It does pre-processing. Second thing is first thing is we need to mention init method. Second mandatory is length. We need to return the length from this method. We need to initialize this length method. We need to define this length method. And this length method does what this length method returns the total number of rows. Total number of samples. Or you can say rows in the data set. So whatever data set we are working with, this method only has to return the total number of samples. Why? Because PyTorch will internally read it. And to check from where our data set is starting. That is what is the initial index. What will be the final index? Because we are going to ask our data loader, which is going to be wrapping our data set class and ask data set.

How will our data set data loader will know that we are at the end of our data set using this length method. Okay, PyTorch will use this internally. Third mandatory function is get item okay get item. This is the third mandatory method which we have to implement in our data set. It defines how to retrieve. How to retrieve a single data sample. A single data sample when an index is provided. Think of this as like a list. We are creating a function on top of list, and when we pass that, we want index of five from that list. It will give me that index of five. Okay. That is the value of that index five. So we have to implement this method. Again data loader will use this data set class which have init length and get item. And it will use get item to get the actual data from that class, which it will use to load data, which it will use to load data in batches.

Okay, if I want 1000 rows, it will give me 1000. If I want single data, it will give me single data. But with the help of get item we will see how to implement it. So first thing what we are going to do is I'm going to copy this whole thing. Okay. We've already loaded our data. That is PD dot read CSV. We don't have to copy. So what I'm trying to do is we have a data frame object. We have data frame, we have df.info. We have train df and test df. I don't want to perform pre-processing because pre-processing is already performed. What I want is I want to load instead of loading all of this 1070 data at once, I want to load in batches. I want to load suppose ten rows or 20 rows at a time. So let's check where exactly our data has done the pre-processing.

So I think at xtrain and xtest this part is already pre-processed. And after that we are converting to tensors and then passing to our model. So before this conversion of tensors we can pass xtrain Ytrain x test and y test. We don't have to perform dataloader loading into batches of x test and y test because this is evaluation part. But for the model training, we don't want to pass all the data to our model while training it as it will be overhead. What we are trying to do is we will try to load Xtrain and Ytrain in batches.

So what I'm going to do is I'm going to create our data set loader. Let me scroll a bit. So here I'm going to define. Import. Torch which we've already done. But I'm going to mention it here again. "from torch.utils.data import Dataset, DataLoader" As I mentioned we need to upload. We need to like import few things like torch utils data data set torch utils data data loader. So from data we are add data. We are going to import data set as well as data loader. And let's run it. Hopefully we have ran all of this. If not let's run this part again so that we will not face any issue. Okay so this is our model. MSE loss and other things.

The model training code is not provided here because we are not doing model training. What we are trying to do is how to access data with the help of data loader. Okay, this is model training if you want to run it. I have set it to 1000 epochs and you can see each time it is taking all this data set. And on 1000 epochs we have this much of loss because of MSE okay. And then this is data loader. So what we're trying to do we are trying to access Xtrain and Ytrain which is before this tensor operation. And we will load this part into batches and pass it to our model.

Let me scroll a bit and then we are going to define our own custom data set. First thing is we have to mention in class and then the name of our data set. So I'm going to mention insurance data set. And then I will mention first thing is it needs to inherit data set class. Once that is done we need to define few things which I mentioned. That is the first thing is init method self. We are going to pass Xtrain and Ytrain. That is the argument. And then I'm going to mention self dot x is equals to x self dot y is equal to y. That's it. I have defined my init method.

So if not xtrain and ytrain and you want to start from reading the data that is this part. So our model our data loader will accept CSV path here okay. And it will do all the operation inside the init method that is doing the standardscaler and everything. Okay. But I feel like that might be redundant because we have Xtrain and Ytrain. And this approach is very simple. So instead of passing my data CSV, I'm passing x and Y, which is my Xtrain and ytrain, which I will pass it here because all the preprocessing has already been done.

Next method we need to implement is "def len(self): return len(self.x)". That also takes self as the first argument and we need to return length of self dot x. Okay this is length dot self of x.

In this part of the lecture, we continue building our PyTorch pipeline by defining a custom dataset class. The __getitem__ method is used by PyTorch to pass an index, and whatever the index is, it expects a single data point. So in the code, we define id and mention the same thing as before, that is, the type of this data as float32. I’m going to mention float32, and that’s it. This will give us features. Then I’m going to copy the same code and mention self.y. Also, in self.y, whenever we access a value, we mention values as well. Same thing: torch.float32 or we can mention torch.long as before. Okay, torch.long. I’m going to mention float32 because this is insurance data. The features — that is, the target variable — were in dollars. So this is going to be the target. And I’m going to just return features, return features, target. And that’s it. This is very straightforward.

So what we have done is defined a dataset, that is, a PyTorch custom dataset. We have inherited Dataset and defined three methods: __init__, __len__, and __getitem__. In __init__, we do all the preprocessing which we have not done before. Preprocessing like loading the dataframe and all we already did earlier. Once we initialize the insurance dataset, we are going to pass Xtrain and Ytrain here. So in __init__, we define the data. In __len__, we define how to get the length of this dataset. And in __getitem__, we define how to get a single row from our dataset.

The next step is to create a data loader and wrap our insurance dataset there. I’m going to mention insurance dataset, and in it, I will pass Xtrain (which is currently not in tensor form). This is Xtrain. We also have Xtest, and Ytrain and Ytest, which contain 1070 rows and 268 rows in the test set. I’m going to use Xtrain and Ytrain to create a dataset object. Once this is done, our dataset is created.

Now I’m going to use DataLoader, which wraps our dataset. In this, I will mention a few things like batch_size=32. Also, there are some other arguments like shuffle=True. You can mention it as false, it is totally fine. So what shuffle does is: out of the 1070 rows we have in our train dataset, do we want to access data sequentially, or randomly? Right now we have mentioned shuffle. You can also set it to false. What we are saying is: whenever we call the dataloader, give me only 32 data points in a single iteration. In a single iteration, inside a batch, give me 32 data.

Once our dataloader is created, we can also mention a few things like number of workers, but we’ll look at that later. For now, let’s load this dataloader. To do that, we write: for features, targets in dataloader. To also include the index, we use enumerate(dataloader). It is going to return us index, batch. I’ll store the index in batch_index. Then I’m going to print the current batch ID. This dataloader is iterable. Whenever we ask the dataloader, it gives us 32 data points.

So we print: print("Current batch:", batch_index+1). And just to show how it looks, I also print features and their shape: features.shape. Similarly, I print targets.shape. This way we check whether we are getting 32 data or not. Until this dataloader finishes all the data (using __getitem__ and __len__), it will keep yielding batches. I also add a break to only see the first batch. We can see in batch 1 that torch.Size([32,6]) is printed, meaning 32 rows and 6 columns, and for targets we get 32.

If I change batch_size to 320, then in a single batch we get 320 rows. If I set it to 1, then we only get one data point. For example, suppose we have 1000 rows in total, and epoch=100, with batch_size=1. Then in each batch we have 1 row, meaning 1000 iterations per epoch. Across 100 epochs, that’s 1000 × 100 = 100,000 iterations. Another example: with 1000 rows, epoch=100, and batch_size=100, then we have 10 iterations per epoch (1000/100=10). Across 100 epochs, that’s 1000 total iterations. So batch size determines the number of iterations per epoch.

If we don’t break early and just let it run, we can see all batches being printed until the dataset is completed. For example, if we set batch size to 1000, then in batch 1 we get 1000 rows, and in batch 2 we get 70 rows, since the dataset has 1070 in total. This shows the power of batch size — it controls the memory passed into our model. I’ll stick with 32, which creates 34 batches in total.

Now I copy the model training code we wrote before and adapt it to work with batches. Same exact thing, but now inside the batch loop. For each batch, after setting the model to training mode, we write: optimizer.zero_grad(), then calculate loss, backpropagate, and update gradients. Instead of passing the whole tensor, we pass batch_x and batch_y. So: predictions = model(batch_x). Then we calculate the loss between predictions and batch_y, compute gradients, and update.

To monitor progress, I print the current batch index using enumerate. I also print the epoch number and batch number. When we run it, we see a lot more weight updates, since we are updating after every batch of 32. This takes longer, but works correctly.

The key takeaway is: instead of loading all data at once, we created our own dataset class, wrapped it in DataLoader, and defined batch size and shuffling. This stabilizes training, reduces memory usage, and improves generalization due to shuffling. This is called mini-batch training, because instead of feeding the entire dataset, we train in small batches. Mini-batch training introduces randomness, prevents overfitting, and allows gradient updates to happen more frequently.

Finally, we can also use the argument num_workers. For example, if we set num_workers=4, PyTorch will create 4 parallel processes to pre-load the next batches while the model is still training on the current one. This reduces waiting time and speeds up training, but also uses more CPU or GPU resources.

By the end of training, after 34 batches per epoch and 1000 epochs, we get the final loss. The key advantages of using custom datasets with DataLoader are: (1) stabilized training with frequent gradient updates, (2) reduced memory usage since we don’t load all data at once, and (3) better generalization through shuffling. This completes our lecture on building a custom dataset and DataLoader in PyTorch.

## **O) Defining Custom Image Dataset Loader and Usage**

In Colab, I have created this dataset using Kaggle. This is the first dataset which I have used that is cat versus dot dataset. I have taken few of the images of cat versus dog dataset, and the second dataset, which I have merged in our cat versus dog is human image dataset. So here we have man and woman. So I have combined man and woman into a new category that is person. Okay. So in total we have three classes that is dog, cat and person. These are the two datasets which I have used.

This dataset is very big. You can see it is 734 MB and this one is also around 730. This is 864 MB. I have combined both, deleted few, cleaned the images and created the final version that is classification dataset underscore v3 dot zip. For the sake purpose of defining our own image dataset loader. So let's import a few libraries. And we are also going to unzip this because this is in zip format.

So first of all we will start by importing some torch libraries. We will mention import torch. I will close this tab once we unzip it. But until then we will write it in this screen.

import torch


We also need import OS because we are going to use those libraries.

import os


Then from Pil I'm going to import image.

from PIL import Image


Then from Torch Vision. Right now we are using Torch Vision. So Torch vision is a library from torch which contains things regarding vision. Okay so here I'm going to import transforms. I will talk about what is transforms.

from torchvision import transforms


And then from torch dot utils. From torch dot utils dot data. We are again going to import the same thing that is data loader and data set, data set and data loader.

from torch.utils.data import Dataset, DataLoader


And the spelling is wrong. This is torch. And let's import it. So now that we have all the libraries which we require, now it's time to unzip this zip. So the command I'm going to use is this is a Linux command. So I'm going to mention unzip. And then the path of this zip copy path. And that's it. Let's run it so we can see that it is inflating. It means it is extracting the files.

And this name folder, the main folder it is going to create is classification data set underscore v3. Inside it we have Images inside it. We have train and then we have person and then we have person dot jpeg. And this is the same structure I've mentioned in our notebook image dataset v3. If you expand it, you will be able to see images inside images you will see train and test.

I have also mentioned this rename file because I have renamed all the file. You can check it out later. And these are the information data set information. If you double click it, you will be able to see the link of the Kaggle which I have appended here. Okay, the link I have used these data set links I have mentioned in this resource.

So in this train and test we have train. Inside train we have cat, dog and person. Inside test we have cat, dog and person again. And if you click on cat double click you will be able to open image. This is image of cat again image of cat, image of cat. And if you want to check the image of dog you can click on the dog. And this is image of dog right.

Let me close all of these tabs. Now that our data set is available in our Google Colab, I'm going to close this tab and we will focus on the coding part. So now what we will do we will create our image data set data loader okay. First we will start with data set. Then we will use data loader to rap on that data set which we have created for things like batches right. And shuffle.

I'm going to start with class image data set. And the first thing is we have to inherit data set. And then we have to mention the first thing that is in it. There are three components of any data set. Custom data set that is init method. Second is length method and third is get item. First I'm going to mention in self and then I'm going to mention the image directory. Okay. So I'm going to pass train directory and test directory separately. So I'm going to create two kind of data set train and test. For that I'm going to pass image directory.

Second thing is I'm going to store that image directory in self dot image dir. Let me scroll a bit. Yes it looks good. Self dot image dir. I'm going to paste it. Then I'm going to create self dot image path. Image paths. I'm going to store all the image path here. And then all the labels in self dot labels. Because as I mentioned we have to create a structured format at the end image as well as their labels. This is what we have to create.

And also for the prediction when the model has like we have predicted, the model and this labels will be zero, one and two. We will also want to store the class names as well. So what I'm going to create I am going to create a dictionary. Self dot class underscore name and you will understand why I want to create as a dictionary.

So once this df init variables are done I'm going to create I'm going to load the data set. What I'm going to do for label for label comma class dir I'm going to use in enumerate. I'm going to use OS dot list dir. And I'm going to pass image dir. First of all if I pass my train directory path that is train directory path images inside images. If I pass train directory path and if I use OS dot list directory, it will give me cat, dog and person. Okay, so first of all OS dot list directory will give me cat, dog and person.

So in the first iteration, cat will come or dog will come based on the alphabetical order. So cat will come here and the label will be enumerate because it will be enumerate, it will give us zero starting from zero. So any class will be assigned to a label of zero based on the alphabetical order. So this is where we will get that if you want to see that as well I can copy.

And before running this I can create a code. And we will be able to see class directory print label comma class directory and image directory. I will also pass here and I will mention the path to train copy path. And that's it. Now if you run it you will be able to see zero as person one as cat, two as dog. We are here right zero and then person.

Now I want to store this person as zero. Cat as one. Dog as two. So we can mention because right now we are loading it, but we are not saving it. That which index belongs to which class? We are not saving it, we are just using it. Right? If I want to save it, what I will do, I will mention self dot class name the dictionary which we have just created and in the index I will mention label. And for that label I want to store class directory. If I do that the dictionary keys will be zero, one and two and the values will be person, cat and dog which I can utilize it later. Okay, I will talk about it where we are going to utilize that.

And once that is done, once we are in train test and where we are, we have to go inside dog, cat and person. Now what we are going to do, we will mention os.path.join. OS, dot path, dot join and I'm going to mention our image directory. This is our image directory and I will go inside single folder which is our class directory. We will go inside first person. The second iteration we will go inside cat. And the third iteration we will go inside dog. And we will store this path as class path class underscore path.

Now that we have a full path of cat, dog or person on each iteration, now we will go inside each folder and get the file names. So if we go inside person and get the file names, all of those files will be assigned to our ID of zero. So now we have person JPEG and its respective label as zero another person and its respective label as zero. So we will go inside each person folder all the JPEG and assign to a single ID of zero. So once person is done we will go and go to the next iteration. Then we will go to Cat cat folder. We will get all the images and the labels will be same. That is one for all of it. So that's how we are getting that structured format.

Okay, so now what we will do once we have this class path, we will mention for image underscore name we will get image name from OS dot list directory this directory of what path of class path. Then we will get all the images name and what we are going to do. We are going to mention self dot image path which we have just created. Here. We are going to append all the image path and the respective labels in self dot labels. Okay, zero index of image path and zero index of labels means they both belong to the same data. Okay, because we are appending it side by side I am going to mention self dot image path dot append.

But we cannot append image name because this is with respect to your class path. It is not the full path. To get the full path we will mention Os.path.join. Full path is in classpath and with image name. This will give you full path of your image. Okay. And then next thing is we are going to mention self dot labels dot append. This is labels and we are going to append. For all of them it will be having a single label that is zero. For each iteration label will be common images, names will change and the next iteration label name will change and image name will be same. That is for cat all the way.

Okay, so now that we are clear with init method, we are able to create a Destructure method. We have to pass few things like. We have to check the length which our PyTorch expects. We will mention df underscore length and then the first argument it takes is self and I'm going to return length of self dot image path. That is total number of images. I will mention self dot image path. This method is also completed.

Now the next method is get item. So we need to mention PyTorch. How to take exact single data. Given the index. So def underscore get item. This is get item. We don't want all of this. This is the boilerplate code. And instead of index I want to use index. I don't want to use this. I will mention that self dot image path from self dot image path. Get the index and store into image path. That is img underscore path. Do the same for label.

But instead of because we don't want to pass, our model does not expect a path. Our model expects numpy array. Okay, so we cannot pass model because this is. This data set is going to be used by data loader. And data loader is being used by our model. Right. In the batches we cannot pass image path. So we have to pass numpy array. What I'm going to use I'm going to use pillow image dot open. And keep this in mind that your model that is your PyTorch model expects image that is from Pil not from OpenCV.

It is compatible with Pil, but you can come. If you have your OpenCV image, you can convert to Pil very easily. I'm going to mention this image path and we have to convert this to RGB. Okay. Even though it will be originally RGB, but we are going to mention RGB for being safe and then it will return us image. Now we have image which PyTorch expects. That is below, right? And the next thing is we have passed the image. We also need label. What we will do we will mention self dot labels. And in the label we will mention ID because for same index which is trying to access, we have already passed image and we need label for the same exact image and we will get it in the same index because that is how we are storing here, right? And then we will store into label and we will return image and label whenever model is going to call get item internally. Right. So we are going to return. Image underscore label. So how we load the data set how we are going to extract the data set with the help of our custom data set is what is going to be different. But other parts of training and other things will remain same. Okay. But this one will use CNN, which we are going to see in upcoming classes. But for now we are going to see how to load, how to kind of load this unstructured data, this complex data sequentially. Okay. So I think everything is correct here.

And also one more thing that this image will be of different different sizes. If you check this person JPEG it might be a little higher in size then your person two. Person three. Person four. So this is taking a lot of time to load. If you load person three again, based on the internet speed, based on your bandwidth, it might take little time, but I'm going to close it. But it will have different different sizes and different resolution. So we have to kind of make it standardized. Right. But we cannot make a standardized during the model. So we have to make standardize while we are creating the data set creating the batches. So there is a method in PyTorch that is transforms. I'm going to mention transforms. Let me load transforms from Torchvision. We have already imported transforms. We are going to use transforms and this transforms applies multiple transformations on your input. You can mention all different kinds of transformation.

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


We are composing multiple kind of transformations on our image. This is transform. And I'm going to mention transformation like this is transforms okay. This is transforms as is there. The first transformation I want is to resize. I want my pillow image to be resized to suppose 128 cross 128. Lower resolution for faster speed. Just for example. Okay. If you want higher resolution, you can increase this part. Next thing is I want to convert. I want to transform. This transforms dot two tensor as mentioned currently this is in below image when the model will ask. Image when the model will ask for data, it will return us below image. But your model, your PyTorch model is not going to expect image. Below image it is going to expect your tensor. So we will convert it to tensor. And that's it. So we will store this transformation into transform. And where will I use this transform. Let me make it a little bit clean.

So now that we have our transform where will I use it. So I can use this transform on my image whenever my model is asking for data. Right. Whenever my data loader is asking for my data from the data set, then I can change this. So I need to change this here. What can I do here? So for this transform I can ask I can create a new parameter here. That transform is equals to none by default. And when the transform is none, okay, when the transform is none, we will not do anything. If transform is none or you can mention. Okay, let's store this transform in self. It will be accessible across self. Dot transform equals to transform. This will be stored.

Now if we check if self dot transform is available that is if self dot transform then we will mention self dot transform because we will pass this self transform. If it is not none, this will be object self dot transform and we will pass our image here and we will get a new image which will be first resized. Second sequence is it will be converted to tensor okay. So whatever you are going to transform here you can mention there can be lot of transformations like you want to create. You want to add additional noise. You want to change the color format. You can do it here. But right now we only have two transformations. First is resizing. Second is tensors applied to the image. Okay. So now this part is also done. Let's run it. Let's run this part as well.

Now the next thing is we are going to create our data loader. Okay. First thing is we have to mention our image data set. But our image data set expects what it expects. Two things. First is image directory. Second is transformation object. If passed we will pass both of them. But before that we will copy and paste our data set link. Let me check for train. Let me copy path. I will mention train image dir. Train image dir and I will paste it in here. The same thing I will copy paste it here. It will be test image dir. Because we only have two folders and I will store into test. That is train and test. So we usually use train to train. Our model and test is to test how model is performing. Okay. So these are two different sets of data. We don't mix it. Otherwise model will learn all of it. And we don't have any data to test on okay. So this will be separate training set. And test set is to check model performance on unseen data. All of them have the same folder structure but training is used for model training. Test is for evaluation.

train_dataset = ImageDataset(train_image_dir, transform=transform)
test_dataset = ImageDataset(test_image_dir, transform=transform)


Let me copy this whole argument so we will not miss the name and the image data set. I'm first going to pass train image dir. And then in the transform I'm going to pass this transform object which we have created. This will be our image data set. Same thing I'm going to do for test transform will remain same. And then I'm going to create this data set train image data set. So now that we have data set which is capable of asking data from our directory, let me copy this and to have additional functionality on top of it we are going to wrap with the help of data loader. Let me run it. Hopefully we will not see any error. No errors.

Now we are going to mention data loader. The data loader object. The first thing is going to be data set. The data set we have is this is data set. Spelling is wrong. The first data set we are going to pass is train image data set. Second thing we are going to mention is batch size. Batch size we are going to mention 32. And we will also mention shuffle equals to true. And we are going to store this data set in train loader.

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


Okay. Let's mention train loader. Screen image loader. This is train image loader. Same thing we will do for test image loader and we will test it whether our data set loader is working correctly or not. This is going to be test image data set batch size is correct. Data set loader is correct. And that's it. Next thing is we are just going to see whether we are getting images of size 32 in our train data set. Okay. So to do that because all of them all of the data set loader is iterable object.

for images, labels in train_loader:
    print(images.shape, labels.shape)
    break


What we can do we can mention for images comma labels. Each image will have 32 images in batches. So images comma labels in what in our train image loader. So what we are going to mention I'm just going to mention the shape of it to understand whether the images are coming correctly or not. Images dot shape, then labels dot shape. Let's print it. If we are able to see the first batch we are able to get is of size 32 and image size is three. Cross 128, cross 128. Because we have converted from Pil to torch and this is the dimension now. And if the dimension is coming, it means image was also coming correctly before. Right? And this was 32. It means 32 total number of labels. Again we are having lot of them. We will have lot of batches here because we have lot of images. But now it is able to load all of them, do the random shuffling, do the batches and get us data.

Now that this is done at the last one, you are able to see that we have only ten images because it might not fit all the 32 images because there is none. Image after ten. Okay, ten is the maximum limit on the final batch as well as ten. This looks like our data set is working correctly. To validate it, whether the images are correct or not, we are going to plot the images with the help of data loader only. So we will mention import matplotlib as pyplot as plt, and we are also going to import numpy. NumPy as np. And we will see that. What are the labels actually present. So labels are present in class name okay. Labels are present in class name and class name of what object object of data set. We have data set object in both train as well as test. I'm going to first load train okay train dot class name.

import matplotlib.pyplot as plt
import numpy as np

print(train_dataset.class_name)


So if you print train dot class name you will see what are the labels available. I'm going to mention print print class name or you can do the same for test. If you want to check whether both of them are same or not. You can see zero is zero, index is person one is cat, two is dog for both of train and test. So it means we can rely on this index and labels dictionary. What we are going to do. We are going to use the same code that is for images and labels or images. Comma labels in train image loader that is train image loader. We are looking for train image loader. And we will take only the index one of this image to just show how our plot looks like.

for images, labels in train_loader:
    print(images.shape, labels.shape)
    img = images[0].numpy()
    label = labels[0].item()
    print("Original shape:", img.shape)
    print("Label index:", label)
    print("Label name:", train_dataset.class_name[label])
    img = np.transpose(img, (1, 2, 0))
    print("Transposed shape:", img.shape)
    plt.imshow(img, vmin=0, vmax=1)  # torch ToTensor already scales to [0,1]
    plt.title(train_dataset.class_name[label])
    plt.show()
    break


I am going to mention print images dot shape just to check shape labels, dot shape just to check shape. And then I'm going to mention images of zero. So out of this 32 matches, I'm going to first select only the zero and convert to numpy. Because it is in tensor. Your matplotlib cannot accept tensor. So we have to convert to numpy and this will be our image. Next thing is we have to take labels. So I'm going to mention labels of zero. Labels of zero is going to get us labels. And if we check item we don't have to convert to numpy or anything, it will automatically a final scalar value which we can use as a label. It will be integer okay. So this will be our label.

Next thing is we are going to print the label what this label is about. So we have our label in train image data set dot class name. And we have our index in label. I'm going to copy this whole thing which contains this dictionary. And I want to access what I want to access the label, which will give us either person, cat or dog for this particular image. So by plotting it, we will be able to see what this image is and what the label says. If both of them are not matching, it means something is wrong with our data loader. And once this is done, we have to transform our image. Because if we check the format, it is three cross 128 cross 128, which is not correct. We have to shift our image channel. I'm going to mention NP dot transpose. I want to transpose my image and I want to change the index whatever was at zero I want to be one. Whatever was at one, I want to be two. Whatever was at two, I want to be zero. Okay, I'm changing it. It is IMG not image. I'm changing it. And then I'm going to print img. This is img. Now we can also print the shape of it. Print And img dot shape. And this one also let's print it to check what was previously and what is now img dot shape. And we will also print label. To validate each and everything.

And finally with the help of matplotlib plt dot I am show we are going to plot image and this image as converted from tensor to numpy might have different range. So we are going to mention that whenever there is the minimum is going to be zero, and the maximum is going to be 255. Okay. So that it will show the ranges correctly. And then we will break it because we don't want to plot for any other batches for the single batch. For the first batch we are taking the zeroth image. And we don't want this to be iterate and just show the image. Let's run it. We are able to see this person and it is person. The index is zero. The shape originally was three. Cross 128, cross 128 as seen in our data loader. But now because of transpose we are able to see width, height and channel. And this is what our matplotlib accepts, right.

So this looks correct. If we again load it because of shuffle equals to true we will see a different image. Let's see. This is again a person and this is also a person. So this is also correct. Let's do few more times. This is cat right. We are able to see cat. So it should also like mention cat here correct. This is also correct. And let's load for dog. And we are able to see dog here as well. So it means our custom data loader for our complex data structure that is unstructured data set images. It is able to load successfully. So from this now we are equipped with handling unstructured complex data set using PyTorch.

### **P) CNN Training using Custom Dataset**

In this lecture, we focus on designing and training a Convolutional Neural Network (CNN) using a custom dataset. Before moving into architecture design and training, we briefly revisit the data preparation process. A custom dataset is structured in a folder-based hierarchy where each subfolder represents a class containing respective images. Using PyTorch’s Dataset and DataLoader, we ensure images are correctly loaded, transformed, batched, and shuffled for efficient training. Additionally, standard preprocessing steps such as resizing, normalization, and augmentation are applied through torchvision.transforms to make the dataset ready for the CNN.

Next, we recap the role of the DataLoader. It allows us to iterate through the dataset in mini-batches, apply shuffling for randomness, and use multiple workers for parallel loading. This setup is essential for handling large datasets efficiently and ensuring smooth integration with the training pipeline.

We then move on to the CNN design process. A CNN typically consists of a sequence of convolutional layers, each followed by Batch Normalization, a ReLU activation function, and MaxPooling. The convolutional layers extract hierarchical spatial features from the images, batch normalization helps stabilize training, ReLU introduces non-linearity, and max pooling reduces spatial dimensions while retaining key information. After several convolutional blocks, the output is flattened into a vector of neurons, which is then passed through fully connected layers to learn high-level representations. The final layer contains neurons equal to the number of output classes, producing the classification predictions.

One important consideration in CNN design is determining the correct number of neurons when flattening convolutional outputs. Since image dimensions reduce after convolution and pooling operations, we must dynamically calculate the number of features. A practical approach is to pass a dummy input through the convolutional blocks during initialization to automatically determine the flattened dimension before defining the fully connected layers.

Finally, we implement the architecture in PyTorch by creating a custom class that inherits from nn.Module. Inside this class, we define the convolutional layers and fully connected layers, arrange them sequentially, and write the forward pass logic. With this setup, the model can handle varying input sizes while ensuring the final feature map is correctly connected to the output layer.

This systematic process—from dataset preparation, DataLoader recap, and CNN block design to final architecture implementation—ensures a robust foundation for building and training convolutional neural networks on custom image datasets.

Now, we extend our understanding of CNNs by not only building the architecture but also training, evaluating, saving, and deploying it for real-world inference.

(i). Model Initialization and Architecture

We begin by defining a Custom CNN class that inherits from nn.Module. In the __init__ method:

We initialize convolutional blocks using nn.Sequential, where each block consists of Conv2D → BatchNorm → ReLU → MaxPool.

The number of filters increases gradually: 3 → 32 → 64 → 128 → 256.

After the convolutional layers, the output must be flattened before feeding into fully connected (FC) layers.

Since we cannot know the exact flattened feature size beforehand, we dynamically calculate it by passing a dummy input through the conv layers. This initializes the first nn.Linear correctly.

The fully connected layers (FC layers) are then defined:

Flattened features → 512 neurons → 128 neurons → output layer.

The final output has as many neurons as the number of classes (e.g., 3 for cat, dog, person).

Non-linearities (ReLU) are used between layers, with optional dropout for regularization.

(ii). Forward Pass

The forward() method defines the data flow:

Input image → convolutional layers → feature maps.

Feature maps are reshaped with x.view(x.size(0), -1) into flat vectors.

Flattened features → FC layers → final predictions (logits).

(iii). Training Setup

We check device availability (cuda or cpu) and move both model and data to the correct device. Training requires:

Loss function: nn.CrossEntropyLoss() (automatically applies softmax).

Optimizer: torch.optim.Adam(model.parameters(), lr=0.001).

(iv). Training Loop

For each epoch:

Set model to train() mode (ensures batchnorm/dropout behave correctly).

Iterate through batches from the DataLoader.

Move images and labels to device.

Clear gradients with optimizer.zero_grad().

Forward pass → compute loss → backward pass → optimizer step.

Track running loss across batches for monitoring.

(v). Evaluation

After training, we switch to eval() mode and disable gradients with torch.no_grad().

Predictions are obtained with torch.max(outputs, 1).

Accuracy is computed as (correct / total) * 100.

Even after just 2 epochs, the model showed ~74% accuracy, improving further with longer training (e.g., 40 epochs).

(vi). Saving and Loading Models

Trained weights and biases are saved with:

torch.save(model.state_dict(), "cnn_model.pth")


To reuse the model later, we must:

Redefine the same CNN architecture class.

Load weights using:

model.load_state_dict(torch.load("cnn_model.pth", map_location=device))

(vii). Inference Class for Real-World Usage

To simplify prediction, we design an ImageClassifier class:

Loads the trained model and class mapping.

Applies the same transformations (resize → tensor → normalize).

Accepts an image path, preprocesses it, and predicts the class.

Uses unsqueeze(0) to simulate batch size of 1.

Maps predicted index (0/1/2) to actual class labels (dog/cat/person).

Optionally overlays the prediction on the image using OpenCV (cv2.putText) and saves the result.

(viii). End-to-End Workflow

Prepare dataset → Dataloader.

Define CNN → conv + FC layers.

Train with optimizer + loss.

Evaluate on test data.

Save trained model.

Deploy with an inference class for single-image predictions.

This end-to-end pipeline equips us to not only train CNNs from scratch but also make them practical for real-world applications where we can classify unseen images directly.

## **Q) Understanding Components of an Application**

Hey everyone and welcome to this exciting project on building an image classification app.

So far we have explored deep learning with PyTorch, built a custom CNN model, and created a custom data loader to handle our own data set. We've also trained our model for X epochs and were able to classify images of cats, people, as well as dogs.

But what's next? The next step is to make this model accessible to everyone. But how can we do that? Instead of running Python functions manually or scripts manually, we will convert our trained model into an interactive web application that allows users to upload an image and receive real time predictions. And we will build this image classification application step by step across multiple lectures, understanding each component in depth.

In this lecture we will understand components of an application. Let's get started. So I will start by writing core components of any ML application. Let me write core components. Of an ML application. ML does not specifically mean linear regression and logistic. It means we will be covering deep learning application as well. Okay, that is what ML application means.

So there are multiple components of any ML application. First is front end. So we have not touched any front end yet. All we are creating was backend or the logic of our model, the logic of our program. So first of all I will create a box of it because they are going to be a lot of components.

This is the first component that is frontend. The front end is responsible for the visible part of your application okay. So whatever you are able to see on the screen, when we create or when we use any application or even a website. So whatever you are able to see on your browser that is called front end, the visible part of your application.

So let me write visible part of application. This is called front end. In this application we are going to create front end to allow users to upload image and see the prediction. And all of this will happen on the front end.

Okay. So here we will mention we will allow users to upload image and see prediction. So here in our front end we have two components which is: we need a method, we need a way so that users can upload image and once the image has been uploaded and the prediction has been made, we have to see the prediction. Whatever the predictions are, we need to again send it back to our front end so that we can see it.

So upload and see are the two components of our front end.

Next component of our ML application is back end. The back end is responsible for logic and processing. So let me write it here. Back end. This is back end. The back end is responsible for logic and processing.

The processing includes two parts: pre-processing and post-processing. So pre-processing is like doing the transformation on the image, getting the image ready so that it can be inputted to the model. And post-processing is when we get labels as output, we get index as output, and we map it back to labels, map it to image, and write the right text into image — all of this comes under post-processing. Okay. So this is what we cover in backend.

So in our ML application we are going to use backend to handle the model, to handle the model loading. We will load our model in backend. We will do the pre-processing as well as handle the inference and all of it we will do in backend.

So our backend can also decide whether we want to run on CPU or GPU based on where we are keeping our application, and we are going to use PyTorch for deep learning inference. This is all going to be present in backend.

Next thing is model. So model is our core component of ML application. This is the third core component of ML application. So let me write model. We are not numbering it but this is model. And we have already created model which can predict dog cat and person. So we are going to use that model. This is a deep learning model.

In our scenario a deep learning model which is trained to classify cat, dog, as well as person. In our case, this is a custom trained CNN model which we have trained using PyTorch.

And then the final component which we are going to have is deployment. The deployment is not the component, but the strategy of where you want your application to run. And this part we handle in deployment. So deployment is a platform — in this part we mention the platform in which we want to deploy our application.

The deployment is where we decide the platform where the application is hosted and made available to users.

There are multiple options for deployment which we can pick. First one is Hugging Face Spaces. So this is free to use, and although free one has some limitations. You can pick the tier one, tier two, and other like paid services which will give you more flexibility.

Second is AWS. You can choose AWS services to deploy your application. You can use Google Cloud Services (GCP). You can use Azure services as well. Also you can use Streamlit Cloud. So Streamlit Cloud also provides free access. This one also has free tier which we are going to use for deployment.

So the one which we are going to use for this ML application deployment, we are going to use Hugging Face Spaces. Okay, Hugging Face Spaces. This is a single term Hugging Face Spaces.

Now let me define how all of this will look like. So in our application we are going to have front end as I mentioned. Now we know all the components which we require. So this is going to be our front end.

And now let's suppose this is back end. In the back end we will have model. We will have inference code which will load the image. This will be pre-processing and this will be model inference. Then this will be post-processing.

So let me mark with a different color. This yellow one will be responsible for whole inference. In the inference we will do first pre-processing, then model inference, then post-processing.

Now let me pick this red and this is our model. I'm going to pick white. And let me write. This is front end. This is our back end. The back end has multiple components. This will be our model. I'm going to write this as pre-processing. This will be our inference, inference to the model and then post processing.

So inference will happen with the help of model. So we are going to have an arrow here. When we call for inference, we will pass our tensor to the model. This will be tensor to the model, and model will return the output back here in the inference part. We are going to have model.

This is data going to the model and model processing the data back which will give us tensor as output in the post processing. What we are going to do: we are going to use torch.argmax or torch.max to get the most probable answer out of those neurons and process it, map to the label. Right.

So this blue part is inference. And this takes part in the pre-processing. It takes input as image. And after this post processing from this whole inference block, I will get something out. This is input. This is output. My input will be an image and output will be either dog or cat or person. This will be my output.

So this is my whole back end. And you can see there is no element of UI present in my back end.

What about front end? In the front end we will be having a drop box — not a drop down, but a section where we can drop images or we can upload images as well. Here we will mention upload image. We will load the image from here. We will load the image from here. This is load image. Or we can say upload image. We will upload the image from here.

And then we will pass the uploaded image to our processing. I will mention it with a different color. So what we are going to do, we are going to use this uploaded image and send it to our back end. So you can see change in color. This is the output from front end which is going to our back end.

And our back end is processing dog, cat, and person which then we will use back in our front end. Okay. And then we need to show this. So what we are going to create — we are going to create a new text box. We are going to create text box as well as we are going to show the image with the text written on it.

Okay. Suppose this is an image of a person. This will be an image of a person which we have inputted. But after this prediction we will also write "person" on the image. And also in the text box we will return "person". And so this is our front end connected to the back end.

So front end is responsible for loading the image where users are allowed to upload an image. And then once we submit, once we click on a button submit, it will send it to backend.

In backend we have multiple processes. In backend we have separate entity of model which we are going to load beforehand and use it in inference. We are going to load image, convert to tensor, do some transformation like resize to like whatever model expects the input dimension to be. We are going to do some pre-processing, and then we are going to pass that pre-processed tensor to inference.

Inference will send the tensor to model, which will then predict multiple classes with the help of torch.argmax or similar. We are going to collect the most probable answer from that particular tensor. We are going to post-process it. We are going to convert index to their labels and then send the output.

Then the output needs to be cached again in the front end. So front end again has two parts which I've already mentioned before. And this is the front end. Users are allowed to upload image and see prediction. Upload is a first part in front end. See prediction is also another part of front end. Both of them will not happen because only after prediction we will see this image. Only after prediction we will see this tag of "person".

So once this process is completed we will upload. We will send this data — that is the "person" text and also the image where we have written "person" on our input image — we are going to send back to our front end.

So this is our whole running application.

Now the issue is where do we want this application to be hosted. We can run this whole application in our local system. But the disadvantage is no one will be able to use it apart from us because it is running in our local system.

So we are going to deploy this application where other people can also use it. So our whole front end and back end, which is an application we are going to wrap with any deployment strategy. Let me select this blue color. So we are going to wrap this whole part which is our whole package — back end as well as front end. We are going to make a package from it. Not a Python package but just the code package.

And then all of this will be present in a single environment. And we are going to use Hugging Face Spaces. Okay. Hugging Face Spaces. This is where we are going to deploy our application.

So just to recap, our core components of an ML applications are front end, back end, model and deployment.

And that's all for this lecture. I'll see you in the next one.

### **R) What is Deployment**

In this lecture we will understand what is deployment?

Let's get started.

So deployment means making an application accessible online so users can interact with it without running your code locally. Let me write it here:

Deployment means making an application accessible via online. We want our application to be accessible online so users can interact with it without running locally.

So what usually happens is that we have a GitHub repo. Suppose we have a GitHub repo where we have a lot of code. Suppose Python code. We have a YAML file configuration file. We have modules. We have multiple modules of our Python. And we have all under GitHub repo okay into a single folder. I will write it as repository or just repo.

And usually suppose this particular repository is creating UI plus your back end — that is your front end plus back end. But the code is written inside GitHub repository. Okay.

So to replicate this, to see how the code is working, first of all any user has to clone it or fork it. Any user has to clone it or fork it, install the requirements, and then run it as well. And then run the application. And once the application is running now it is ready to use.

So whoever wants to use this code, whoever wants to run this application, they have to perform all of this operation: that is clone/fork, then install the requirements, run the application and then the application is now running. But again this application which is running is present locally. So already there is a lot of hassle to run this application and then it will only run locally.

So what you can do — this is user A. So if user A has an instance of your application A1, it is local to that user. And there are a lot of users available worldwide and they want to access your application. They want to run. They want to see how your application is working. They want to access this application. But it will not be possible.

But it will not be possible because the application you are running — suppose you are in a container. This is container and you are running your application locally and the other users are not able to access it.

So what you can do instead of running your application where it is local and not connected to internet, what you will do, you will run this application. This is you. You as a user will run this application in suppose AWS or GCP or Azure or something like Hugging Face Spaces or as well as Streamlit Cloud.

Okay. Now instead of running and setting your application, you are setting your application one time in cloud. Now it will give you a URL. After deploying on any of this cloud it will give you a URL.

For example:

https://myapp/run


This is the URL you got.

Now the advantage is once you have this URL, you can share this URL to anywhere in the world to anyone, and then they will be able to access your application because now it is not running locally, it is somewhere on the cloud, and cloud is connected to your internet, and everyone who has access to internet can access your application.

There is no need for all of these users. Now let me mention this is user one, user two, user three, user four, and user n. So all of these users — now there is no need for any of these users to install dependencies, to set up their system or anything. They just need access to browser and this specific link so that their browser can open this application. And with the help of your back end plus front end they can see a full application running okay.

They can upload the image. They can see the predictions.

The steps involved in deployments are:

(i) Develop the model. First of all, we need our AI engine which can predict based on the tensors. It can predict the output.

(ii) Create an interface. And then once the interface is created, an interface is connected to your pre-processing as well as post-processing which we call as back end.

(iii) Host the app online. We are going to have all of this one and two and we will host it online.

And we will see how to host our application online in coming lectures. So we will use Hugging Face Spaces to deploy our application.

**S) Tools to Create Interactive Demos**

In this lecture we will see tools to create interactive demos.

Let's get started.

So I will categorize these tools which will help us create interactive demos into two categories. Let me define interactive tools.

The first category is those which provide UI and also handle back end automatically. So these tools have the ability to create UI automatically as well as attach it to backend. And another kind is which only provides API flexibility. So it will have all the controls over API, but it does not provide any UI. Okay. And when it is an API, we can also control backend, but we don't have UI here. We have to create our own UI.

In terms of this section of interactive tools, the tools that come under UI and backend are Plotly (from Dash), Streamlit, and Gradio. And in the API-only part we have things like Flask and FastAPI. So Flask and FastAPI will give us flexibility to have all the controls over our API. We can create an API to access database, we can create an API to call our model, we can create an API to call a simple function — even a simple “hello world” function as well. So all of the flexibility we have here, but this does not provide a front end.

But if I talk about tools present on the left side (interactive tools), we have Plotly, Dash, Streamlit and Gradio. They allow quick UI creation. It auto-generates interface from simple Python functions. Okay, so what this section allows is quick UI creation. Once we have that UI, we can connect to our back end which can exist in that same file. We don’t have to create a separate back end, though we can do so for more standardized coding. Also, all of this auto-generates interfaces but not from any other language — only from Python functions.

If I talk about the tools on the right side, which are Flask and FastAPI, we can create an API. This is how the diagram will look like, the flow will look like. So we have to create front end. Suppose this is our front end, and we are going to create front end here with the help of HTML, CSS, and if you want JavaScript you can have JavaScript, or we can use more advanced things like React. Then all of them will be connected via our Python backend, which will have two things: one will be our core code of the backend, and another part will be API which will access those core backends.

So if I call an API from the front end, we will be creating it with the help of Flask and FastAPI. For example, API1, API2, API3, API4. From the front end, suppose this is the first text box and we have a submit button. When I type “Hello” in the text box and click on submit, I will mention in my API that it should be active whenever this submit button is clicked. So API1 will work when this particular button is clicked.

Now suppose I have another box for email and password, and I want to do login. This button performs login check. Once I click on login using email and password, we will go to API3 (just as an example). API3 internally is using a function, suppose function A. What it is doing is accepting email and password and checking whether this email and password is correct or not.

When we click on the hello text and submit, API1 is called. Suppose API1 is calling an NLP model which will take this text and try to predict the next word. In this API we will write which function it will call. Suppose it calls function B. All of these functions are Python code. So our API is the endpoint which we are going to use in our front end (HTML, CSS, JavaScript), which is non-Python. Meanwhile, the backend functions are Python.

So now you can see why this right-hand approach is more flexible. But the left side (Plotly, Streamlit, Gradio) ignores all the hassle of creating front end, because it auto-generates the interface from Python functions. On the right, we have to create the front end, write Python functions, and connect which button calls which API. It is more complex but gives more control.

If I were to create a small prototype or demo of a product where I want to see how my model is working and I want other people to use my simple application, I will use the left side (Plotly, Streamlit, Gradio). But if I’m creating a website (e.g., Amazon) with multiple products, databases, and large-scale features, I need more flexibility. In that case I will use HTML, CSS, JavaScript, React for front end, and Python backend (Flask/FastAPI) for APIs. For example, the backend could use a recommendation system in Python that tells Amazon which products to recommend.

So the right side is for more complex scenarios, but for fast prototyping we use Plotly, Streamlit, or Gradio. In this project we are going to use Gradio. The advantages of Gradio are: it allows quick UI creation, auto-generates interfaces, is compatible with text, images, audio, and video, and works seamlessly with Hugging Face Spaces. So we are going to create our application with Gradio and deploy it on Hugging Face Spaces.

Now, let us talk quickly about Plotly Dash. This is the home page of Plotly Dash. In the documentation you can see quickstart, dash fundamentals, callbacks, component libraries, and deployment strategy. Plotly is mostly used for data-intensive apps — if you want to see how your data looks, plot distributions, or visualize graphs, it is very good. But for model inferencing, Streamlit or Gradio are better.

Here’s a minimal Dash example. First install Dash. Then from Dash import dash, html, dcc, callback, etc. Import plotly.express as px and pandas. Read a CSV file into a DataFrame. Create a Dash app object, define layout, add header, dropdown, and graph. Use a callback function to update the graph. Run it with:

python app.py


This will start a local server using Flask (Dash uses Flask internally) and serve your app on a local URL. If you want to deploy on the cloud, Plotly provides documentation for production deployments.

Next is Streamlit. On the Streamlit homepage, they show how simply writing st.write produces UI output. You can create sliders with st.slider, upload files with st.file_uploader, create radio buttons with st.radio, charts with st.line_chart, and more. Streamlit has many built-in components like charts, input widgets, media elements, layout, containers, and even elements for chatbots. This makes it powerful because you can focus on modeling instead of UI.

For deployment, Streamlit Cloud makes it very easy. You connect your GitHub repository or sign in with email. Whenever you push changes to GitHub, your Streamlit Cloud app updates automatically.

Now, Gradio. Just like Streamlit, Gradio allows sliders, chat components, file uploads, audio, video, even 3D models. It is simple and integrates very well with Hugging Face Spaces. Many multimodal demos use Gradio. The documentation shows how to create quick apps, and deployment is easy. That’s why we will use Gradio in our project.

Finally, Flask and FastAPI. These don’t provide UI, only APIs. Flask and FastAPI are widely used in companies. If you need custom APIs and want full control, use these. But for quick demos or prototypes, go with Gradio or Streamlit. If you want data-intensive dashboards, use Plotly Dash.

One note: in my experience, Streamlit had some compatibility issues with Torch (temporary, will be fixed soon). Gradio didn’t have such issues. So we will stick with Gradio for building and Hugging Face Spaces for deployment.

This is all the information we need to create interactive applications.

**T) Hosting Platform:**

In this lecture we will learn about hosting platforms, what they are and which hosting platform is suitable for which kind of ML application.

Let's get started.

Let me start by defining the main use case of hosting platforms in our scenario.

Hosting platform – the main use case for hosting platforms is to make our ML model accessible online.
It is used to make our machine learning models accessible online.

So I will talk about popular ML model hosting options. Let me write:

(i) Platform

(ii) Features

The first platform I will talk about is Streamlit.

If I talk about Streamlit, it is free to use and good for data apps. This is where we can also upload. So this is Streamlit Cloud. We can use applications based on Streamlit to be deployed on Streamlit Cloud. But only applications based on Streamlit.

Next, if I talk about Hugging Face Spaces – this is also free to use, and it is very easy for both applications made from Gradio as well as Streamlit. You can use both here.

Let me draw this table as well. This is our platform. Now it looks much better.

The next one I am going to define is Google Cloud Run.

Google Cloud Run has serverless capabilities and it is scalable. I will talk about serverless and scalable. But the same thing is also available in Amazon Web Services (AWS). They have multiple services for deployment:

(i) SageMaker

(ii) AWS Lambda

(iii) AWS EC2 instance

And I will talk about how we do the deployment and which one to pick based on our ML application.

If I talk about AWS, it is very high performance because it has different components like SageMaker, Lambda as well as EC2. It is high performance but costly. This will also be costly because it is from our cloud provider.

But if I talk about Streamlit Cloud and Hugging Face Spaces, they are free to use. And in between Hugging Face Spaces and Streamlit Cloud, I can see a clear advantage because Hugging Face Spaces is not just dependent on Streamlit – we can have Gradio as well as Streamlit. But in Streamlit Cloud we only have Streamlit applications.

So let me define some color to make our table complete. And that's it.

Now let's talk about which kind of platform can handle which kind of ML application.

I will start with Streamlit / Gradio because both of them are equal here. In terms of Streamlit, we can only mention Streamlit applications. But all of them have a single category – that is, they both are used for Gradio / Streamlit kind of applications, i.e. applications that create UI by their own.

So what kind of applications?

If I talk about Streamlit / Gradio – they are used for machine learning demo apps. We don’t use them in production. We can use them, but only if we don’t have a lot of users. If we have very few users, because it is used for demo and we want a quick deployment.

We just want to check and want our code to be accessible by other people. So this is what we use for ML applications as well as quick deployment.

Okay. We want something which is of low maintenance. And we don’t want to manage our infrastructure. So we don’t want to worry about how much RAM we have, how much GPU we have, Docker, scaling our application with the help of Kubernetes. We don’t want to manage all of those applications. We don’t want to manage the infrastructure which is required for that big application.

So the next thing is – we don’t want to work with cloud platforms. If we don’t want cloud platforms, we will use Streamlit / Gradio in those scenarios.

Now if I talk about AWS SageMaker.

We use AWS SageMaker when we have to do end-to-end ML pipeline support.

End-to-end pipeline support means we need model training support. It means we need to have access to GPU, as well as a place where we can infer from our training code.

So for large scale applications, for production-ready applications, we use AWS SageMaker. This is for demo, not for production. But applications which are ready for production, we use AWS SageMaker.

It has infrastructure like training and pipeline. If you want to create something like an advanced CNN architecture (like ResNet for medical imaging classification), you can use that. If you want to train, fine-tune, and deploy your model with AWS SageMaker, you can do so here.

SageMaker also provides auto-scaling. Auto-scaling means that you are running on a particular box. Suppose this is a box and currently it is using 12 GB of RAM. But later your model wants 14 GB. It can automatically scale and create a bigger box which has 14 GB of RAM. Okay. For real-time inference. Not for training but for inference.

This is just an example to show what we mean by auto-scaling. So it also provides auto-scaling. And none of this is present in Streamlit / Gradio because the use case is different.

Now if I talk about the next one, which is AWS Lambda – which we call as serverless inference.

We use this if we want serverless. It means we want event-driven inference. We will process the image, pass it to the model, and get the output.

When the system is not running we don’t want any cost. That is what we mean by serverless.

So if you want to deploy a small PyTorch model (not a large one), and you don’t want the system to be running always – you can deploy it on AWS Lambda. For example, to classify something like dog or cat or person.

It is very powerful, but it will be called on the go only when we need it. This instance will be running in sleep mode (or energy-efficient mode). But whenever we call for inference, then this system will be active.

That is what serverless inference is. And whenever it is in sleep mode, you won’t get much cost. But whenever the system is invoked, cost will be there.

So the model only runs when a user uploads an image to this particular model, or sends to AWS Lambda.

Also, AWS Lambda automatically scales and shuts down after execution. The main point is:

(i) Automatically scales

(ii) Shuts down after execution

So if you have a specific requirement for this, then you will be using Lambda. If not, you can use AWS SageMaker as well.

Next is AWS EC2 instance.

You might be using this a lot if you don’t have a good system. If you don’t have good hardware, your company might provide you AWS EC2 instance. This is a virtual machine.

This will be an exact machine – suppose you have 12 GB of RAM, 4 cores in your CPU, GPU – just like your physical computer or laptop, but on the cloud.

You will be getting ID and password to login and the SSH URL. You login to this cloud using your terminal. And it will be a full Linux system.

So whatever you are able to do in your local system, you will be able to do in EC2 instance as well. It is basically just a powered version of your local system, but running on cloud.

So you will use AWS EC2 instance when you want to have full control over your ML training and inference.

You don’t want AWS Lambda. You don’t want it to automatically shut down. You don’t want it to autoscale automatically. You want to write each and every code on your own. Then you use AWS EC2 instance.

So you can write your own APIs here. You can do your own model training. Because it is basically your own system, but on the cloud. That is what AWS EC2 is.

And the final one we have is Google Cloud Run.

We are going to use Google Cloud when we are using a Dockerized ML model.

So we have an ML model, and we are using our ML model wrapped with Docker. We will create a Docker image.

If you are not familiar with Docker, it’s fine – we have other options. But if you know Docker image, it is just a wrapper of your ML model. Your Docker image will have Linux and Python installed. You create an image of it, which you can send to any system.

Once you have this Docker image, you can use Google Cloud Run. Pass this Docker image, and it will do operations like auto-scaling automatically. It will build REST endpoints (APIs). It will create APIs.

And if you don’t want to manage infrastructure – like scaling, servers, etc. – you just pass the Docker image and Google Cloud Run handles everything automatically.

But the requirement is you need to have containerized deployments. That is the requirement for Google Cloud Run.

This is also serverless. You don’t need to manage servers. You just need to create this part. Just like AWS Lambda – you don’t have to create servers, you just create AWS Lambda instance, and it will handle all configurations automatically.

And this covers all of the deployment strategies which we can pick based on the machine learning application which we have. And now we know which hosting platform to pick based on the kind of ML application.

**U) Setting Up Gradio App in Local Space**

In this lecture we will set up our app locally and run our first radio app. Let’s get started. So I’m using VSCode as my editor. And for Python I’m using Anaconda distributor.

First I will start by opening terminal. This is my Windows system. So this is my Windows terminal. And I will start by creating conda environment. "conda create -n radio_app_temp1 python=3.11". Then the name of the environment. The name of the environment I’m choosing it as radio underscore app underscore temp one. This is the environment which I’m creating. And I’m picking Python version 3.11. You can pick 3.12 as well as 3.10 based on your requirement. But currently I find that Python 3.11 is the most stable version. So I’m sticking with 3.11 and I will start creating this environment.

It will ask me for yes and no. That which of the packages which I want to install, I have to mention yes. And it will start the installation. And now you can see this is the environment has been created. So to activate this environment I have to copy this command and I can activate it. "conda activate radio_app_temp1". And now you can see this particular conda environment has been activated.

The first thing before creating any application, any application is that we have to install our radio. And we will install radio with some standard method that is with requirements dot txt. I will be creating a folder where we are working. This will be our project main root directory. I am going to create Gradio app one. Inside it I’m going to create requirements dot txt. Requirements dot txt and then I will mention "gradio" here and save this file and close this file.

So to install the requirement I have to first move to this Gradio app one folder. Now I have to mention "pip install -r requirements.txt" and this will install the Gradio requirement. It is looking for PyPI index and looking for Gradio. So you can see this is downloading Gradio 5.16.0. So we are able to see this FastAPI. This Gradio is installing FastAPI. And when we are checking the documentation of Plotly, Plotly was using Flask internally and Gradio is using FastAPI internally. And the installation is almost over.

While installing Gradio, a lot of the libraries are also installed. This is huge number of libraries which Gradio is dependent on. So now that Gradio is installed, we can check by typing "python" here and "import gradio". If this is successful, it means Gradio is working in our system. So there is no error. It means this is working. We can exit from this terminal and clear the screen.

So what we will do, we will create our first Gradio application and we will name it as app dot py. But we have to move this app dot py inside our Gradio app hyphen one. And this app dot py should be inside our working directory. So we are working under Gradio app one.

"import gradio as gr"

Then we will define function. And I want to define this function by predict. So this function will take class index. This is just a dummy function which I’m making. And based on this class index I’m going to return something.

" def predict(class_index): class_index = int(class_index) if class_index == 0: return 'cat' elif class_index == 1: return 'dog' elif class_index == 2: return 'person' else: return 'no class label found' "

So I’m returning something from this function. Okay. This function takes index and based on the index it returns something.

So what I’m going to do next is that with the help of Gradio I’m going to create an interface. In that interface the first argument I’m going to pass is the function. The function I want is predict. I want to pass predict function and the input I want to take input from a text box. And if I talk about output, output I also want to mention in text box. So whatever is going to return from this function predict is also a text right? Whatever the value is I want that value to be visible in our text box. And I want to take this input from you can say input text box okay.

" demo = gr.Interface( fn=predict, inputs=gr.Textbox(label="Class Index"), outputs=gr.Textbox(label="Output") ) "

And also this class index because I’m comparing with the numerical value I will convert this class index to int. Now because we will be passing string I’m converting string to integer. So now it is compatible. That is integer with integer, integer with integer, integer with integer. And if index is not falling under zero, one and two, I will say no class label found. And I will store this Gradio interface into demo.

" if __name__ == "__main__": demo.launch() "

This demo which we have created earlier I’m going to launch that demo. Okay. So let’s try to run this app dot py. So it is simple I have to mention "python app.py". And let’s see the response. Okay. This is inputs and outputs. That was wrong because it takes multiple inputs and outputs. Now let’s try to run it again. And hopefully we will not see any error.

And we are able to see that this particular application is running on local URL. So if you hover over it we will be able to see the link and click on this link tab will open. So this is the application running on my browser tab. And you can see this is the class index. And it is able to show you output.

So class index it is taken from the function argument. And output is just output right. So if I just mention class index it should show us dot. If I mention zero now it is able to show us cat. If I mention two, it will show person. And if I mention suppose something as 4078, it will mention no class label found.

So it is able to create this UI, even though we have not mentioned any HTML, CSS, JavaScript, React or any other thing, it is able to create this UI and when we like deploy this, it will look exactly same as running on this browser.

So currently we are not going to deploy it. This is the function which is going to drive our application. So what we will do in coming classes is that we are going to update this predict function. We are going to create a class which will take image as input. And with the help of our model, we will predict something like cat, dog or person which we are going to show as output on our screen. So we have to update this function and we will be loading few more classes, few more modules here in order to update, in order to provide all the functionality required to do the inferencing on our model.

**V) Implementing Gradio App Interface Backend**

To make image classification work, we need a back end function and in this lecture we are going to implement Gradio app inference backend. Let’s get started.

In the previous lecture, we have created a very simple Gradio application which has two different kinds of text box. One is used for input and another is to show the output. So in the input text box we were processing the input with the help of another Python function which was taking that input and based on the condition returning the possible answer. In that case it can be cat or dog, person or no label found. And we were using this output. We were using this prediction from this predict function and showing it into another text box called output. Okay.

And in this lecture what we are going to do, we are going to update all of this code as well as going to add the back end for our image classification application. Let’s get started.

So I will start by creating our back end. And I will create our back end into another module called as core, and this core will be called as module only when we will create init file. Here init file can be empty, but this is what is the requirement to create the module. Now this will be served as a module.

Now in this core I’m going to create another file called "predict.py". So all the methods, all the classes, all the functions with respect to creating our prediction is going to be present in "predict.py".

What I will do here I will "import torch". And create our first class which we are going to use for image classification. Class name is going to be "ImageClassifier".

Our class will have two components. One will be "def __init__(self): pass". This is just definition. Next function is going to be "def predict(self): pass".

So this init method is going to be responsible for loading model weights, loading our model architecture. And this predict method is going to be responsible for taking the image as input and predicting the labels, predicting the class — whether the image belongs to a dog, image belongs to cat, image belongs to person.

So in init what we’re going to do we are going to define our CNN architecture. Also we are going to load our CNN architecture trained weights. And we are going to also create index to label map because our model predicts index not label. So we need to have that index to label map in order to use this predict method. And we are going to show the same thing on our Gradio application here.

We are going to create that map. Here also we are going to define transformations which are required for any image transformation. While we were training our CNN application, we have seen that we need to apply certain transformation on the image like resize, convert to tensor and normalize as well, which needs to be applied to all the images. Right. So what we are going to do, we are going to load the transformation here and use it multiple times in predict.

So these are the tasks which we have to perform here. In the predict method what we are going to do, we are going to load image with Pillow. Why Pillow? Because our PyTorch expects Pillow. And then that particular Pillow image is going to be applied with transformation. That is the basic requirement here. After that, we are going to load that and perform prediction on the loaded architecture with the loaded weights.

We are going to do prediction. Now with the help of label map, which we are going to create here, we are going to map it to classes. And then with the help of OpenCV, we are going to write text on our image, on our input image. So if the image we are passing contains dog, our model will predict dog and we are going to write dog on top left corner of the image. That means that in this image we have found dog. So I’m going to write that text on our input image.

Next thing is the class model predicted as well as output image. And we will show it into our Gradio application. So we are going to return this here. And then we are going to implement our Gradio application. So these are the components which we have to define.

So let me write pass here. So we will start by writing code which will help our model decide whether it is going to be processed with CPU or GPU, that is CUDA. I will mention:

"self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"

Based on self.device, our model is either going to be processed with CUDA or with CPU, so we don’t have to worry about whether our system has CUDA or not. If there is any issue with the CUDA installation or anything, it will be using CPU.

Next thing is we are going to define our model, but our model is not present in our directory. In the previous lecture, we have trained our CNN architecture. Also, after training we have loaded the weights. So we are going to load that code as well as those model weights and load it here in our Gradio application, along with some things like transformation which we have used to create that model.

Okay. What are the things we need? We need CNN architecture. And using the CNN architecture we are going to create that model instance. On that model instance we are going to load model dict using our trained model using our .pt file. And then we are going to load transformations which we apply on the image. Because this CNN architecture is trained on specific kind of images, which we have already transformed earlier. We need to also provide same kind of transformation, because this particular CNN, which we are going to bring, has those transformations.

So this is the notebook where we have created our architecture. As you can see this is our image architecture. We don’t need data loader because we are only going to pass image path and get the prediction. We are not going to pass any batches because we are talking about inference here. So this is our model architecture.

Our model name was "CustomCNNModel" and it has all the components — convolution layers, a forward method, get convolution output as well. And then we are going to initialize our model. Once the model is initialized we have to load our model just like we have loaded here previously. What we have done, we have to load our model weights like this which we have saved earlier.

So I’m going to train our model for more. In total, I’m going to train this particular model for 100 epochs, and I’m going to save that path. And you will be able to find that path in the resource as well.

Also, we need to copy one more thing from here that is transform. What are the transformations which we have applied on the input image? First we have done the resize. Then we have converted to tensor. Then we have normalized those image pixels to have a mean of 0.5 and a standard deviation also to be 0.5, and the value will range from -1 to 1. I’m going to copy this transform as well as model architecture as well as that model which we have trained from here.

This is the model code. This is the model architecture code and I’ve copied it here. You can see this is model. And there are few components which are missing. That is "import torch.nn as nn". So I’m also going to import. You also need few more components like transform. I will mention "import torchvision.transforms as transforms".

And to use all of this PyTorch we need to mention in requirements.txt. First of all, we have mentioned gradio. Next thing few other things which we need are torch, torchvision, opencv-python, pillow. These are the few libraries which we need. I am going to save in the "requirements.txt" file.

In the "predict.py" we are going to load few more things like "from PIL import Image". And now we have our model architecture. This is our model architecture. This whole thing I’ve copied from our previous notebook.

And then this is the method in which we are working, that is "ImageClassifier". So now I will create, I will initialize our model here. The model name is CustomCNNModel. It has parameters like we can see here on the init method. It has parameters like input dimension and number of classes, which I’m going to copy.

" self.model = CustomCNNModel(input_dim=128, num_classes=3).to(self.device) "

Now this model weights are empty. Although we have initialized this model, this model has not learned anything. This is just plain model. But we already have model which we have trained earlier for hundred epochs. I have already trained that model for 100 epochs, which I’m going to load it here. And then using this self.model, we are going to load that file and load all the weights and biases from the trained model.

So I’ve added that module here. You can see I’ve added another folder called model. And inside this one I have this model that is .pt file. I have trained this for 100 epochs. Also I’ve created another "__init__.py" to create this. The model folder is also module. Core folder is also module which we can use in app.py. So we have this model which we are going to load and all the weights and biases we are going to store in self.model so that it is usable. It will be able to predict either cat, dog or person.

So now we have defined the CNN architecture. Now it’s time to load the architecture trained weights. I’m going to mention:

"self.model.load_state_dict(torch.load(model_path, map_location=self.device))"

So torch.load will be responsible for loading our model. So I will pass model path here because it should be configurable. And if I’m passing model path here my init should also accept model path. This torch.load is going to accept this model path. And based on the model path of this file it is going to load the model and then model location. Also, we will mention that we want to map this model location to self.device.

So now using torch.load given the model path we are going to load the model to the device, either it can be CPU or GPU. That loaded model, we are going to load the state dict of that model and which is already defined in the CustomCNNModel, but this one is going to update with the new trained weights. So self.model is going to have the latest weights and biases. After that we are just going to mention "self.model.eval()".

So we are going to initialize. We are going to create object of ImageClassifier once and use the predict method again and again. So all the components which we don’t want to run again and again we will mention in def init method. That’s why I’m mentioning self.model.eval. Once the model has been created, we have loaded the model weights as well as we have set our model to evaluation mode. So this sets all the things for model.

Next thing is we need to load index to label map. So while we were loading dataset and wrapping it up with the help of data loader, we got this dictionary, and this dictionary is what our model is used to. I’m going to copy this dictionary and save into our ImageClassifier class. So I will mention:

"self.class_names = {0: 'cat', 1: 'dog', 2: 'person'}"

Also, if you want to pass some other dictionary you are working with some other module, you can. So I will mention flexibility. If class_name is None, use default. Else use the given dictionary.

Now we have defined the device, we have defined the model, we have defined the class map, next thing we need is transformation. Again this transformation part I’m going to copy from our previous lecture because that transformation was specific for our model.

" self.transform = transforms.Compose([ transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ]) "

So now we have all the components which we require. In our init we have defined device. We have loaded the model. We have defined index to class mapping. Also we have defined self.transform which we will apply on every image. Okay.

So our model, our ImageClassifier class is going to take model_path as well as class_names.

Next we will define predict method. So this method is going to take image_path.

" def predict(self, image_path): image = Image.open(image_path).convert('RGB') image_tensor = self.transform(image).unsqueeze(0).to(self.device) with torch.no_grad(): output = self.model(image_tensor) _, predicted = torch.max(output, 1) label = self.class_names[predicted.item()] img = cv2.imread(image_path) cv2.putText(img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) cv2.imwrite(image_path, img) return label, img "

So this is how we load image, apply transform, predict class, and write the label text on the image. And then return both the class label and the modified image.

Because we are not handling any database here, we just want to show this output on our Gradio app. The output path is going to be labeled image.jpg and save this. So I also want to import OS because I want to save this file. "import os". I want to save this file in the same current directory. So I will mention "cwd = os.getcwd()" and then I will mention "os.path.join". I want to save "cv2.imwrite". This is going to be the output path, and then the image. I’m going to write this image in the same directory where we are running this path, which is going to be "predict.py". And then where this file is located, we are going to get this file path and send it to our Gradio application so Gradio application can plot it.

So currently we are getting what is the current working directory while we are running this file. Then with the help of "os.path.join", I’m going to mention cwd and then the output path which contains the image name. I’m going to mention it in output path. And then finally I will return two things to our Gradio application. That is, label — the label which we got for the input image — as well as the path of our label image. Okay, that is label and output path.

This covers our back end. Our back end will take — first of all we need to initialize our image classifier. We need to pass the model path, the model trained path. Okay, this is the trained model. And then we need to mention class names if you want to change this dictionary. Otherwise we will leave this part empty. Once this is done it is very simple. We will use predict method, we will pass the image path which we want to process, and then we will get label and output. This will be handled by Gradio and we will see while we were creating front end with the help of Gradio.

So now that our back end is ready, we will work on front end and create our application. First of all, we already imported "import gradio as gr", but we also need to import few things like "import os". Then from core — this is our core — from "core.predict" we are going to import "ImageClassifier" which we have created. This is ImageClassifier.

So next thing is we are going to mention "os.getcwd()". This is os, "os.getcwd()". I will get into a variable cwd and then I will mention few things like where our model is in the model path. I’m going to store where exactly our model is. So our model is stored inside folder "model/". This is the name I will mention "os.path.join". From the current path, go look for model and then inside model look for — let me rename to copy everything — and this is the model name "cnn128_model-100.pt". So from the current working directory go to model and get this, and get to this path and create a final path that I will store in model_path.

Then what we are going to do, we are going to create object from our ImageClassifier. First argument it takes is model path. Model path we have as model_path and then we also have class names which we are going to pass it as null. We are not going to pass any class name. And if you want to pass class name, what you can do: we can copy the same thing which we have created earlier, this is the class name just showing you example this class name, and we will pass the same class name to class_name. Okay. This is also what you can do. Instead of mentioning it here, instead of mentioning your "self.class_name" here, you can directly pass this dictionary to our ImageClassifier.

And then we are going to store this ImageClassifier to "classifier". And then we have loaded all the components. That is we have loaded the model, we have loaded our ImageClassifier and passed the model path here. And we also have this class names as well which is index-to-class mapping.

Next thing is we are going to change this method because we already have a predict method inside our ImageClassifier. So I’m just going to mention "classify_image". It is going to take image as input. Let me remove all of this. And in the function I’m going to pass classify_image. And let’s define this classify_image function. I’m going to mention image_path, a dummy image path, and I will mention "uploaded_image.jpg". Whatever this image we are going to pass to this classify_image, I’m going to save it temporarily so that we can also plot on the Gradio. So what I will do, this image_path I’m going to "image.save()" — that is from Pillow. I’m going to save this image temporarily.

And then using this classifier which we have here, that is the object of ImageClassifier, we are going to pass in the predict method the image path. So we are passing the image and what we get as an output, we are going to get label and output_path which I am going to copy from here, paste it here. This "classifier.predict" is going to return two things, that is label and output path. And then we are going to return the same thing that is label and the image object. Okay. We are passing "Image.open(output_path)". Then the output path. We also need to import Image here "from PIL import Image".

Now the input is going to be little different. The input is going to be — let me mention — it is not going to be just text. It is going to be "gr.Image". So it will take an image as input. So it is going to create a UI where we can drop our images, drop icon as well as it can open our file directory as well which we can use. So "gr.Image" it is going to take image of type Pillow. That is going to be the input. So because this image is of type Pillow itself we are going to plot it. We are going to save this with the help of "image.save()". Okay. This is the image path.

And then on the output we have multiple outputs because we have label as well as "Image.open". On the label I want to show this label on the text box. Simple. What I will mention in the output because we have multiple, we have to mention it in list. I will mention "gr.Textbox(label='Prediction')". And then I’m going to label this text box as this is the prediction. And this image which we are sending from Image.open, I want to capture this image using "gr.Image(label='Labeled Image')". Okay. This is not a text box, I’m capturing with image itself.

And I will mention title of this application: "Image Classification Gradio App". Description which we will see as a paragraph: "Upload an image to classify it as dog, cat or person." And that’s it. You already have this "demo.launch()" and our app name is demo.

Let me minimize it a bit. What we have done: we have created, we have found a path to our model, we have taken this path and passed to model path because our ImageClassifier is going to load this model with the help of architecture which we have already defined. This is the model architecture which we have defined. And using that model path, we are going to load the trained weights into our model. Okay.

Our model is CustomCNNModel. And this is our CustomCNNModel. So we need the architecture, we need the blueprint of the model. Then we are going to load the weights. And we also have a predict method which, when given the image, is going to transform the image, add the batch to it, get the image tensor and going to predict from the model. And the prediction we are going to take from the maximum logit. Okay. And we are going to do on the dimension of one because this will be dimension one — all the rows — because we are only going to have a single output. If we are going to have multiple outputs, this will look something like this. Okay. But because we are only going to have a single image, we only support single image here, our output is going to look like this.

Then with the help of our label map, we are going to — whatever the index we are getting from the model — we are going to create label from it. And then we are going to put that label on our image and save that image and pass the path of that saved image along with the label. And then we are going to use this ImageClassifier into our Gradio application. We have a function which takes this image, saves this image, uses this image with the help of "classifier.predict", we are going to predict label as well as the output path in which we have written the text on the image. So it is going to return label as well as the image, not the image path. This will be our output.

So because this is the output of our classify_image which we have mentioned in our Gradio interface as a function, that is classify_image, we need to capture those outputs. One is label — we will mention "gr.Textbox" — and then "gr.Image", which will be our image object from here. And we are going to have few things like title for our Gradio application and the description for our Gradio application. And to launch it we will mention — this is our environment which I have already activated. We have to be inside this folder. So I will go into "gradio_app1".

Now if I run "python app.py", it will initialize the server so that our application can be working here. Okay, so no module named torch. Let me install torch. As I mentioned "pip install -r requirements.txt". It will install all the things which is required. So now we can see all the libraries are installed. We will clear the screen and run "python app.py". Okay. So it is saying that "torch.cuda has no attribute available" because I have mentioned it wrong. It is "torch.cuda.is_available()" in the ImageClassifier we have mentioned. I’ll fix it, clear screen, and then run "python app.py" again.

And you can see it is successfully running. It is running on my localhost "http://127.0.0.1:xxxx". This is the URL of localhost. And then this is the port which this application is running. If you click on it, it will open this tab. So it is available here. I’m going to drag on this screen and you can see this is the component. This is the drop component, drop image component. If you click on it, it will open a file browser. If you don’t click on it, you can directly drag the image. Also these are the output things which is disabled by default because right now we have not done any prediction. So like we will not get any output, right.

I’m going to load an image. I’m going to use this image which is person and drop it here. I’m going to drag the image and drop in this section and the image is loaded. You can see this image is saved and now it is loaded as well. If I click on submit, then this particular function will run, that is our classify_image, and it has already taken the input. Then it will produce output and the output sections will be enabled. If I click on submit it will say this is a person. And you can see if you expand this part where we are showing the labeled image, you can see this is our labeled person image. It is producing two different outputs. So this is your application.

So now that we have tested our application on local system, everything is working fine. We will deploy this application because currently this is running on my local system. If you try to access this URL, you will not be able to open this application because this is local to me. If you run this application in your system, your URL will be same for you but will not be accessible for other persons. Okay. So to make it accessible across web, we are going to deploy this with Hugging Face Spaces.

**W) Setting HuggingFace Space**

Welcome back to another exciting lecture on building our image classification app. So far, we have successfully built our image classification model using PyTorch, designed a Gradio application for inference, and even tested it locally. But now, it is time to take things to the next level, and that is deploying our Gradio application on Hugging Face Spaces.

Before we move on to deployment, the very first step is to set up our Hugging Face account and configure our space properly. So let’s get started. Open any browser and search for Hugging Face Spaces. Click on the first link which is "https://huggingface.co/spaces". Once you are on this page, you will notice that there are multiple categories available like image generation, video generation, text generation, language translation, speech, 3D modeling, object detection, image editing, and many other modules. All these are applications contributed by developers across the community.

Now, just like them, we are going to create our own application which other people can also access. To begin, you need to sign up. So create a new account by entering your email address and password. Then click on Next. You will be asked for your username and your full name. Fill in all the required details, read the terms of service and the code of conduct, and then check the box to agree. Finally, click on Create Account.

After that, you will be prompted to verify your email. Go to your inbox, open the mail sent by Hugging Face, and click on the confirmation link. Once the email is verified, the notification asking you to confirm will disappear. At this stage, your Hugging Face account is successfully set up.

Now let’s go ahead and create our first space. From the Spaces section, click on New Space. For the owner, it will automatically display your username. For the space name, you can provide something like "classification_gradio_cnn". In the description, you can write: "This is an Image Classification Application". For the license, you can choose either MIT or Apache 2.0. In this case, we will go with Apache 2.0 since it is open source.

The next important setting is the Space SDK. If you open Hugging Face documentation, you will see that there are multiple supported frameworks like Streamlit and Gradio. Since we are deploying our Gradio app, we will select Gradio as the SDK. For the space template, select Blank to start from scratch. By default, Hugging Face provides two virtual CPUs and 16 GB of RAM for free, which is good enough for our application. Keep Dev Mode disabled, and finally click on Create Space.

Once the space is created, you will notice that Hugging Face automatically generates a repository for you, which will hold all your project files. This is not GitHub, but it works in a similar way. You will be able to clone this repository, move all your Gradio app code into it, and then push the files using git commands. Hugging Face will then automatically detect your "app.py" file, because by default the Gradio SDK expects this file as the main entry point.

Now, before pushing your code, you must create an Access Token. To do this, click on your profile picture, go to Access Tokens, and then click on New Token. Give it a name, for example "test1_demo". For the repository permissions, select "Write" and for inference, also select "Write". You don’t need other permissions. After that, click on Create Token. Hugging Face will display your token. Copy it immediately and save it securely in your system, maybe in a notepad file, because you will need it later while pushing the code. Remember, you can always delete or regenerate tokens from the token dashboard if required.

At this point, we are ready. We now have our Hugging Face account set up, our space created, and our access token ready. We also have the remote repository URL where we will be pushing our application code. In the next step, we will take our Gradio app files, commit them, and push them to this Hugging Face repository. Once that is done, Hugging Face will automatically deploy our app, and it will become accessible to everyone on the web.

**X) Deploying Gradio App on HuggingFace Spaces**

Congratulations on reaching this far in the course. You have successfully built a custom CNN model, created your dataset loader, trained the model, and even set up a Gradio app for inference. That is truly a huge achievement. Now it is time to take the final step, which is deploying our Gradio application on Hugging Face Spaces so that anyone in the world can access it online.

Let’s get started. I am currently inside our application space, and on this page you will notice the HTTPS URL of our remote git repository. This URL is provided by Hugging Face, and we are going to copy this URL in order to clone the repository onto our local system. Once we do that, we will push our Gradio application code into this repository, which will automatically deploy the application. It is really that simple.

Now let’s move over to our VS Code. I have my terminal open, and you can see that our application is still running locally. For example, if I open Google Chrome and go to the local tab, you will see that our Gradio application is running. When I input an image, the application is able to predict correctly—for example, it detects a person and even shows the label “person” on the image itself. But remember, this is only running on my local machine. What we want is to deploy this exact same application on Hugging Face Spaces.

So let me close the application and copy the repository URL. This is the URL we will be using with the "git clone" command along with the HTTPS of our remote repository. Instead of using the regular VS Code terminal, I prefer to use Git Bash since it behaves very close to a Linux environment. First, I will make sure I am inside the correct directory, and then I am going to clone the repository inside a folder called "SVU", not inside the existing Gradio app folder. Once I run the command, the repository is successfully cloned.

The next step is to copy all our important code components and files into this new repository. That means I am going to copy our "core" folder, the "models" folder, the "app.py" file, and the "requirements.txt" file. Once copied, I will paste them into the Hugging Face repository folder we just cloned. This new folder is still inside our local system, so the changes have not been uploaded yet.

Once the files are copied, I navigate into the cloned repository using "cd classification_gradio_svu" and check the contents with "ls". All the files are present along with the default README generated by Hugging Face, so we don’t need to add another one. Now keep in mind that for Gradio to run your application, your main file must be named "app.py". Hugging Face documentation also clearly mentions this. So make sure the entry point file is exactly "app.py".

Now comes the final step: committing and pushing the files. First, I run "git add ." to stage all the files. Then I commit with the message "git commit -m 'application files added'". Once the commit is done, I run "git push". At this stage, Git will ask for your credentials, which is your Hugging Face username and the access token we created earlier. So keep that token handy.

When I try to push, I encounter an authorization error. This is happening because I had already used an access token from another account on this system, and Git is trying to reuse it. To fix this, we will use the Hugging Face CLI to log out and log in again. First, make sure your conda environment is activated, and then install the CLI with "pip install -U huggingface_hub[cli]". In my system it is already installed, so it shows “requirement satisfied.” For you, it will install for the first time.

Once installed, type "huggingface-cli login". It will ask you to paste your access token. Note that your input will not be visible for security reasons. Paste the token you saved earlier and press Enter. Now the CLI shows that the active token is "test1_demo", which matches the token we created. That means our system is now linked with the correct Hugging Face account.

Now, we return to our repository folder and try "git push" again. This time the code is successfully pushed to the remote repository. You might see some git-related warnings depending on your system, but the important thing is that the push goes through.

Once the push is complete, Hugging Face automatically detects the new code. The page refreshes, and the build process starts. Hugging Face installs all the dependencies listed in "requirements.txt", checks for the "app.py" file, and loads your application. In our case, since we also have "core" and "models" modules, it will take those into account as well. Once the build is successful, the application is automatically started.

You can check the build logs and container logs if you want to track progress. The container starts up, dependencies are installed, and then the application is launched. Since we have already tested the app locally, we know that it should work. And now, we see the green “Running” status, which means our application is live on Hugging Face Spaces.

The interface looks exactly the same as before: we have the drag-and-drop image component, the prediction text box, and the prediction output image section. Let’s test it with the same image as before. I drag the image in, click on submit, and the model predicts correctly—“person.” Let’s try a different image, for example, a cat. Again, the application correctly predicts “cat.”

And that’s it. Our Gradio image classification app is now deployed and live on Hugging Face Spaces. You can share the URL with anyone, and they will be able to use your application directly from the browser.

Congratulations once again. You have successfully completed the journey from building a CNN model, preparing the dataset, training the model, creating a Gradio inference app, setting up Hugging Face Spaces, and finally deploying the application for public access. Your model is now live, interactive, and shareable. Great job, and I will see you in the next one.

#### **III) Deep Dive Visualizing CNN's**

**A) Image Understanding with CNNs vs ANNs**

So in this particular lecture we will be going through the visualization of Anns and CNNs. And finally we will try to build the understanding of how image gets built. Okay. So we will try to get the all the bits and pieces. And we will try to connect all the dots together so that we can finally build the images. Okay. So first of all, in this lecture we will be going through the visualization of CNNs and Anns. Okay so let's start.

So I have already picked a awesome visualization tool by Adam Hawley. So uh, right now so this is the blog, as you can see, and in this particular blog. Okay. So there are visualization tools simply. So the first one that we'll be starting is with the 2D fully connected. So this is basically an Ann based visualization tool that I have opened right now in this particular tool, as you can see. So you can see here draw a number. So simply I will try to draw a number just like three okay. And now what is going on here. You can see the downsampled uh drawing is close to three. And the first guess is three. And the second guess is four. Okay. So here the first guess that matters to us. And right now we can see the result is pretty good. So right now so this is the tool.

So the first thing what happens when the network sees three okay. Then automatically inside an an okay. And inside an artificial neural network you can see that all the pixels of the image has been connected together. Okay. So I can say that a single node is looking into all the pixels okay. And now this particular information is again sent to the second hidden layer. Okay. So this is the first hidden layer. And this is basically the second hidden layer. And finally this is your output layer. So in our input layer as you can see that we have multiple uh pixels. So it is connected with all the nodes of the first hidden layer. Okay. So when we try to do this so I can say that every node is actually getting all the information about the pixels of the image. And later on some transformations are done further on for our hidden layers.

So actually we cannot see that okay, how the feature extraction has been done exactly in an image. Okay. So based on this we will just try to look into the CNN visualization right now. Okay. So as you can see that, uh, okay. So you can actually play with this tool okay. A lot of things that can be done here. So the first part that comes around. So if I just try to click here you can see the input layer unit 28. So it has multiple. So this is 250. So simply you can look into this is 28. And finally this is 28. So the final size is like 28 into 28 which is equal to 784 okay. And this particular 784 is basically the total number of pixels available in our input image. And that same information has been passed to the hidden layer. Okay. So now let's just go back. Okay.

So I will try to just take a step back and look into our CNN based convolution. So right now when I'm talking about a CNN based convolution. So I will just also just try to draw a three here okay. And you can see okay. So the feature extraction part is right now started. You can see the layer visibility. So multiple layers are there. The input layer the convolution layer one. Then comes the downsampling layer one okay. Then comes your second convolution layer and then finally downsampling. So when I'm talking about this downsampling layer, you can simply think about as a max pooling layer okay. Which you have already learned in the previous module. So and finally we are closing the network with a FC layer which is basically your hidden layer or dense layer. Okay. So we have two. And finally you have the output layer.

So right now if you look into this particular visualization okay. And let's try to see actually what is happening. So in this particular the tool in the first step okay after the input layer because here also a very similar configuration of 784 pixels which are available. Okay. But in the next step if you look into. So this is the first convolution layer. And in this layer there are total how many feature maps if I talk about. So this is 123456 total six feature map have been used Okay. And now each feature maps. Okay. Or simply if I talk about, like when we get feature maps. Okay. It totally depends upon the number of filters that we are using, or the total number of kernels that we have used. So in this particular convolution layer we have used six different kernels. So for that we got six different type of feature maps.

And right now you can see that three has been extracting in different ways okay. So if I talk about three you know that there are multiple type of curves in three okay. So somewhere it is horizontal. Somewhere it is vertical. So it is actually trying to get those type of features finally okay. And in the next step. So this is the second layer which is basically the downsampling layer or the max pooling layer okay. So here you can see the reduction in the size. So the first part. So right now I have just clicked here. So this is a convolution layer okay. So simply it's written as like filter three in that there is a unit which is basically the 398 unit. And right here you can see the weight which is getting passed here, which is 0.09 in negative. And finally, based on the calculation you can see here like ten has been used as an activation function. And finally the output has been shown to you.

So here total six kernels have been used. So six feature maps. In the next step there is a max pooling has been applied. So the size will be reduced this time. Okay. So if I talking about the size so you generally know that it gets halved. Okay. So what about this size. So generally it started with 28 cross 28. And after that if we take a convolution of three cross three okay. And then a pooling operation and then again a CNN based operation which has been applied. Okay. So this is a CNN layer. And after that again a downsampling layer, you can see the size has reduced pretty much. And then finally this has been connected with your hidden layer, the first hidden layer. Okay, so I can say that all the features which has been extracted by CNN. Okay. Now the pooling layer. Okay. That information is being transferred to the first hidden layer and from there to another hidden layer. Okay. And finally, which is connected with the output layer that we have. Okay.

So right now if I try to compare Ans and CNN so simply, you can understand that the feature extraction is much clearer. In CNN based networks. We are where we are using convolution layers. But if we are going to use ans then this feature extraction is pretty much naive. Okay. We won't have any type of ideas. Okay, so this is why I thought about just giving you a rough idea. And then we will proceed ahead with the concept of image building. All right. So the first part is like why not? Okay. Now simply like CNNs are better with respect to feature extraction. Okay. And why not? And so if I am talking about Anns.

So the first problem that we do face okay. So I will just try to mention the problems with CNN. Problems in Ann. The first one is loss of features. So when I'm talking about this loss of features it means that you are converting a 2D into 1D. So when you try to convert an image into an 1D. So the first thing that you lose is the structural integrity of the image. So what do I mean by the structural integrity? So let's take any object in an image. So for example if I am talking about a dog so you know that what are the features of a dog. The dog will have eyes cat okay then. Eyes tail okay. Legs. Body. Fur. Nose. Tongue okay. Now they have a very specific placement. So I cannot place the tail of the dog with the face of the dog. Okay? Because we know that it is connected with the body though. But yes, all the features of the dog. Okay. It has to be in correct place. Then only we can say that it is a dog, otherwise it will be a totally different animal.

So that's why when you try to convert a 2D into 1D, it will completely lose the structure of the object and which we don't want. So because of this, like aliens are not so much preferred and we will be using CNNs with depth with the different type of variations. Okay, so that it will be much better in feature extraction. Now CNNs are also very good in one important thing. Okay. So let me just try to write the prose. There is something called as spatial. Invariance okay. So when I am talking about spatial invariance so what does it mean. Okay. So it is basically the model ability. To generalize. Features. And patterns in an image. Okay. So. This is referred to as spatial invariance. So when I'm talking with respect to spatial invariance you have to understand that this feature is completely missing in an okay in an there is no ability of the model to generalize the features and the patterns which is available inside an image. Okay.

So that's why because of this feature okay. It is actually very good to work with CNNs whenever you are working with respect to images and videos. Now let's come to the second part. So there is something called as. Location invariance. Now this is also a very similar property like the spatial invariance but as the name suggests like translation. So for example let's take the image. Okay. In that in this particular image I am taking an object. Okay. So let's take something like a person here. Okay. So the network should understand that if it is getting changed. From one place to another okay. So if I say in the next step the person is here, okay. Or probably in our next step, we can simply say that okay, so the person has changed the position in the image from here to here. So simply the CNN should understand. Okay. And this is basically the translation invariance that I was talking about. Okay.

So this is basically the property that allows the network to recognize objects in an image, even though they have been shifted or translated in different positions. So right now so this is basically the positions that we were talking about. So these are the two main pros if I'm talking about okay. Because of which it works. Very good. Now we'll be jumping towards our image understanding concept a little bit okay. So like what are the bits and pieces. So this is how a simple CNN network looks like okay. And if I'm talking about any type of CNN based network. So the first network if I am talking about. Okay. So this is a similar network that we have. Okay. And here you can see. Okay. So convolution layer two is there. Okay. Then again max pooling is there. So multiple types of layers are available right now.

So the idea should be that in a CNN network okay. The idea is for feature extraction in multiple steps. So that's why I kept it. So for example the low level features. So I can say that if I have three convolution layers okay. So this is just an assumption for now. But like going forward you can actually break your network. So the first part of the network is responsible for extracting low level features okay. So low level features when I'm talking about. So these are something like this edges basically okay. So simple edges and gradients we call them okay. The second level I if I'm talking about. So based on the assumption. So if I talk about my second CNN layer okay. So it will be responsible for extracting my mid-level features. Okay. And finally, my later CNN layers will be responsible for extracting the high level features. Okay.

Now this is very, very important because this is how you finally build an image. Okay. So, uh, as you can see that I have taken multiple examples here. Okay. So like low level features mid level features. Okay. And finally like here like layer one is responsible for extracting your low level features. Layer two for mid level features. And finally comes your layer three which is for your high level features. Okay. Just like you can you have started seeing the features right now. Similarly for other objects, if I'm talking about like faces, cars okay, elephant chess a very similar process is always used. Okay. So right now if I talk about the smallest component okay. So the smallest component is always the pixels. After pixels you will be having the edges, okay from there like parts of object. And finally you will be having your object models. Okay.

So object models when I'm talking about so they are basically the object okay. Or like the different because in my data set there will be multiple faces. So that's why we are calling it the object models. So this is another example of feature extraction. As you can see that in the uh input layer we have past faces. Similarly for the first convolution layer you can see that some type of stroke marks. So it's just trying to understand some horizontal if I'm talking about vertical features or probably some type of diagonal features in the next level of convolution it is extracting more. Or if I'm talking about some mid-level feature and finally comes your convolution three. So like this is how the intuition is to build up your image okay.

So based on this. So we can follow a simple rule. The rule that will be following here so will be following. We will be always starting with edges. Okay. So when I'm talking about edges okay. So edges simply means okay. We can call it edges slash gradients also. Now from edges and gradients we will be having some type of textures. From textures we will be having patterns. From patterns we will be having parts of object. From parts of object we will be having the object. Okay. And finally we'll be having the scene. So scene is basically the complete image okay with the background. So probably if I talk about an example of the scene so you can think about there is a, uh, dog in the image. And behind the dog there is a building. So that becomes a complete scene. Okay.

And for this also I have taken a simple, uh, example, okay. Which will help you to actually visualize each of the steps in detail. So the first part that comes around is basically the edges when I'm talking about. So this is how the edges looks like. So this is the low level features that I am talking about right now. Your initial convolution layers will be used for this type of feature extraction. Then in the next layer as you can see here simply you have your textures okay. Now textures are something like using multiple edges. You will be building multiple types of textures. So this is something like if I talk about, okay, the mid level where you are actually extracting some type of textures. So basically the idea is you will be using multiple type of edges. So using those multiple edges you will be building your texture.

Then finally once you have extracted the textures the next level will be basically to look into the patterns. So using multiple textures we will be building the patterns. And finally once you have your pattern. So right now as you can see. So from here we can understand that okay it looks like something. So probably if I'm talking about this. So this looks like flower. So from this part we actually understand that okay. So this is how an object looks like. So if I talk about this one. So probably the face of the dog is coming okay. And probably here. So some type of balls this is some type of network if I try to visualize it. And finally once you have your patterns you can see the parts of the object. So this is where it is much better actually. From here you can actually take a decision. Okay, so like how the object looks like. So this is a flower. If I talking about this is some type of dashboard of a car. This is probably the nose of a dog if I talk about okay.

And finally when you start looking into the objects okay, you will have always a better idea. Okay. So right now I can say this is something like the face or the eyes. Okay. So this is how the dog face with the ears that is coming okay. So in this way we try to build an image. So this is a very important rule. The first is basically the edges. Then you have your textures. Then you have your patterns. Then you move to parts of object, then finally the object and the thing. Okay. So this is basically the rule of image understanding. If I talk about and this you have to remember because for every network will be following this rule.

**B) CNN Explainer**

Welcome to the lecture of CNN explainer.

So in this particular lecture, we'll be going through the visualization of the entire CNN network. And what happens when you apply kernels. When you apply activation activation functions it will give you much more clarity. Okay.

So I have already provided the link so directly you can jump into the link. So let me just switch. And this is how the CNN explainer looks like okay. So there is a research paper attached to it. So directly if you just click on this PDF okay. So you can see that okay. So the CNN explainer is available now. Similarly you have a corresponding and video and YouTube video which is also available. You can actually go through that.

So right now in this particular network. So let's go step by step. So what are the different classes that they have? The first one is basically the lightboard that you can see. Then comes the ladybug, then pizza, bell pepper, schoolbus, koala, espresso, red panda, orange and sports car. Okay.

So directly you can see here you can simply click on this. Okay. So for each of the image it will show all the visualizations. So let's try to go in depth and try to understand. So first of all let's take the example of the lifeboat. And let me just zoom into this.

The first part. What happens the image. The input image is broken into three different channels: the red channel, the green channel and the blue channel, the RGB. So whenever you are passing a monochrome. Okay, so for sure it doesn't have an RGB channel. But if you are working with color images so it will be having for sure. So the first thing as you can see that these three channels okay, becomes input for your convolution layer.

All these three channels becomes the input for the first convolution layer. You can see conv underscore one underscore one. And right now the first input image that you have passed, the input size is 64 × 64 × 3. So 3 stands for the number of channels and the width and the height that is 64 and 64.

So after performing convolution you can see the size has been reduced because whenever you perform convolution operation you will be losing pixels at the top, at the bottom okay. We will be coming in depth why it loses okay at a later point. But for now, yes, it is losing. And you can see that from 64 it has become 62. And right now based on this. So we can easily assume that okay, it is using a 3 × 3 kernel for now.

Now in the next step after the convolution you can see. So this is how the feature maps looks like. So little bit understanding if I just click on this so you can see that okay, so how this particular thing is being passed. So this is the intermediate thing which is being shown right here. And finally this is how it looks like okay. So the weights and the biases have been shown here.

Now if you just try to click on this so simply it will try to show you the mathematical convolution operation how it has been performed. So step by step. So right now if you look here as you can see that it's going like this. Okay. So I can say that it is convolving on top of the entire image okay. So the direction will be from left to right. And then it goes down, then again left to right. Then again it goes down left to right, left to right. So the signature that it follows is a left to right always. And right here you can see the values.

So similarly these are the values okay with respect to. So right now you can see the bias has also been added. So similarly if you try to look for some other. So right here okay so I have just tried to play it. And you can see that our left to right. And then it goes down. And finally it's try to build the feature map. So I can say that from all this feature map the finally this is how it looks like. And the total number of feature maps are 10, because we have used 10 kernels right now or 10 filters for now.

Okay. So now let's get back okay. All right. Now after the convolution operation. So there is an activation function which is ReLU that has been used here. And after the activation function you can see that there has been some type of changes with respect to the feature map. And if you click here so directly so max zero comma x. So that is how the ReLU function works. And you can see that okay after applying activation. So this is how finally it looks like.

Now similarly, you can have a look into other feature maps and then pass it with an activation function. Okay. You will have a very good idea. And right now. So this is basically the first convolution layer with the activation. So similarly you have your second convolution layer. So right now from 62 okay. So in the next convolution layer the size has been reduced to 60 okay. And you can simply click on this. You can see that all these feature maps has been passed okay. And finally you get your the first feature map in the second convolution layer okay. And that led to uh write rule okay. So that is basically the convolution operation which is going on top of the image.

So similarly here also the total number of kernels that are used here is basically 10. In the next step, again using the activation function ReLU. And finally moving ahead with max pooling. So when you are applying max pooling okay. So at this particular point if I try to do it you can see that okay. So input 60 × 60. And finally the output is 30 × 30 okay. So I can say that okay the dimension has been reduced by half.

So after max pooling again a convolution layer that comes around okay. So 28 × 28 okay. Then again a ReLU that has been applied and finally comes here the convolution layer. Then again ReLU okay. So from here 28 × 28 become 26 × 26 finally comes here ReLU. And then again a max pooling where the dimension got reduced directly from 26 to 13.

Okay. And let me just try to zoom out. Okay. And finally you can see so the hidden layers that are available so directly like this part I can click. And this is where the flattening is happening okay. So by default the flattening you cannot see. But you have to understand that yes there is a flattening layer.

So after the max pool there has been a flattening operation and this particular flattening operation. So if you just try to do the multiplication like 13 × 13 × 10 okay. So based on that idea so 13 × 13 is 169. And then multiply it by 10. So all this information has been encoded into a single 1D array. And finally adding up the bias okay. And you can see that the output that you are getting. So you can simply click it here.

So let me just click on this okay. So this is where the softmax okay with the logits. And right now you can see that okay. We have selected the first one the lifeboat. Based on that you are getting the output. So similarly if you change it to some other okay. So I will just close this one and let's try to. Um okay. All right. Let's take espresso for now okay. And what I will try to do. So let's look into this espresso part.

So right now based on this you can see that if I show you the softmax okay. Now this is having the highest value that I can say normalized logits into class probabilities. So this is where softmax take place okay. And this is based on the formula E to the power Z okay. For like each of the classes. And finally adding them up. And this is where you get the output.

Now from this visualization I can simply say that okay. So like the entire working of the CNN with respect to using all the activation functions, the max pooling, the CNN layers, everything gives you a very good idea. Now, what is this like the CNN explainer? It gives you a very good intuition, like how what is a CNN and how it works.

So if I talk about the basic building blocks, so there is a tensor, an n dimensional matrix, then comes your neuron can be thought of as a function that takes multiple inputs and yields a single output. Then comes your layer or simply a collection of neurons. And then finally comes your kernels, weights and biases. So very very important because weights and biases are the trainable parameters okay.

Now what are the different like layers if I talk about the first comes is basically the input layer. Then you have your convolution layers which you have already learned. Then finally comes your max pooling layers. Okay. So if I talk about so there is an activation layer softmax layer okay. You can see the pooling layer. Then comes the flattening layer.

So let's quickly go through one by one. So you can understand that the convolution layer is basically where like the filter will be convolving on top of the entire image from left to right. And finally it will try to extract some features. So this is basically the working that it has shown here. And so this is a simple example like input okay. And right now the feature extraction. So the kernel values okay. So that gets multiplied basically. And this is how you get the output okay.

So if I am so right here so these are the values from the input. With that your kernel values 0.25 okay. And right now you can see I'm just hovering over. So this gives you a very rough idea that how the values are getting changed. So similarly this is how the convolution works. And then comes your hyper parameters.

So input size. So this is basically the image input size or the feature map input size for now. So which is 5 × 5. So total 5 × 5. It means that the total number of pixels that I have is right now 25. And finally from here we are getting an output of 4 × 4. So here you can see padding is zero by default. The kernel size that I'm using is 2. Let's make it 3 okay. And you can see right now the output becomes 3 × 3. So by default the stride is always 1 okay.

And now let's try to hover over this okay. You can see step by step how the entire convolution operation is taking place okay. And it's done. So if I talk about the input size it was 5 × 5. And then the output was 3 × 3. It means that I can say that we have lost two pixels. Okay. And again two. So total number of pixels that we have lost is 4 because we have two different dimensions the width and the height right.

So similarly you can play with this hyper parameters for your better understanding. So for example if I make the kernel size 5 then what will happen. So directly you can see that the input is 5 × 5. And then in the output it's only 1 × 1 because right now the entire convolution has been done. So the input size and the kernel size was the same. So finally we got a single pixel in the output okay.

So generally we will be using odd size kernels or filters. So 3, 5, 7, 9, 11. And even I am going to tell you why we use odd size much more. But these are few of the hyperparameters simply that you can play with.

Now coming to activation functions. You know that you do have the choices of using different types of activation functions, because there are different variants of prelude that are available. Okay. And finally comes your softmax your pooling layers and your flattening layer.

So like this particular tool, the CNN explainer is a very good start just to understand the entire working of CNN networks.

All right guys, I hope that this visualization gives you a better idea with respect to like if you add an activation function, then what happens to the feature map similarly into next level, then what happens? And finally like how the object changes with respect to multiple type of operations like convolution activation, max pooling. Okay, it gives you a rough idea okay.

So I hope that you enjoyed this visualization, and I hope that you have a better understanding that how CNN works.

**C) Visualization with Tensorspace**

Welcome to the lecture ten Subspace Playground.

So right now we have looked into multiple types of visualizations of CNN networks. So here also we will be exploring one of the last ones, the tensor space. So I have already provided the link. So let's jump into the website.

So right now this is the Tensor space playground. And you can see that here multiple type of architectures are there right now, different type of CNN based architectures. And you can actually like click and drag and you can actually open all the layers. So first of all let's start with the most basic one, the LeNet.

All right. So as you can see here simply, this is how the network looks like. And we will try to collapse it much more. So first of all let me just right here. Okay. And you can see here the linear things, you write a three. As simple as that. Now let's see what happens to three step by step. So this is basically the entire thing, I am just zooming in and zooming out.

Now let me just rotate it. Okay. So from this angle now the first thing. So this is basically the input. So right now you cannot look into the dimensions. So if I just try to click it here, you can see here simply it's written 28 by 28. So first thing, the input image that goes into okay the resolution that comes around is 28 cross 28. So let me just rotate it a little bit so you can simply see.

Now after that you can see that in the next layer, the second one, so it becomes 32 cross 32. Okay. So it means that there is an increase in the size a little bit from 28. And finally these yellow colors are basically the CNN layer, okay, where there are multiple type of filters. And finally this is basically the feature maps—total six feature maps. So this yellow color that you can see, this is basically the feature map. And right now the size of the feature map which is showing is 28.

So you can see that there has been a reduction. So first of all here it's 32. So simply you can see here it's 32. And here it's 28. So after the convolution operation we can see that there has been a reduction in the number of pixels.

Now moving towards the green color one. So this is basically the max pooling. So if I just try to click here, okay, you can see that it has been now a single block. But if I click here again, back to our original one. So right now, if I talk about the total number of filters used in the CNN layer (in this yellow color one), so total six were used. So right for the max pooling also, total six feature maps that you can see. Okay. Because in max pooling the total number of features doesn't change.

Then comes to the second CNN layer. I will just try to click it. And right now as you can see here, total is like four and four. So 1 2 3 4. So four cross four which is 16. So in the next convolution layer I can say that we have used 16 filters. So based on that we have 16 feature maps. And right now if you look into the size. So for this, okay here it was 28. In the max pooling layer it became 14. As you can simply see the number is written.

So if I just try to rotate it a little bit and show you the green color. Okay. So right now based on the… let me just try to rotate it much more from this side. Okay. You can see the number 14 that comes. Okay.

Now let's jump into the second convolution layer and let's look into what is the dimension here. So if I just try to click here, okay, and then I will just try to zoom it. So you can see that right now the size is ten cross ten for the second convolution layer feature map.

And let's look into the max pooling. Okay. I will just try to zoom out a little bit. Okay. And now if I just try to collapse it again you can see that there will be 16 feature maps for the convolution layer. Okay. So here also again the size will be reduced from ten, because if you remember in the last step, okay, once I click here it's ten. Now let's look into the green color which has become now five.

So if you just try to zoom, okay, you will see that okay, the number five comes here. Okay. So let me just try to zoom it. Okay. All right. You can see here five. This was ten. Now it has been directly halved with respect to the dimensions.

Now let's look into our hidden layers. So this green color is basically the hidden layer that we have. So I will just try to zoom it here. And I will just try to click it here. And right now you can see that that 1D array type, okay, which is basically the first hidden layer. And similarly you have your second hidden layer. So let me just try to collapse that one also.

So this is the second one. And right now if I just try to zoom it. So let me just try to zoom it. Okay. So the first hidden layer that comes around and this is basically the second hidden layer. Okay. So as you can see now the most important thing, the things are interconnected.

So if I just try to click here, you can see that the first hidden layer is basically taking all the inputs from the max pooling layer. So you can see that everything is interconnected. Now in the next step, if I just try to just go here, and right now, so the second hidden layer. So everything is interconnected right now. And finally this is your output layer.

So like if I… so this is a single block. Let me just collapse it and I will just try to zoom it here a little bit. This tool is little bit like tough to play with, but yes you will get a very good intuition. So right now as you can see that okay, so this is five, okay, for like 3 2 1, all the different classes that we have. And finally based on this, this is basically the output that we are getting, which is basically the number three. Okay.

So now like this is one of the best visualizations with respect to CNN network understanding. And I hope that you enjoyed this. And from this, things will become much more clearer in your mind with respect to how each of the blocks are working for the CNN network.

So I hope you enjoyed this lecture. So guys, see you in the next one.

**D) CNN Filters**

Welcome to the lecture CNN filters.

In this lecture we will try to go in depth with respect to filters. So first of all I have written like multiple terminologies which are used for filters. So the first one is basically the default one the filters. The other name is basically the kernels. And finally comes your feature extractors. Now what is the job of a filter in a CNN to extract different type of features?

So right now I have taken two different examples in this particular example that I have. So this blue color is basically the input okay. So if I talk about this. Is basically the input the blue color. Then you have your green color. Okay. Now this green. Is the feature map. And finally, the black color that you can see that it is moving from left to right. So this black color is basically the filter. Okay. So we can call it kernel or feature extractor also.

So right now as you can see that in this particular example. So what is the size of the filter. So generally when I'm talking about the filter the filter is convolving on top of the input. Okay. And the input is basically the blue color. And finally we get our output which is the green color. The feature map. So right now in this particular example if I talk about the size of the filter is three cross three. As you can see that total. Okay. So filter size.

Similarly in the next example if I'm talking about. Okay so here I have changed the input. Okay. So based on that the output is also changing. So in this case if I talk about the input. Is five cross five the blue color one okay. And finally the output that I am getting. It is three cross three okay. And similarly for the next example if I am talking about the filter size. Again it is three cross three. Okay. And then I have the input okay. This time I have changed the size of the input. Which is four cross four right now. And finally, based on that, as you can see that our output has also changed. And the output is two cross two.

Okay. So from this particular example just try to understand that the filter is convolving on top of your input. And finally the output that you get is basically the feature map. And right now we have multiple names. So sometimes we call it filters sometimes kernels and sometimes feature extractor. So the first most important thing about the filter if I'm talking about it's all about the size. So what should be the size of the filter. This will be covering in depth in the later lessons. But for now just try to grab a basic idea. The size of the filters actually mattered to us.

Okay, so for example, generally the sizes that we use the most common one is three comma three five comma 577. Okay. So these are the most commonly used filters. Okay. And majorly if I'm talking about we use odd size filters. Now sometimes we do use even size filters also. But they are lesser used as compared to odd size filters. We will be coming to this particular point. Why odd at later lectures, but for now the size of the filters matters because if you are using different size of filters. The output or basically the feature map will be different in most of the cases.

Then comes depth. So when I'm talking about the depth, just think about an RGB image okay. A color image. So when I'm talking with respect to color or RGB image so you know that. So it has three different channels. So for those three different channels. So I know that the minimum number of filters or kernel or feature extractor that I will be using is three okay or more also. So you have to understand your input matters. If in the input there is a depth. So based on that you have to select your number of kernels.

And finally coming to the next part which is basically the stride. Okay. So stride basically defines the movement of the kernel. Okay. So for the explanation of stride. So let's jump into a visualization. So right now as you can see in this particular visualization I have taken a different input size. So last time I have used four cross four five cross five. But this time I have selected seven cross seven okay. So the padding is zero. And right now the kernel size that I have taken is three cross three. And by default the stride is always one guy's okay. So you have to remember that by default the stride is always one. But you can change it.

Okay. So right now what the visualization is looking like okay. So if I try to hover over this you can simply look into that okay. So it's not skipping any pixels. But if I actually change the stride from 1 to 2. Okay, so let me just do the changes from 1 to 2. So right now you can see that it's skipping some of the pixels. So if I'm talking with respect to the skipping part. Okay. So you can see that. Okay. So the this particular part actually skipped. Okay. Then if I move to then again it has skipped. So actually if you try to use like higher stride you have to understand that you are always losing some type of information. So it's not a very good practice of using high strides. So by default the stride is always one. You can make it two sometimes, but don't use with higher stride values because most of the times then you will be actually losing information because of which the feature extraction process won't be very efficient.

All right. So now I have changed this. So just think about if I make my stride. Two you can see that there is no impact in the shapes right here. Okay, so let me just, uh, keep the stride one. Okay. So the size impact matters to us. So I want to show you that how stride can actually affect the output feature map size. So first of all the input is seven cross seven. The kernel size is three okay. And the stride is one. So right now based on that the output is five comma five okay. If I change stride. You can see that it has reduced to three cross three okay. And finally if I increase stride more okay. So right now you can see that okay. So this becomes the red actually okay. So like uh it's not good okay. But so stride of three is not taking as an input right here. So stride of four which is working. And based on that we have an output of two comma two.

So actually the size of the stride actually matters to us because the output will always be in smaller shape. If you try to increase the stride value, then the output will always be smaller because from seven cross seven total 49 pixels that I have. So it has changed to only four pixels, which is two comma two. So generally you will be using 1 or 2 not more than that because of the loss of features. So I will just make it one okay. And it's perfect. Okay. So most of the times we'll be using one and two only.

Now let's get back to the notes. So here also I have taken a simple example. For the stride. Okay. And I have kept a simple visualization. Okay. So right now in this particular visualization, if you just try to have a look. Okay. So you can see that yes the kernel is moving. So in this case a simple case of stride is equal to two that I have taken into account. Okay. So stride. Equal to two. Okay. But by default it is always one. Okay. So by default I will just add this. By default. It will be always one. Okay. So from here we got the basic understanding of stride the depth and the size for any type of filters.

Now let's jump into and actually like what is the filter. Because as I told you the size of the filter is three comma three. It can be like five, cross five, seven cross 7 or 2 cross two. Okay, so actually how the filter looks like. So this is a simple example that I have taken okay. And right now here total there are six filters. So when I am talking about this six filters you can see that in each of these filters there are nine values okay. So I will say that three cross three six filters okay. And everywhere there is nine values. So simply I can say that okay. So you can think about a matrix. And in this matrix you can think about there are some type of values. Okay. So for now let's just assume so something like this.

Okay. So in this particular matrix I have my nine values. And for this also. So at any of the filters if you try to look okay now the values will be different because we don't know the values initially okay. Now that's why we call also filter as learning filters okay. So this term is also used learning filters because the values of the filters will be learnt during the training process. Okay. The idea is that the CNN will adjust the filter values to minimize the error between the predicted and the actual outputs. Okay. So simply like updating the values. If I am talking about updating values is taken care by back propagation. Okay. Your back propagation algorithm.

So during training these values will be efficiently updated so that your loss will be minimum okay. So at any point of time when you think about a filter, you have to think about okay, if the size of the filter is three cross three, then this is just like a matrix which has nine different type of values. And now based on that it will keep on changing okay. So that our error will be minimum. And finally those will be the filter values.

Now let's jump into the kernel visualization a little bit. So I will just go here okay. And let me go to image kernel. So right now this is a very awesome blog okay. Which gives you a very good idea with respect to the visualization part of the kernel. So as I told you. Okay, so total if I take an example of a three cross three kernel okay. So right now there are different types of three cross three kernel. We will go in depth with respect to the kernels. But for now this is one example the sharpening kernel.

Now you can see that there are some values. So based on these values you will get your final feature map. So right now so this is our input image as you can simply understand okay. And the filter is actually convolving when I am moving okay. You can directly see that from left to right the filter will convolve. And based on that we will get our feature map. So this is basically the feature map. So right now if I tell you so this is basically the kernel value that I am taking. And based on that kernel value this is the output okay. So the input then becomes your kernel. And finally this is your output.

And let me just change the kernel here. This is a sharpen one. So let's start with something blur. Okay. In blur you can if you look here I will say that it's some type of averaging operation that it is trying to do. Okay. So if you look here so 0.0625 okay. Now this is actually multiplied with 255. So if I actually take it here you will see that okay. So like right now the pixel value of the input has changed which is actually 206. And right now you know the range. The range is between 0 to 255. So based on that 206 into 625 okay. So based on that you can see that we have our first output. Then if I move to in the next one. So similarly this particular value. So let me just take something from here. Okay. And we are getting the updates.

So whatever the input values you can directly multiply it with the kernel just like your convolution operation. And finally you get but we have different type of filters. This was a blur filter. Then comes a bottom sobel. So right now if you look here. So somehow I will say that this is actually focusing much more, uh, towards some type of edges, because the job of a kernel or a filter is to extract features. And right now it has extracted some type of features. Now let's change our kernel from bottle Sobel to emboss okay. So right now you can see this is a different type of feature which has been extracted.

If I look into the identity okay you will look that it's pretty much the same. That's why the name is also identity kernel. And if you look into the values of the identity okay. So like 000 in the mid there is one rest. Others are all zero. Now let's change this kernel to left Sobel. Okay. So right now if you look at it is much more focusing towards the vertical features. So you can see this much more vertical features that it's actually looking into. Then comes your outline. So right now you can see that it's focusing on vertical as well as horizontal features. Okay. And then like top Sobel. Okay. So this is also I will say that it's focusing much more towards the vertical and horizontal. Much more lesser vertical much more towards the horizontal part.

So in this way whenever you think about a filter you have to think about what is the size of the filter. Once you think about the size, then you can directly think about okay, so those are basically some mathematical values which are being put and using this values will be simply performing the convolution operation. And based on that we will be getting our feature map. So this is how the feature extraction process happens. Now the question comes like if you use ten different kernels, the idea is always that it will try to extract unique features. Ten unique features. So if you are using ten different kernels. So ten unique feature maps or basically ten unique features will be extracted. And finally you will get your feature maps okay.

So similarly you can play with this. So right now as you can see. So this is just like an input which has been provided. And you can actually change out the values here. So let me change this to one okay. And you can see right now uh I will say like here horizontal features are much more focused. If I change this to one much more you can see as horizontal. Now if I make this as one okay. So right now you can directly see that okay. So it will be that identity effect which has been added okay. So if I make it zero let me make this zero. Okay, so not a lot of changes if I make this one zero. So again that black ish color will come. So actually like you can play with this and you can play with these values.

So if I just make it 0.5 okay. So you can see that little bit towards gray side. Last time it was much more whitish okay. So similarly like 1.2 I am just increasing it. And in this way you can actually play with it so that you can understand that whenever you're having the filters. So they are having some unique values. And those values are actually affecting your input. And based on that, finally the features are extracted and you get your feature maps.

I hope that from this particular explanation you got a better understanding with respect to filters and what filters are doing. I hope you enjoyed this lecture.

**E) Building our own Custom Filters**

In this video we'll be going through building our own custom filters. So we will try to build some simple filters just like horizontal, vertical and diagonal edge detectors. So as in the last lecture we have gone through that we'll be using odd size filters. So here also the same thing odd size filters that I will be picking up. And the size of the filters will be three cross three. So we will try to manipulate nine values within the filter. So it is just like a matrix. And inside those matrix I have my nine values. And I will just try to manipulate them. Okay.

So let's get to our notebook. So this is basically the notebook. Notebook for building the custom filters. So right now, uh, when I start with the code. Okay. So there are two important functions. The first one is basically downloading the image from the internet. So as you can see here. So with "URL open" and then I will be passing the URL link okay as the response. And then I will be getting the image data. And I am reading it using "numpy" okay. And finally the data you can see "dtype" that I have selected it as "int eight". And finally like I am decoding the original image and finally returning the image in the similar way when I'm talking about.

So once this image is available, then the next step it becomes like rendering the image in the notebook. Okay. So for that I can use uh direct "cv2" or I can use "matplotlib". So in this particular function you can see that I am reading an RGB image okay. And in that so I am basically changing the color scale from "BGR" to "RGB". So this takes place very specifically in "OpenCV". In other libraries. You might not need to do it. Okay. And finally rendering it using "matplotlib". So the first part. So I have my functions ready and I am just passing the URL. Okay. And finally I am displaying the image here. So let me start by executing the first function. Okay. So like both the functions are available the "download image" and the "display image".

In the next step I am just passing the URL. So in this URL the image is available. And finally like downloading it and displaying it. Okay now this particular display we have done using "matplotlib" but using "cv2" also you can do it. Okay so like from "Google Colab". So they do have the patches of using "cv2 IAM show". So using that also you can actually simply render the image. So in the next step we will start with our edge detection. Okay so right now so there is a default function from "OpenCV". And one of the famous edge detectors which is referred to as "canny edge detection". Okay.

So right now I am just importing it and I am passing my image. So in the last step, as you can see that I have read my image basically using this "image" variable. And finally what I am doing I am passing this image inside my canny edge detector. So "cv2 dot canny" and inside that and then basically the resolution that I am passing here. Okay. And in the next step. So let me just execute the edges. And finally let me show you after execution how it looks like. So right now if you just try to analyze the original image okay. So in that image. So whatever types of features if I am talking about okay or edges right now okay. Because what are the features. Those are basically the edges and right now based on this. So you can see here we have our vertical edges. Okay. So some type of horizontal edges. So majorly it's trying to cover the diagonal edges also.

But this particular. So right now we have not defined a filter. We have just used a default function from "OpenCV" which is basically the canny edge. Now in the next step we will be building our vertical edge detectors. Okay. So right now as you can look into. So this is basically the three cross three matrix okay or basically the filter that I am giving. So in this particular filter I am passing the values. So in the first column you can see everything is minus one. Then in the second column everything is zero and finally one okay. So let me just execute this and let me just show you how it looks like okay. So right now if you look into this particular image you can see that my most of the horizontal features are not so much detected here, it's focusing much more towards the vertical features.

Now if you just try to compare here the top part of the P okay, here the top part of the P is not so much visible, but this long straight line of the P is actually visible. Similarly, you can actually look into like every day here okay. And here every day. So little bit like the horizontal features. It's not so much focusing, but it's actually right now focusing much more towards the edge detector or basically the vertical edges right now. Similarly if I change this particular matrix value. So what happens right now in the next step. So last time we have used minus one zero and one. So this time we have actually changed the columns okay. The first column is right now one. And the last column that I have that is minus one. So little bit that I have changed. And now let's have a look.

All right. So very similar. Only I can say but little bit if you look into this frozen part like f r okay. And now if you look into this frozen so you can see that uh, some part it is focusing on the vertical much more and lesser towards the horizontal features. So you can see the f, uh, dashes are completely missing. Similarly you can check it out like if I talk about Telos here. And if you just try to have a look into this particular telos in both of the time, you will see that very similar. Not a lot of difference. Uh, if you look into Zolo here. Okay. So now this part, the horizontal is missing here. Little bit of that horizontal dot that you can see here. So when I am changing the values of the matrix simply I can see that right now I am focusing much more towards our vertical uh features or vertical edges right now.

Similarly, let me do a change. So you can see here. So in this minus one minus one. And right now in the mid column I have placed two okay. And in the last column also minus one. So right now when you look here now this is a little bit different okay. So like in this particular edge if I am talking about so here yes we do have our vertical edges for sure. But little bit of horizontal edges are also visible. Okay. So from this tells if I am talking about time. So little bit not completely but slightly. The horizontal edges are visible. So in this way you can actually build your own filters. Now this was the example for vertical edge detection using your own custom filters.

Similarly, like if I try to build a horizontal edge detector okay now you can see here. So let me just execute this piece of code. Okay. And right now, what I have done. So last time if you look into. So it was much more towards the vertical side everywhere it was minus one, minus one, minus one okay. And in the mid column. But this time we have just changed the alignment a little bit. So you can see here. So minus one okay. Minus one minus one. Now it's horizontally okay where we have placed it. And finally in the second row it's 222 okay. So the integer value that I am using basically. And in the last row again minus one minus one. So here when I have executed this so simply you can see that okay. So right now it's focusing on the horizontal features.

So if I talk about here the horizontal features in Telo's the horizontal feature. So it's actually lacking the vertical features okay. So in this way you do the feature extraction. So when I'm talking with respect to CNNs you don't use your custom filters rather than it's the job of the back propagation to find out those filter values. The idea is that it will be extracting different types of edges. Now for our second example, let's move to the next one. So right now. So minus one, minus one, minus one that I have taken in the first row. Then in my second row all are zeros. And finally in the third row that I have is 111. So right now when I try to do a render so using "cv2 dot filter 2d" I'm passing basically the image okay. And basically this kernel value that I have written. And finally when I am returning the image and you can see here. So here also I can see that it is actually focusing towards the horizontal features much more okay. Whereas you can see lesser focus towards the vertical features as simple as that.

So whenever you are trying to manipulate the values within a kernel, it will be responsible for a very specific type of feature extraction, you need to play with it. So like before, uh, like 2010, it was the job of the computer vision engineers to actually get these values. What will be the values of the kernel based on that? The feature extraction will take place, but after deep learning the entire process is taken care by back propagation. Okay. So similarly like another change so 11100 and then minus one a very similar example. If I talk about again this one will be focusing towards your horizontal features okay.

Now similarly like we have covered horizontal and vertical features. Now let's move to our next part which is basically looking into diagonal features right now. So if you look here also okay so some type of diagonal features like font of this I will say that yes it is there. So it's good. But how to focus much more okay. So this is basically the 45 degree or one. 35 degree diagonal angle detector. Okay. So right now if you look into this so minus one I have my minus one minus one two. Okay. And you can see that diagonally I have actually given this like 222 okay. Now this I have passed this values diagonally okay. And the rest like minus one minus one two okay.

Now see you need to actually play with these values. Then you will get a better idea in the next time I will change the diagonal position. Because right now if I talk about okay so this is in this direction okay. So 2 to 2. But I can make this into this direction also okay. So let me just execute this one. All right. And you can see that right now a lot of diagonal properties are basically the diagonal edges that you can simply see. So based on the kernel values that we have kept it. So right now it's actually focusing towards your diagonal detection. So as you can see here some edges here some edges okay. So like some horizontal and vertical edges that you might see but it's majorly focusing on the diagonal part much more okay.

So this time we have used 222 okay. And let's move with our second example of diagonal. So this time you can see the direction of two that I have changed. Now it's from this side to this side okay. Last it was from this side to this side. So let me just execute it right now. Okay. And right now as you can see here. So if you look here like this Zulu the Z the diagonal features is actually being shown. Then for us also like here little bit of curves are there that is visible. But for L if I'm talking about the vertical features are totally missing. If I'm talking about the horizontal features, it is missing. Similarly, if you try to look into some other word right now. Okay. So like every day. So I will say that yes, some type of diagonal features of that little part of the Y in every day. The last Y if you have a look okay. Then again you can understand. So it's not focusing completely on vertical and horizontal. But rather than it's focusing on the diagonal edges right now. And based on that it will it is extracting those type of features.

So it's the same thing that goes in CNN. In CNN you don't place the values of the kernel, but those values of the kernel is basically figured out by back propagation during our training process. And finally, once you get the best values for this kernels, you know that your loss will be in minimum. Okay. So based on that idea, I just tried to show you that you can even build your custom filters. But actually in deep learning you don't need to do that. Okay? It's completely taken care by. All right guys I hope you enjoyed this lecture. See you in the next video.


