# Computer_Vision_Skills
pip install opencv-python

pip install ipykernel

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

