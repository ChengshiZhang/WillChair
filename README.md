# Willchair

Many wheelchair users face difficulties opening doors and operating elevators. The goal of this project was to build a smart wheelchair with a depth-sensing camera, a tablet running a user-interface, and a robotic arm that opens the doors and operates elevator buttons at the user’s will. 

The system uses image processing techniques to find the doorknob and open the door using the robotic arm, so that the user would no longer need to reach for the door. The integrated system operates under different weather conditions independent of a Wi-Fi connection, and functions without an AC power source up to a day.

## Computer Vision Algorithm
### Initial Approach
To implement the algorithm, we first took sample pictures in controlled environments.  

In the pictures we took, we had the camera close to the door so that the door occupies the majority of the screen, and the camera is on the right side of the door. With these sample inputs, we can safely assume that the length of the door object is the longests among other objects in the image. Thus, we looped through every single pixel in the entire image horizontally. In every row of pixels, we find the length of the longest object among all the objects appeared in the same row. As we move to the next pixel (the pixel to the right of the current one), if the the difference of the RGB value of the current pixel and the RGB value of the previous pixel (the pixel to the left) is above a certain threshold, we will consider these two pixels to belong to two different objects; if the difference is below the threshold, we will consider them to be within the same object. This way, we are able to find the longest possible length on every row of the image.

This way, we are able to sketch a rough shape for the door in the image. Now, the objective is to use a bounding box to cut the door from the image and mask everything else in the image so that we can focus on the door itself. To find the two points in the x-axis of the door’s bounding box, we put all the x-axis values of the white pixels into a histogram. Using this history, we can see where most white pixels stack up vertically. Therefore, we will use both local maximas as the x-coordinates of the bounding box. The y-coordinates will simply be the top and bottom pixels of the image, since we expect most doors to occupy the entire y-axis on our sample input images.

Using the bounding box, we are able to erase everything else on the image so that we can focus on the door. The doorknob will be much easier to find now that the only object left in the image is the door itself. Since we assume the door to have a uniform color and the doorknob to have a distinct color from the door itself, we applied adaptive thresholding to find the object on the door that does not have a similar color to the door. We chose the adaptive thresholding method since we need to deal with direct light source from random angles in the sample image, and other segmentation method are not as optimal when applied against most lighting conditions. After applying the adaptive thresholding method, we are able to successfully find the objects on the door, including the doorknob.

Then, using the position of the objects we found, we can easily distinguish the doorknob and other objects on the door. For example, in Figure 1.3, you can see that the segmented image has two large white components: the doorframe on the bottom left, and the doorknob on the right side. The one of the bottom left is obviously below the height of a regular doorknob, so we ignore it. The only qualified object in the image is the doorknob, so we simply put a red box around it.

Thus, we have managed to successfully find the doorknob on our sample images.

### Revision of Initial Approach
When we tested our algorithm against more low-resolution pictures, the result is not very optimal. The reason is that the adaptive thresholding method does not work well against low-resolution images. To apply adaptive thresholding, we have to cut off the outer frame of the target area in the image, since we need to apply the kernel to every pixel and the pixels on the edge will not be included. Since the doorknob is very close to the edge of the door, when the kernel has a large size, we will cut off a portion of the doorknob, thus messing up our algorithm. The reason we had some low-resolution sample images is because of the improved run-time. Since our current algorithm does not work well with these images, we do not have a choice be sticking with high resolution images and sacrifice performance.

Also, the program does not work well with some images where there are shadows on the door, either from the doorknob itself or other objects. Since we are using RGB values when we find the door’s bounding box and apply adaptive thresholding, these shadows will severely affect the result. For example, suppose that there are shadows of the doorknob displayed on the door. The door color within and outside the shadow are drastically different, and will be treated as if they belong to different object when we apply the adaptive thresholding method. However, even though the RGB values are very different, these two regions will have similar hue values, since hue values are obtained independently from the saturation and lighting values. Thus, we changed all the instances where we used RGB values to using hue values instead, like the adaptive thresholding and finding the door’s bounding box. This way, our program is much more robust and and handles many more sample images than before.

### Final Approach
Even though our first approach has been a success so far, there are still some problems remaining to be solved:

1. When the camera is not as close to the door, the program fails to recognize the door’s bounding box. This is because we assumed the door to have a longest width among other objects in the image.
2. When there are other objects on the door, like posters or small glass windows, the program has a hard time figuring out which object is the actual doorknob.
3. The bounding box we found is not as accurate since the door is not a perfect rectangle in the image. When we used the rectangle as the bounding box of the door, we almost always include a portion of the doorframe or cut off a portion of the doorknob.
4. The doorknobs on the left side of the door is not handled due to time constraints.

We are determined to make the program robust, and we decided to rethink how we should approach the problem. We start with the only clues we have: the basic assumptions. When we read an input image, these are the fundamental things we can assume:
1. The door color is uniform
2. The door handle has a distinct color from the door color
3. The door has the largest area compared to other objects in the image

With these assumptions, we do not have a choice but to scrape the original algorithm we implemented. The original assumption we had, assuming the door object to have the longest width in the image, is too bold and will not work when there are objects on the door besides the doorknob. Thus, we need to find a new way to locate the door in the image before we try to find the doorknob.

In order to locate the door in the input image, we decided to use edge detection to find all possible edges in the input image, and then flood fill every closed region and find the largest possible area. Based on our fundamental assumption, we can assume the largest area found is the region of the door. The biggest issue with this approach is the fact that the edge detection algorithm might not be able to find the edges of the door well enough so that the door region is a closed region. If one of the door edges has an opening, then the flood fill algorithm will include regions outside the door as part of the door as well.

We first used regular edge detection algorithm with RGB values. When the difference of RGB values between two pixels is above a certain threshold, then we will mark the pixel on the right as white color. The result we obtained is not optimized, as there are many openings in the door edges. We attempted to use hue values instead of RGB values with our edge detection algorithm. However, even though there are improvements, the result is still not good enough for us to flood fill. We have to further optimize our edge detection algorithm.

Then, we implemented the Sobel-Feldman edge detection filter, as it has a kernel better suited to detect edges for our sample images. The result we obtained with this algorithm have much more optimized door edges that we can use for the flood fill process. We tried to use hue values instead of RGB values for the Sobel edge detection algorithm, but the result seems similar and there isn’t any obvious improvements. Since the hue values takes time to compute, we decided to stick with the default Sobel edge detection algorithm without modifications. After flood filling all the closed regions in the image, we are able to find all the closed regions in the image. Then, we remove all the closed regions and only leave the largest one in the image, which is the door itself. This way, we are able to find the region of the door correctly and accurately, even if there are random obstacles or objects on the door.

Now that we have a segmented image with only the door’s region remaining, we can now use template matching to find the shadow on the door that closely resembles a doorknob. We have 6 templates total, with 3 of them being the left-side doorknob on the door with different sizes and 3 of them being the right-side ones with different sizes. We chose to only have 3 layers for each template because of performance concerns.

Then, after we find the rough region of the doorknob, we will find the first black pixel of the found region, and flood fill the rest of the black regions near it with red color. This way, we can find the precise shape of the doorknob and obtain very accurate results. On top of that, we also flood fill the regions near the first black pixel where the hue value is different from the average hue value of the door itself. This optimization is added because some of the doorknobs are made of steel and have reflections. Sometimes, the edge detection algorithm has a hard time recognizing the correct edges around the doorknob, and the result from the flood fill algorithm with include a portion of the doorknob as part of the door. Therefore, we added all nearby pixels with a distinct hue value than the door itself, and include them as part of our doorknob. This way, we are able to accurately locate almost all doorknobs in our sample images despite any lighting condition, camera angle, and random objects or glass windows on the door.

## Acknowledgements
This project was supported and advised by Professor Alan Pisano and Osama AlShaykh from Boston University.
