import numpy as np
import cv2

#This function is used to detect and identify pixels that exceeds a given threshold.
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    #    1. An image and a desired threshold is given to the function.
    #    2. Useing numpy to create a 0 initialized array where the detected element will be marked on
    color_select = np.zeros_like(img[:,:,0])
    #    3. Iterate on the three channels of the given image comparing it to the threshold
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    #    4. If the given threshold is met, then it will be marked on the color_select which will be returned later
    color_select[above_thresh] = 1
    return color_select

#This function is used to Convert from image coords to rover coords
def rover_coords(binary_img):
    #    1. Take the binary image then extract the nonzero pixel from it
    ypos, xpos = binary_img.nonzero() 
    #    2. Subtracting the y coordinates of the image from the rovers y position then invert it
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)

    #    3. Subtracting half of the x coordinates of the image from the rovers x position then invert it
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    #    4. return the float values of both x and y coordinates.
    return x_pixel, y_pixel


# This function converts from cartesian coordinates to polar coordinates.
def to_polar_coords(x_pixel, y_pixel):
    #    1. It takes x and y coordinates
    #    2. It calculates the distance by taking the square root of the squared coordinates 
    dist = np.sqrt(x_pixel*2 + y_pixel*2)
    #    3. It takes the arctan of the coordinates to calculate the angles
    angles = np.arctan2(y_pixel, x_pixel)
    #    4. Then it returns both destinations and angles
    return dist, angles

# This function is used to map the rover space to the world space
def rotate_pix(xpix, ypix, yaw):
    #   1. The function takes x,y and yaw axis as a parameters
    #   2. Then converts the yaw into radiant
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    #   3. Then rotates the x and y coordinates by using the converted yaw radiant
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    #   4. Then it returns the rotated coordinates
    return xpix_rotated, ypix_rotated

#The function applies both translation and scaling on any given coordinates
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    #   1. X and y coordinates, the amount of translation in x and y coordinates, and the scaling factor is a parameters of the function
    #   2. The scaling is a division/multiplication operation, and the translation is plus/minus operations.
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    #   3. Then it returns the translated and scaled coordinates.
    return xpix_translated, ypix_translated

#This function applies different geometric transformations to output the final world map image. It also ties the previous functions together
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    #   1. First it rotates the x and y coordinates
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)

    #   2. Then it translates and scale them using the given value
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    #   3. Then at the end it clip the unwanted values to only have the wanted values
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    #   4. It returns the final x and y coordinates for the map image
    return x_pix_world, y_pix_world

#This function creates the perepective transformation
def perspect_transform(img, src, dst):
    #   1. The function takes the image, coordinates in the source image, and coordinates in the output image
    #   2. Then it generates a transformation matrix and store it at M
    M = cv2.getPerspectiveTransform(src, dst)
    #   3. Then it uses the transformation matrix with the image to apply the perspective transformation to the image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    #   4. Then we create a mask which will have a values of 1s for navigable pixels and 0s for non-navigable pixels
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]),M, (img.shape[1], img.shape[0]))
    #   5. Then return both the mask and the transformed image
    return warped , mask

#This function uses thresholding to identify the rocks but with minor modifications
def find_rocks(img, thresh = (110,110,50)):
    #   1. It takes the image and the wanted threshold
    #   2. It compares the image three channels to the given threshold and store the values to rock_pixels
    rock_pixels = ((img[:,:,0]>thresh[0])\
                  &(img[:,:,1]>thresh[1])\
                  &(img[:,:,2]<thresh[2]))
    #   3. A zero array is generated
    colored_pixels = np.zeros_like(img[:,:,0])
    #   4. The identified pixels that met the conditions will be used as an index for the zero array generated and every found rock will be equal to 1
    colored_pixels[rock_pixels] = 1
    #   5. The zero array is returned
    return colored_pixels

def impose_range(xpix, ypix, range=80*2):
    dist = np.sqrt(xpix*2 + ypix*2)
    return xpix[dist < range], ypix[dist < range]

def trim_ellipse(image):
    # create a mask image of the same shape as input image, filled with 0s (black color)
    mask = np.zeros_like(image)
    rows, cols, _ = mask.shape
    # create a white filled ellipse
    mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows)-5), axes=(int(cols / 2), 95), angle=0,
                       startAngle=180, endAngle=360, color=(255, 255, 255), thickness=-1)
    # Bitwise AND operation to black out regions outside the mask
    return np.bitwise_and(image, mask)

# This function is the function that tie the previous functions together to create a better perception of the world and achieve the objectives we want
def perception_step(Rover):
    #1. Define the dst_size which is the destination size
    dst_size = 10 
    #2. Define bottom_offset which is just an offset the gives us a buffer by moving the position 6 units to the front
    bottom_offset = 5
    #3. Define the image, source, and destinations points for the perspective transformation
    image = Rover.img
    #image = cv2.medianBlur(image,11)
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                    [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                    [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                    [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                    ])
    #4. It will generate a perspective transformation and return the transformed image and the mask
    warped, mask = perspect_transform(image, source, destination)
    warped = trim_ellipse(warped)
    #5. Then it will apply color thresholding to identify navigable terrain, non-navigable terrain and rocks
    threshed = color_thresh(warped)
    #6. Then it will generate obstacle maps in obs_map by multiplying the mask with threshold -1 so we get only the things in the field of view
    obs_map = np.absolute(np.float32(threshed) - 1)*mask
    #7. Using the threshold and the obstacle map we modify the vision image for both the obstacles and the navigable terrain
    Rover.vision_image[:,:,2] = threshed *255
    Rover.vision_image[:,:,0] = obs_map *255
    #8. Then It will convert the image coordinates to rover coordinates
    xpix, ypix = rover_coords(threshed)
    dist, angles = to_polar_coords(xpix, ypix)
    Rover.nav_dists = dist
    Rover.nav_angles = angles
    mean_dir = np.mean(angles)
    #9. It will define variables related to the scale and world size which will be used for the next step
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    x_world, y_world = pix_to_world(xpix,ypix, Rover.pos[0],Rover.pos[1], Rover.yaw, world_size, scale)
    #10. Then it will convert the rover coordinates to world coordinates for both the obstacles and the walkable world
    obsxpix, obsypix = rover_coords(obs_map)
    obs_x_world, obs_y_world = pix_to_world(obsxpix,obsypix,Rover.pos[0],Rover.pos[1],Rover.yaw,world_size,scale)
    #11. An updated rover worldmap is generated to be on the right where steps 12 and 13 happens taking into concideration pitch and roll
    #12. The navigable terrain becomes blue given the world coordinates
    #13. The obstacles become red given the obstacles coordinates
    #if(Rover.pitch< 1.6) and (Rover.roll<5):
    if (Rover.pitch < 1 or Rover.pitch > 359) and (Rover.roll < 1 or Rover.roll > 359):
        Rover.worldmap[y_world,x_world,2]+=10
        Rover.worldmap[obs_y_world,obs_x_world,0]+=1
        nav_pix = Rover.worldmap[:, :, 2] > 0
        Rover.worldmap[nav_pix, 0] = 0
            # clip to avoid overflow
        Rover.worldmap = np.clip(Rover.worldmap, 0, 255)
    #14. Then the world is converted from rover coordinates to polar coordinates
    dist, angles = to_polar_coords(xpix,ypix)
    Rover.nav_angles = angles
    #15. The find_rocks function is used to find the rocks
    rock_map = find_rocks(warped,(110,110,50))
    #16.  If rocks are identified, then it will convert the rock position to rover coordinates then to world coordinates then to polar coordinate
    #17. Then color the rock with white on the map
    if rock_map.any():
        rock_xpix , rock_ypix = rover_coords(rock_map)
        rock_xpix_world , rock_ypix_world =  pix_to_world(rock_xpix, rock_ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
        rock_dist,rock_ang = to_polar_coords(rock_xpix,rock_ypix)
        Rover.samples_dists = rock_dist
        Rover.samples_angles = rock_ang
        rock_idx = np.argmin(rock_dist)
        rock_xcen = rock_xpix_world[rock_idx]
        rock_ycen = rock_ypix_world[rock_idx]
        Rover.worldmap[rock_ycen, rock_xcen,1] = 255 # colour the rock with white on the map
        Rover.vision_image[:,:,1] = rock_map *255
    else:
        Rover.vision_image[:,:,1]=0
        Rover.samples_dists = None
        Rover.samples_angles = None

   

    #18.Show the pipeline of our processes
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('camera',image2)
    warped2 = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    cv2.imshow('precpective transform',warped2)
    mask2 = mask*255
    cv2.imshow('precpective mask',mask2)
    cv2.imshow('obstacle',obs_map)
    threshed2 = threshed * 255
    cv2.imshow('thershold',threshed2)
    rock_map2 = rock_map*255
    cv2.imshow('rock',rock_map2)
    arrow_length=100
    x_arrow = arrow_length * np.cos(mean_dir)
    y_arrow = arrow_length * np.sin(mean_dir)
    if( (x_arrow == x_arrow) and  (y_arrow == y_arrow) ):
        color = (0, 0, 255)    
        thickness = 2
        view = image = cv2.rotate(threshed2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        start_point = (int(view.shape[1]),int(view.shape[0]/2))
        end_point=(int(x_arrow),int(y_arrow)+int(view.shape[0]/2))
        direction = cv2.arrowedLine(view,start_point, end_point, color,thickness)
        cv2.imshow('Direction', direction)

    
    cv2.waitKey(5)

    return Rover