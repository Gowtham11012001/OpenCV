import cv2
import numpy as np

def zoom_crop(image, scale):
    height, width, channels = image.shape
    cropped = image[scale:height-scale, scale:width-scale]
    resized_cropped = cv2.resize(cropped, (width,height))
    return resized_cropped

video = cv2.VideoCapture(0)
success, ref_img = video.read()
flag = 0

while(1):
    success, img = video.read()
    # print(flag)
    if(flag == 0):
        ref_img = img
        replaceImage = zoom_crop(ref_img,20)

    # Create a mask for background removal
    diff1 = cv2.subtract(img, ref_img)
    diff2 = cv2.subtract(ref_img, img)
    diff = diff1 + diff2
    diff[abs(diff)<13.0] = 0
    gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray[np.abs(gray)<10] = 0

    fgmask = gray.astype(np.uint8)
    fgmask[fgmask>0] = 255

    # Invert the mask
    fgmask_inv = cv2.bitwise_not(fgmask)

    # Use the fgmask and the inverted fgmask to extract the foreground and background parts

    fgimg = cv2.bitwise_and(replaceImage, replaceImage, mask = fgmask)
    bgimg = cv2.bitwise_and(ref_img,ref_img, mask = fgmask_inv)
    dst = cv2.add(bgimg,fgimg)

    # To get the live vedio in zoomed format 50 times
    # zoomed_img = zoom_crop(img, 50)
    # cv2.imshow("Zoomed", zoomed_img)

    cv2.imshow("Invisible OpenCV project", dst)
    key = cv2.waitKey(5) & 0xFF
    if ord('q') == key:
        break
    elif ord('d') == key:
        flag = 1
        print("Background Captured")
    elif ord('r') == key:
        flag = 0
        print("Ready to Capture")
cv2.destroyWindow()
video.release()
