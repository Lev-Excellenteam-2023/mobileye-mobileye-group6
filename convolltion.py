import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from scipy import ndimage
from scipy import signal as sg
import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageShow
import matplotlib.pyplot as plt


def get_kernel_high(image, kernel_size=15, color_channel=0):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    np_img = np.array(img, dtype=np.float_)[:, :, color_channel]
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float_) / (kernel_size**2)
    high_pass_kernel = np_img - cv2.filter2D(np_img, -1, kernel)
    normalized_kernel = (high_pass_kernel - np.mean(high_pass_kernel)) / 255.0
    return normalized_kernel


def get_kernel_low(tmp_loc, color_ind):
    img = cv2.imread(tmp_loc)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (3, 3), interpolation= cv2.INTER_LINEAR)
    np_img = img[:, :, color_ind]
    np_img = np_img/255.0
    return np_img/np.sum(np_img)


def circle_image(mask, img_to_circle, color):
    h_circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 80,
                                 param1=2, param2=1, minRadius=1, maxRadius=10)
    if h_circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(h_circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            cv2.circle(img_to_circle, (x, y), r, color, 4)
            print(f"x: {x}, y: {y}, r: {r}")
    return img_to_circle


def color_adjacent_white_to_red(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    height, width, _ = hsv.shape

    lower_green = np.array([75, 140, 50])
    upper_green = np.array([90, 180, 100])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([20, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if mask_red[y, x] == 255:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if mask_white[y+i, x+j] == 255:
                            mask_red[y+i, x+j] = 255
            elif mask_green[y, x] == 255:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if mask_white[y + i, x + j] == 255:
                            mask_green[y + i, x + j] = 255

    return mask_red, mask_green


def identify_traffic_lights_with_hsv(image_path, temp_path1, temp_path2):
    img = Image.open(image_path)
    np_img = np.array(img)
    mask_red, mask_green = color_adjacent_white_to_red(np_img)
    print("Red x, y, r:")
    np_img = circle_image(mask_red, img_to_circle=np_img, color=(255, 0, 0))
    print("Green x, y, r:")
    np_img = circle_image(mask_green, img_to_circle=np_img, color=(0, 255, 0))
    return np_img


def get_kernel1():
    return np.array([[-1 / 9.0, -1 / 9.0, -1 / 9.0],
              [-1 / 9.0, 8 / 9.0, -1 / 9.0],
              [-1 / 9.0, -1 / 9.0, -1 / 9.0]], dtype=np.float_)


def get_kernel2():
    return np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]], dtype=np.float_)


def identify_traffic_lights_with_hps(im_path, temp_red, temp_green):
    im = Image.open(im_path)
    np_image = np.array(im)
    h, w, _ = np_image.shape
    data = np.array(im, dtype=np.float_)
    kernel_red = get_kernel_high(temp_red, 0)
    kernel_green = get_kernel_high(temp_green, 1)

    hps_red = ndimage.convolve(data[:, :, 0], kernel_red)
    threshold = 0.6  # Adjust this threshold based on your needs
    red_light_coord = np.argwhere(hps_red > threshold * hps_red.max())
    red_light_img = np.zeros_like(np_image[:, :, 0])
    red_light_img[red_light_coord[:, 0], red_light_coord[:, 1]] = 255
    red_light_img[0:50, :] = 0
    red_light_img[(h * 2) // 5:h, :] = 0

    hps_green = ndimage.convolve(data[:, :, 1], kernel_green)
    threshold = 0.8  # Adjust this threshold based on your needs
    green_light_coord = np.argwhere(hps_green > threshold * hps_green.max())
    green_light_img = np.zeros_like(np_image[:, :, 1])
    green_light_img[green_light_coord[:, 0], green_light_coord[:, 1]] = 255
    green_light_img[0:50,:] = 0
    green_light_img[(h*2)//5:h, :] = 0

    print("Red x, y, r:")
    np_image = circle_image(red_light_img, img_to_circle=np_image, color=(255, 0, 0))
    print("Green x, y, r:")
    np_image = circle_image(green_light_img, img_to_circle=np_image, color=(0, 255, 0))
    return np_image


def main():
    base_path = "C:/Users/user/PycharmProjects/pythonProject5/mobileye-mobileye-group6/images_set/"
    img_path = base_path + "aachen_000001_000019_leftImg8bit.png"
    temp_red = "red_light7.png"
    temp_green = "green_light4.png"
    img_np = identify_traffic_lights_with_hps(img_path, temp_red, temp_green)
    # img_np = identify_traffic_lights_with_hsv(img_path, temp_red2, temp_red3)
    # hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    plt.imshow(img_np)
    plt.show()


if __name__ == "__main__":
    main()


