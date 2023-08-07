import time
from scipy.ndimage import maximum_filter
import cv2
import numpy as np
from scipy import signal as sg
from PIL import Image, ImageDraw, ImageFont


def identify_red_traffic_lights(image_path, template_path1):
    # Open the input image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Convert image to numpy array
    input_array = np.array(image)
    height, width, _ = input_array.shape
    #
    template = Image.open(template_path1)
    resized_temp = template.resize((3, 3))
    temp_np = np.array(resized_temp) / 255.0
    temp_np = temp_np[:, :, 0]
    k_red = temp_np / np.sum(temp_np)

    # Apply convolution to each channel
    #k = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
    redness_score_r = sg.convolve(input_array[:, :, 0], k_red)
    redness_score_g = sg.convolve(input_array[:, :, 1], k_red)
    redness_score_b = sg.convolve(input_array[:, :, 2], k_red)

    threshold_r = 175  # Adjust the threshold as needed
    threshold_g = 190
    list_point = []
    for y in range(height):
        for x in range(width):
            if redness_score_r[y, x] > threshold_r and redness_score_r[y, x] >\
                    (redness_score_g[y, x] + redness_score_b[y, x]) * 0.65:
                for i in range(len(list_point)):
                    if x - 10 <= list_point[i][0] <= x + 10 and y - 10 <= list_point[i][1] <= y + 10:
                        if list_point[i][2] < redness_score_r[y, x]:
                            list_point[i][0] = x
                            list_point[i][1] = y
                            list_point[i][2] = redness_score_r[y, x]
                            break
                        else:
                            break
                else:
                    list_point.append([x, y, redness_score_r[y, x]])
                #draw.rectangle([(x - 2, y - 2), (x + 2, y + 2)], outline="red")
            elif redness_score_g[y, x] > threshold_g and redness_score_g[y, x] > \
                    (redness_score_r[y, x] + redness_score_b[y, x]) * 0.5:
                None
                #print(f"g: {redness_score_g[y, x]} r: {redness_score_r[y, x]} b: {redness_score_b[y, x]}")
                #draw.rectangle([(x - 2, y - 2), (x + 2, y + 2)], outline="green")

    for [x, y, _] in list_point:
        draw.rectangle([(x - 2, y - 2), (x + 2, y + 2)], outline="red")
    print(len(list_point))
    return image


if __name__ == "__main__":
    base_path = "C:/Users/user/PycharmProjects/pythonProject5/mobileye-mobileye-group6/images_set/"
    img_path = base_path + "aachen_000002_000019_leftImg8bit.png"
    temp_red = "red_light2.png"
    temp_green= "green_light1.png"
    result = identify_red_traffic_lights("red_to_check2.jpeg", temp_red)
    result.show()
