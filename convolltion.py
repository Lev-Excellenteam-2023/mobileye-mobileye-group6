import cv2
import numpy as np
from scipy import signal as sg
from PIL import Image, ImageDraw


def convert_to_green(image_path: str):
    """
    Convert an image to green by emphasizing the green channel.

    Args:
        image_path (str): Path to the input image.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Split the image into color channels
    blue, green, red = cv2.split(image)

    # Set the blue and red channels to zero, emphasizing the green channel
    blue = np.zeros_like(blue)
    red = np.zeros_like(red)

    # Merge the channels back together
    green_image = cv2.merge((blue, green, red))

    return green_image


def do_convolve(image_path, temp_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    input_array = np.array(image)
    height, width, channels = input_array.shape
    temp = Image.open(temp_path)
    resized_temp = temp.resize((3, 3))
    temp_np = np.array(resized_temp)
    temp_np = temp_np[:, :, 1]
    #temp_np = np.array([[96, 114, 148], [161, 165, 157], [127, 89, 66]])

    # Calculate the sum of all elements in the array
    #
    # mean = np.mean(temp_np)
    #
    # # Normalize the array by subtracting the mean
    # normalized_array = temp_np - mean
    sum_of_elements = np.sum(temp_np)

    # Normalize the array by dividing each element by the sum
    k = temp_np / sum_of_elements

    redness_score = sg.convolve(input_array[:, :, 1], k)
    # image = Image.fromarray(redness_score.astype('uint8'))
    # image.show()
    for y in range(height):
        for x in range(width):
            # Check if the redness score is above a threshold to consider it as a red region
            if redness_score[y, x] > 240:
                # Draw a black "X" mark at the current pixel
                draw.line([(x - 5, y - 5), (x + 5, y + 5)], fill="green", width=2)
                draw.line([(x - 5, y + 5), (x + 5, y - 5)], fill="green", width=2)
    return image


if __name__ == "__main__":
    base_path = "C:/Users/user/PycharmProjects/pythonProject5/mobileye-mobileye-group6/images_set/"
    img_path = base_path + "aachen_000011_000019_leftImg8bit.png"
    temp_img = "out_img23.png"
    result = do_convolve(img_path, temp_img)
    result.show()