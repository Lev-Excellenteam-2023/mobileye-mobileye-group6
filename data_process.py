from PIL import Image, ImageDraw, ImageFont
import json
from typing import List, Tuple


def put_text(image_path: str, x: int, y: int, text: str) -> None:
    """
    Put text on an image and save the result.

    Args:
        image_path (str): Path to the input image.
        x (int): X-coordinate of the text.
        y (int): Y-coordinate of the text.
        text (str): Text to be added.
    """
    # Open the image using PIL
    image = Image.open(image_path)
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    # Choose a font and size
    font_size = 16  # Adjust the font size as needed
    font = ImageFont.truetype("arial.ttf", font_size)
    # Draw the text on the image
    draw.text((x, y), text, font=font, fill=(0, 255, 0))
    image.show()


def find_min_max_point(points: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """
    Find the minimum and maximum points from a list of (width, height) points.

    Args:
        points (List[Tuple[int, int]]): List of (width, height) points.

    Returns:
        Tuple[int, int, int, int]: Minimum and maximum height and width values.
    """
    max_w, max_h = 0, 0
    min_w, min_h = float('inf'), float('inf')

    for w, h in points:
        if h > max_h:
            max_h = h
        if h < min_h:
            min_h = h
        if w > max_w:
            max_w = w
        if w < min_w:
            min_w = w

    return min_h, max_h, min_w, max_w


def crop_image(points: List[Tuple[int, int]], img_path: str, output_path: str):
    """
    Crop an image based on a list of points and save the cropped image.

    Args:
        points (List[Tuple[int, int]]): List of (x, y) points representing the polygon.
        img_path (str): Path to the input image.
        output_path (str): Path to save the cropped image.

    Returns:
        None
    """
    img = Image.open(img_path)
    min_h, max_h, min_w, max_w = find_min_max_point(points)
    offset = 0  # Adjust this offset as needed
    cropped_img = img.crop((min_w - offset, min_h - offset, max_w + offset, max_h + offset))
    cropped_img.save(output_path)


def get_json_points(json_path: str) -> List[List[Tuple[int, int]]]:
    """
    Extracts polygon points from a JSON file containing object information.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        List[List[Tuple[int, int]]]: List of polygons, where each polygon is represented as a list of (x, y) points.
    """
    lst_points = []
    with open(json_path, 'r') as file:
        json_data = json.load(file)
        for obj in json_data["objects"]:
            if obj["label"] == "traffic light":
                lst_points.append(obj["polygon"])
    return lst_points


def main():
    base_path = "C:/Users/user/PycharmProjects/pythonProject5/mobileye-mobileye-group6/images_set/"
    img_path = base_path + "aachen_000011_000019_leftImg8bit.png"
    json_path = base_path + "aachen_000011_000019_gtFine_polygons.json"
    lst_points = get_json_points(json_path=json_path)
    for i in range(len(lst_points)):
        crop_image(lst_points[i], img_path=img_path, output_path=f"out_img{i + 13}.png")

    print(lst_points)
    min_h, max_h, min_w, max_w = find_min_max_point(lst_points[0])
    put_text(img_path, (min_w + max_w) // 2, (min_h + max_h) // 2, "X")


if __name__ == "__main__":
    main()
