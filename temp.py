import matplotlib.pyplot as plt
import mplcursors

# Read the image and rescale to 0-255 range if necessary
img = plt.imread("outputs_image_one_chanel/tl_grey_scale_green_img.png")
if img.max() <= 1.0:
    img = (img * 255).astype('uint8')

# Display the image
plt.imshow(img, cmap='gray') # Use cmap='gray' if it's a grayscale image
plt.axis('off')

# Enable mplcursors for the current axes
mplcursors.cursor(hover=True)

# Show the plot
plt.show()
