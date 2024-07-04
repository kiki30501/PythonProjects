from PIL import Image
import numpy as np

# Open the image file
image = Image.open('low_res_star.jpg')

# Convert the image to grayscale
image = image.convert('L')

# Convert the image to a 1D array
array = np.array(image).flatten()

# Print the array
print(array)