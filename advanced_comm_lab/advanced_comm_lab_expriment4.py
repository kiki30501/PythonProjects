
#####################################################################################
# Date:         4/7/24
# Last Update:  4/7/24
# Course:       Advanced Communication Lab
# Author:       Reshef Schachter, Boris Karasov
# Lecturer:     Irena Libster
# Project:      Advanced Communication Lab - Experiment 4
# Description:  Hamming(7,4) encoding and error correction on a black and white image
#####################################################################################

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

######################################
############# Question 1 #############
######################################

# Open the image file
image_in = Image.open('C:/Users/sixsi/Code-Projects/PythonProjects/advanced_comm_lab/low_res_star.jpg')

# Convert the image to grayscale
image_in = image_in.convert('L')

# Convert the image to a 1D array
array_in = np.array(image_in).flatten()

# Convert the array to 8 bit binary values
array_in = np.unpackbits(array_in.astype(np.uint8))

######################################
############# Question 2 #############
######################################

# Run a loop for every 4 bits, choose one randomly and flip it's value
array = array_in.copy()
for i in range(0, len(array), 4):
    random_index = np.random.randint(0, 4)
    array[i + random_index] = 1 - array[i + random_index]

# Convert the array back into an image
array_q2 = np.packbits(array)
array_q2 = array_q2.reshape(image_in.size[::-1])
image_q2 = Image.fromarray(array_q2)
image_q2.save('C:/Users/sixsi/Code-Projects/PythonProjects/advanced_comm_lab/low_res_star_q2.jpg')


# Plot the original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.imshow(image_in, cmap='gray')
plt.title('Original Image')
# Plot the modified image
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.imshow(image_q2, cmap='gray')
plt.title('Modified Image for Question 2')
plt.show()

######################################
############# Question 3 #############
######################################

# Add Hamming(7,4) encoding in the form of: m1,p1,m2,p2,m3,p3,m4
# p1 = m1 xor m2 xor m4
# p2 = m1 xor m3 xor m4
# p3 = m2 xor m3 xor m4
encoded_size = array_in.size * 7 // 4
array_hamming = np.zeros(encoded_size, dtype=np.uint8)
for i in range(0, array_in.size, 4):
    m1 = array_in[i]
    m2 = array_in[i + 1]
    m3 = array_in[i + 2]
    m4 = array_in[i + 3]
    p1 = m1 ^ m2 ^ m4
    p2 = m1 ^ m3 ^ m4
    p3 = m2 ^ m3 ^ m4
    array_hamming[i * 7 // 4] = m1
    array_hamming[i * 7 // 4 + 1] = p1
    array_hamming[i * 7 // 4 + 2] = m2
    array_hamming[i * 7 // 4 + 3] = p2
    array_hamming[i * 7 // 4 + 4] = m3
    array_hamming[i * 7 // 4 + 5] = p3
    array_hamming[i * 7 // 4 + 6] = m4

# Run a loop for every 7 bits, choose one randomly and flip it's value
array_hamming_noise = array_hamming.copy()
for i in range(0, len(array_hamming_noise), 7):
    random_index = np.random.randint(0, 7)
    array_hamming_noise[i + random_index] = 1 - array_hamming_noise[i + random_index]

# compare how many bits were flipped
flipped_bits = np.sum(array_hamming != array_hamming_noise)
print(f'Errors added: {flipped_bits}')

# Decode the Hamming(7,4) encoding in reversed order and correct the error, assign the corrected value to the decoded array
decoded_array = np.zeros(array.size, dtype=np.uint8)
for i in range(array_hamming_noise.size - 7, -1, -7):
    m1 = array_hamming_noise[i]
    p1 = array_hamming_noise[i + 1]
    m2 = array_hamming_noise[i + 2]
    p2 = array_hamming_noise[i + 3]
    m3 = array_hamming_noise[i + 4]
    p3 = array_hamming_noise[i + 5]
    m4 = array_hamming_noise[i + 6]
    if(i<10):
        print("flag")
    p1_check = m1 ^ m2 ^ m4 ^ p1
    p2_check = m1 ^ m3 ^ m4 ^ p2
    p3_check = m2 ^ m3 ^ m4 ^ p3
    error_bit = p1_check * 1 + p2_check * 2 + p3_check * 4
    if error_bit != 0:
        array_hamming_noise[i + error_bit - 1] = 1 - array_hamming_noise[i + error_bit - 1]
    decoded_array[i * 4 // 7] = m1
    decoded_array[i * 4 // 7 + 1] = m2
    decoded_array[i * 4 // 7 + 2] = m3
    decoded_array[i * 4 // 7 + 3] = m4

# Convert the decoded array back into an image
array_q3 = np.packbits(decoded_array)
array_q3 = array_q3.reshape(image_in.size[::-1])
image_q3 = Image.fromarray(array_q3)
image_q3.save('C:/Users/sixsi/Code-Projects/PythonProjects/advanced_comm_lab/low_res_star_q3.jpg')

flipped_bits = np.sum(array_in != decoded_array)
print(f'Errors after decoding: {flipped_bits}')

# Plot the original image
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
plt.imshow(image_in, cmap='gray')
plt.title('Original Image')
# Plot the noisy image
plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
plt.imshow(image_q3, cmap='gray')
plt.title('Modified Image for Question 3')
# Plot the decoded image
plt.subplot(1, 3, 3)  # 1 row, 3 columns, 2nd subplot
plt.imshow(image_q3, cmap='gray')
plt.title('Decoded & Corrected Image')
plt.show()

######################################
############# Question 4 #############
######################################