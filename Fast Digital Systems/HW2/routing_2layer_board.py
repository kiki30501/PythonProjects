import numpy as np

# Create a 3D array of size 25x25x2
array_3d = np.zeros((25, 25, 2), dtype=np.uint8)

# Example: Set the value of an element at index (10, 10, 1) to 3 (binary: 11)
array_3d[10, 10, 1] = 3

# Example: Get the value of an element at index (10, 10, 1)
value = array_3d[10, 10, 1]

print(array_3d)
print(value)