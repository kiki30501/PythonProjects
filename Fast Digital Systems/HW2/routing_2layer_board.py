import numpy as np
import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')

# Create a 3D array of size 25x25x2 initialized with zeros
board  = np.zeros((25, 25, 2), dtype=np.uint8)

# choose 10 unique random points on the board on the first layer but only between 1 and 23
points = np.random.choice(np.arange(1, 23), size=(10, 2), replace=False)

# assign 5 points to the origins and 5 points to the destinations
origins_2d = points[:5]
destins_2d = points[5:]

# expand the 2D points to 3D points by adding the layer index to be 0 (will be useful in the routing check operations)
origins    = np.hstack((origins_2d, np.zeros((5, 1), dtype=np.uint8)))
destins    = np.hstack((destins_2d, np.zeros((5, 1), dtype=np.uint8)))

# assign the origins and destinations to the board as 1
board[origins[:, 0], origins[:, 1], 0] = 1
board[destins[:, 0], destins[:, 1], 0] = 1
board[0,  :,  :] = 1
board[24, :,  :] = 1
board[:,  0,  :] = 1
board[:,  24, :] = 1

# visualize layer 0 of the board
plt.figure(figsize=(8,8))
plt.imshow(board[:, :, 0])
plt.xlim(-0.5, 24.5)
plt.ylim(-0.5, 24.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(origins[:, 1], origins[:, 0], c='r', label='Origins')
plt.scatter(destins[:, 1], destins[:, 0], c='g', label='Destinations')
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()


print("Origins:\n", origins, "\n")
print("Destinations:\n", destins, "\n")
print("Points:\n", points)


