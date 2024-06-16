#################################################################################################################################
# Name:     Reshef Schachter
# ID:       207701970
# Date:     11-06-2024
# Project:  HW2. Routing on a 2-layer board (PCB) using Lee's algorithm
#
# Dependencies: This project uses numpy, matplotlib and tqdm libraries. Make sure to install them before running the code.
#################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures as cf
import threading
from tqdm import tqdm

# I'm using the defenition of distance where straight lines and diagonals are both distance 1
search_pattern = [[0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 0, 1]]
num_origin = 5 # Number of origins
num_destin = 5 # Number of destinations
S = 25 # Size: Length and width of the board
L = 2  # Depth (layers). This is always 2. I added it here for clarity
finish_found = threading.Event() # define a threading event to signal that the finish is found (helps when using concurrent execution)

def create_board():
    # Define the board variable and the points of interest
    global board, origins, destins, L, S
    # Create a 3D array of size 25x25x2 initialized with zeros
    board  = np.zeros((S, S, L), dtype=np.uint8)
    # choose 10 unique random points on the board on the first layer but only between 1 and S-2 (in our case it's 23)
    points = np.random.choice(np.arange(1, S-2), size=(num_origin + num_destin, 2), replace=False)
    # split the generated points into origins and destinations
    origins = points[:num_origin]
    destins = points[num_origin:]
    # expand the 2D points to 3D points by adding the layer index to be 0 (will be useful in the routing check operations)
    origins = np.hstack((origins, np.ones((num_origin, 1), dtype=np.uint8)))
    destins = np.hstack((destins, np.ones((num_destin, 1), dtype=np.uint8)))
    # assign the border of the board as occupied on all layers
    board[origins[:, 0], origins[:, 1], 1] = 1
    board[destins[:, 0], destins[:, 1], 1] = 1
    board[0,   :,   :] = 1
    board[S-1, :,   :] = 1
    board[:,   0,   :] = 1
    board[:,   S-1, :] = 1
    return

def plot_board():
    # plot the board with the origins and destinations (before the routing)
    plt.close('all')
    #plt.figure(figsize=(8,8))
    #plt.imshow(board[:, :, 1])
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
    return

# def plot_result():
#     # plot the board with the origins and destinations and the paths after routing
#     plt.close('all')
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c='r', label='Origins')
#     ax.scatter(destins[:, 0], destins[:, 1], destins[:, 2], c='g', label='Destinations')
#     for path in paths:
#         if path is not None:
#             path = np.array(path)
#             ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()
#     return

def plot_result():
    # plot the board with the origins and destinations and the paths after routing
    plt.close('all')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c='r', label='Origins')
    ax.scatter(destins[:, 0], destins[:, 1], destins[:, 2], c='g', label='Destinations')
    for i, path in enumerate(paths):
        if path is not None:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], label=f'Path {i+1}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()  # Add this line to ensure the legend is displayed
    plt.show()
    return

def is_occupied(point, wave):
    # Check if a point is occupied on the board
    return board[point] == 1

def not_in_bounds(point):
    # Check if a point is out of bounds
    return point[0] < 0 or point[0] >= S or point[1] < 0 or point[1] >= S

def wave_search(wave, point, distance, finish):
    # A recursive function to expand the wave and search for the shortest path to the finish
    if point == finish:
        finish_found.set()
        return (point,)
    if not_in_bounds(point):
        return None
    if is_occupied(point, wave) and distance != 0:
        return None
    wave[point] = distance
    with cf.ThreadPoolExecutor() as executor:
        futures = []
        for pattern in search_pattern:
            futures.append(executor.submit(wave_search, wave, ((point[0]+pattern[0]), (point[1]+pattern[1]), point[2]), distance + 1, finish))
            futures.append(executor.submit(wave_search, wave, ((point[0]), (point[1]), point[2] ^ 1), distance + 1, finish))
        for future in cf.as_completed(futures):
            result = future.result()
            if result is not None:
                board[point] = 1
                return (point,) + result
        # future = 
        # result = future.result()
        # if result is not None:
        #     board[point] = 1
        #     return (point,) + result
    return None



# def wave_search(wave, start, finish):
#     queue = deque([start])  # Initialize the queue with the start point
#     paths = {start: [start]}  # Dictionary to save the paths

#     while queue:
#         point = queue.popleft()  # Pop the point from the front of the queue

#         if point == finish:
#             return paths[point]  # Return the shortest path to the finish point

#         if not_in_bounds(point) or is_occupied(point, wave):
#             continue

#         for pattern in search_pattern:
#             next_point = ((point[0]+pattern[0]), (point[1]+pattern[1]), point[2])
#             if next_point not in paths:  # If the next point is not visited
#                 queue.append(next_point)
#                 paths[next_point] = paths[point] + [next_point]  # Save the path to the next point

#         next_point = ((point[0]), (point[1]), point[2] ^ 1)
#         if next_point not in paths:  # If the next point is not visited
#             queue.append(next_point)
#             paths[next_point] = paths[point] + [next_point]  # Save the path to the next point

#     return None
# from collections import deque

# def wave_search(wave, start, finish):
#     queue = deque([start])  # Initialize the queue with the start point
#     wave[start] = 0  # Assign the distance of the start point as 0

#     while queue:
#         point = queue.popleft()  # Pop the point from the front of the queue

#         if point == finish:
#             return wave[point]  # Return the distance to the finish point

#         if (not_in_bounds(point) or is_occupied(point, wave)) and wave[point] != 0:
#             continue

#         for pattern in search_pattern:
#             next_point = ((point[0]+pattern[0]), (point[1]+pattern[1]), point[2]^pattern[2])
#             if not_in_bounds(next_point) or is_occupied(next_point, wave):
#                 continue
#             if wave[next_point] == -1:  # If the next point has not been visited
#                 queue.append(next_point)
#                 wave[next_point] = wave[point] + 1  # Assign the distance of the next point

#     return None

def start_search():
    # A function to start the search for all origins to all destinations
    global paths
    paths = []  # Define "paths" as an empty array before the loop
    for i in tqdm(range(num_origin * num_destin), desc="Executing Routing Algorithm", unit="path"):
        start  = tuple(origins[i // num_destin])
        finish = tuple(destins[i %  num_destin])
        # wave   = np.zeros((25, 25, 2), dtype=np.uint8)
        wave = np.full((S,S,L), -1, dtype=np.int8)
        finish_found.clear()
        # paths.append(wave_search(wave, start, 0, finish))  # Append the result of wave_search to "paths"
        paths.append(wave_search(wave, start, finish))  # Append the result of wave_search to "paths"
    missing_paths = [i for i, path in enumerate(paths) if path is None]
    print(f"Number of missing paths: {len(missing_paths)} Out of {num_origin * num_destin} paths")
    if input("Do you want to display the missing paths? (y/n): ").lower() == "y":
        for i in missing_paths:
            print("No path found between: Origin: ", origins[i // num_destin], " Destination: ", destins[i % num_destin])
    return

def main():
    create_board()
    display_board = input("Display board graph before routing? (y/n): ")
    if display_board.lower() == "y":
        plot_board()
    start_search()
    display_result = input("Display result graph? (y/n): ")
    if display_result.lower() == "y":
        plot_result()
    return

main()