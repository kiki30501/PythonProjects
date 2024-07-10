#################################################################################################################################
# Name:     Reshef Schachter, Eden Maman
# ID:       207701970,334055498
# Date:     08-07-2024
# Project:  HW2. Routing on a 2-layer board (PCB) using Lee's algorithm
#
# Dependencies: This project uses numpy, matplotlib and tqdm libraries. Make sure to install them before running the code.
#################################################################################################################################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plot_during_process = True
num_displays = 2

EMPTY    =  0
OCCUPIED = -1
num_origin = 5
num_destin = 5

# Declare global variables
board   = None
origins = None
destins = None
paths   = None
finalPaths = [None] * num_destin * num_origin

def initBoard(S, layer):
    global board, origins, destins
    # create a 25x25 board with 2 layers
    board = np.full((S, S, layer), EMPTY, dtype=int)
    # choose 10 unique random points on the board on the first layer but only between 1 and S-2 (in our case it's 23)
    points = np.random.choice(np.arange(1, S-2), size=(num_origin + num_destin, 2), replace=False)
    # split the generated points into origins and destinations    
    origins = points[:num_origin]
    destins = points[num_origin:]
    # expand the 2D points to 3D points by adding the layer index to be 0 (will be useful in the routing check operations)
    origins = np.hstack((origins, np.zeros((origins.shape[0], 1), dtype=int)))
    destins = np.hstack((destins, np.zeros((destins.shape[0], 1), dtype=int)))
    # assign the border of the board as occupied on all layers
    board[origins[:, 0], origins[:, 1], origins[:, 2]] = OCCUPIED
    board[destins[:, 0], destins[:, 1], destins[:, 2]] = OCCUPIED
    board[0,   :,   :] = OCCUPIED
    board[S-1, :,   :] = OCCUPIED
    board[:,   0,   :] = OCCUPIED
    board[:,   S-1, :] = OCCUPIED
    return


def plot_board_points():
    global board, origins, destins
    # plot the board with the origins and destinations (before the routing)
    plt.close('all')
    plt.figure(figsize=(8,8))
    plt.imshow(board[:, :, 0], cmap='gray', origin='lower')
    plt.xlim(-0.5, 24.5)
    plt.ylim(-0.5, 24.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(origins[:, 1], origins[:, 0], c='r', label='Origins')
    plt.scatter(destins[:, 1], destins[:, 0], c='g', label='Destinations')
    plt.xticks(range(-1, 26))
    plt.yticks(range(-1, 26))
    plt.minorticks_on()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


def plot_result(paths):
    # plot the board with the origins and destinations and the paths after routing
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c='r', label='Origins')
    ax.scatter(destins[:, 0], destins[:, 1], destins[:, 2], c='g', label='Destinations')
    colors = iter(cm.tab20b(np.linspace(0.0, 1.0, num_destin * num_origin)))
    for path in paths:
        if path is not None:
            path = np.array(path)
            c = next(colors)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=c, label='Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(-0.5, 24.5)
    ax.set_xticks(range(-1, 26))
    ax.set_yticks(range(-1, 26))
    #plt.legend()
    #ax.scatter(np.where(board == OCCUPIED)[0], np.where(board == OCCUPIED)[1], np.where(board == OCCUPIED)[2], c='k', label='Occupied')
    plt.show()
    return


def plot_progress(distance, curr, origins):
    plt.close('all')
    plt.figure(figsize=(8,8))
    plt.imshow(0.01 * distance[:, :, 0] , cmap='viridis' , origin='lower')
    plt.xlim(-0.5, 24.5)
    plt.ylim(-0.5, 24.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(origins[:, 1], origins[:, 0], c='r', label='Origins')
    plt.scatter(curr[1], curr[0], c='g', label='Destinations')
    plt.xticks(range(-1, 26))
    plt.yticks(range(-1, 26))
    plt.minorticks_on()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return

k=-1
j=-1
p=-1


def lee_algorithm():
    global board, origins, destins, paths, k,j,p
    flag=0
    # initialize the paths list
    paths = [None] * len(destins)
    
    # define the neighbors. The location of the element in the neighbors array is the direction as an integer
    neighbors = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, -1, 0], [-1, 0, 0], [0, 0, -1], [0, 0, 0]])
    r=-1
    # start the algorithm
    for destin in destins:
        for origin in origins:
            r+=1
            queue = np.array([origin])
            # initialize the distance matrix. The first value is the distance and the second value is the direction (based on the neighbors array)
            distance    = np.full(board.shape, np.inf)
            # Set the distance of the origins to 0, and keep the direction as None
            distance [origins[:,0], origins[:,1], origins[:,2]] = -1
            while queue.size > 0:
                k+=1
                print(f"k={k}")
                # get the current point from the queue
                curr = queue[0]
                queue = np.delete(queue, 0, axis=0)
                # check if the current point is a destination
                if np.all(destin == curr):
                    if plot_during_process == True and flag <= num_displays:
                        plot_progress(distance, curr, origins)
                        flag+=1
                    idx = np.where(destins == destin)[0][0]
                    paths[idx] = [curr]
                    while not np.all(curr == origin):
                        j+=1
                        print(f"j={j}")
                        for h, neighbor in enumerate(neighbors):
                            if curr[2] + neighbor[2] == board.shape[2] or curr[2] + neighbor[2] == -1:
                                continue
                            if distance[tuple(curr)] - 1 == distance[tuple(curr + neighbor)]:
                                direction = h
                                break
                        board[tuple(curr)] = OCCUPIED
                        curr = curr + neighbors[direction]
                        paths[idx].insert(0, curr)
                        #plot_result(paths)
                    finalPaths[r] = paths[idx]
                    break

                # check the neighbors
                for i, neighbor in enumerate(neighbors):
                    p+=1
                    print(f"p={p}")
                    new = curr + neighbor
                    if curr[2] + neighbor[2] == board.shape[2] or curr[2] + neighbor[2] == -1:
                        continue
                    if np.all(destin == new):
                        distance[tuple(new)] = distance[tuple(curr)] + 1
                        #board[tuple(curr)] = OCCUPIED
                        queue = np.vstack((queue, new))
                        continue
                    if board[tuple(new)] == EMPTY and distance[tuple(new)] == np.inf:
                        distance[tuple(new)] = distance[tuple(curr)] + 1
                        queue = np.vstack((queue, new))
    return paths


def main():
    initBoard(25, 2)
    plot_board_points()
    paths = lee_algorithm()
    print("hehe it's done! hooray!")
    plot_result(finalPaths)
    return

if __name__ == "__main__":
    main()