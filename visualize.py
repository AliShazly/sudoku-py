import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from solve_puzzle import get_possibilities

with open('assets\\puzzles.json') as f:
    puzzles = json.load(f)


def puzzle_to_str(puzzle):
    str_puzzle = [[str(j) for j in i] for i in puzzle]
    str_puzzle_spaces = [[' ' if j == '0' else j for j in i] for i in str_puzzle]
    return str_puzzle_spaces


counter = 0  # Counts each iteration of the puzzle
base_puzzle_coords = []
frame_list = []
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('assets\\output.mp4', fourcc, 30.0, (504, 503))


def array_to_image(array, output):
    global counter
    global base_puzzle_coords
    global frame_list

    if not output:
        counter += 1
        return
    # Only grabbing every nth frame, depending on how many iterations the solve took. Speeds up execution greatly
    if not counter % frame_num == 0:
        counter += 1
        return
    # Grabbing the base numbers to draw in blue
    if counter == 0:
        for row, row_values in enumerate(array):
            for col, value in enumerate(row_values):
                if value != 0:
                    base_puzzle_coords.append((row, col))

    array = puzzle_to_str(array)
    coord_x = 12
    coord_y = 0
    step = 55.4
    image = Image.open('assets\\board.png')
    d = ImageDraw.Draw(image)
    for row, row_values in enumerate(array):
        for col, value in enumerate(row_values):
            fill = 0
            if (row, col) in base_puzzle_coords:
                fill = (0, 127, 255)
            fnt = ImageFont.truetype('assets\\FreeMono.ttf', 61)
            d.text((coord_x, coord_y), value, fill=fill, font=fnt)
            coord_x += step
        coord_x = 12
        coord_y += step
    # Converting from PIL to cv2 for output
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame_list.append(image)
    counter += 1
    print(f'Visualizing {round((counter / max_iterations) * 100, 2)}% finished...', end='\r', flush=True)
    return


def solve(puzzle, output):
    solved = True
    for row, row_values in enumerate(puzzle):
        for col, value in enumerate(row_values):
            if value == 0:
                solved = False
                break
        else:
            continue
        break
    if solved:
        return puzzle
    for i in range(1, 10):
        if i in get_possibilities(puzzle, row, col):
            puzzle[row][col] = i
            array_to_image(puzzle, output)
            if solve(puzzle, output):
                return puzzle
            else:
                puzzle[row][col] = 0
                array_to_image(puzzle, output)
    return False


puzzle_num = int(input('What puzzle do you want to visualize?: '))
print('Solving...', end='\r', flush=True)

solve(puzzles[puzzle_num], False)  # Solving puzzle to get amount of iterations first

frame_num = counter // 900  # counter / amount of total frames in the end video
# Only used for progress bar
max_iterations = counter
counter = 0

with open('assets\\puzzles.json') as f:  # Reloading the json because the puzzles var changed for some reason?
    puzzles = json.load(f)

solved = solve(puzzles[puzzle_num], True)  # Solving again to append frames to output

# Appending solved frames to the end of the animation
for i in range(frame_num * 80):
    array_to_image(solved, True)

print('\nSaving output...')
for i in frame_list:
    out.write(i)
out.release()
