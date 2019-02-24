from PIL import Image, ImageDraw, ImageFont
import json
from sudoku import get_row, get_column, get_possibilities

with open('assets\\puzzles.json') as f:
    puzzles = json.load(f)

def puzzle_to_str(puzzle):
    str_puzzle = [[str(j) for j in i] for i in puzzle]
    str_puzzle_spaces = [[' ' if j == '0' else j for j in i] for i in str_puzzle]
    return str_puzzle_spaces

loop_num = 0
base_puzzle_coords = []
frame_list = []
def array_to_image(array):
    global loop_num
    global base_puzzle_coords
    global frame_list
    if loop_num == 0:
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
            fill = (0)
            if (row, col) in base_puzzle_coords:
                fill = (0,127,255)
            fnt = ImageFont.truetype('assets\\FreeMono.ttf', 61)
            d.text((coord_x, coord_y), value, fill=fill, font= fnt)
            coord_x += step
        coord_x = 12
        coord_y += step
    # image.save(f'frames\\{loop_num}.png')
    frame_list.append(image)
    loop_num+=1
    if loop_num % 100 == 0:
        print(f'Processed {loop_num} frames')
    return

def solve(puzzle):
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
            array_to_image(puzzle)
            if solve(puzzle):
                return puzzle
            else:
                puzzle[row][col] = 0
                array_to_image(puzzle)
    return False


solved = solve(puzzles[0])
# Appending solved frames to the end of the animation
for i in range(40):
    array_to_image(solved)

frame_list[0].save('assets\\output.gif',
               save_all=True,
               append_images=frame_list[1::20],
               duration=16.67,
               loop=0,
               optimize=True)