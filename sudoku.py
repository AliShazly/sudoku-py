puzzle_easy = [[5,3,0,0,7,0,0,0,0],
          [6,0,0,1,9,5,0,0,0],
          [0,9,8,0,0,0,0,6,0],
          [8,0,0,0,6,0,0,0,3],
          [4,0,0,8,0,3,0,0,1],
          [7,0,0,0,2,0,0,0,6],
          [0,6,0,0,0,0,2,8,0],
          [0,0,0,4,1,9,0,0,5],
          [0,0,0,0,8,0,0,7,9]]

puzzle_hard = [[0,0,0,5,0,1,7,0,2],
            [0,0,0,6,0,0,0,3,0],
            [1,0,0,0,8,0,6,0,4],
            [5,0,7,0,0,3,0,6,0],
            [8,0,0,0,0,0,0,0,3],
            [0,2,0,9,0,0,1,0,8],
            [9,0,2,0,6,0,0,0,7],
            [0,8,0,0,0,4,0,0,0],
            [4,0,3,2,0,7,0,0,0]]

def get_row(puzzle, row_num):
    return puzzle[row_num]

def get_column(puzzle, col_num):
    return [puzzle[i][col_num] for i, _ in enumerate(puzzle[0])]

def get_square(puzzle, row_num, col_num):
    square_x = row_num // 3
    square_y = col_num // 3
    coords = []
    for i in range(3):
        for j in range(3):
            coords.append((square_x * 3 + j, square_y * 3 + i))
    return [puzzle[i[0]][i[1]] for i in coords]

def get_possibilities(puzzle, row_num, col_num):
    possible = set(range(1,10))
    row = get_row(puzzle, row_num)
    col = get_column(puzzle, col_num)
    square = get_square(puzzle, row_num, col_num)
    not_possible = set(row + col + square)
    return possible - not_possible

def depth_first_solve(puzzle):
    unsolved = False
    for row_num, row_values in enumerate(puzzle):
        for col_num, value in enumerate(row_values):
            if value == 0:
                unsolved = True
                possibilities = list(get_possibilities(puzzle, row_num, col_num))
                if len(possibilities) == 1:
                    puzzle[row_num][col_num] = possibilities[0]
    if unsolved:
        return depth_first_solve(puzzle)
    return puzzle

print(depth_first_solve(puzzle_easy))