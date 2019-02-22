sudoku = [[5,3,0,0,7,0,0,0,0],
          [6,0,0,1,9,5,0,0,0],
          [0,9,8,0,0,0,0,6,0],
          [8,0,0,0,6,0,0,0,3],
          [4,0,0,8,0,3,0,0,1],
          [7,0,0,0,2,0,0,0,6],
          [0,6,0,0,0,0,2,8,0],
          [0,0,0,4,1,9,0,0,5],
          [0,0,0,0,8,0,0,7,9]]

def get_row(puzzle, row_num):
    return puzzle[row_num]

def get_column(puzzle, col_num):
    return [puzzle[i][col_num] for i, _ in enumerate(puzzle[0])]

def get_square(puzzle, row_num, col_num):
    square_x = row_num // 3
    square_y = col_num // 3
    # Fuck you stackoverflow
    # coords = [[(square_x * 3 + j, square_y * 3 + i) for j in range(3)] for i in range(3)]
    coords = []
    for i in range(3):
        for j in range(3):
            coords.append((square_x * 3 + j, square_y * 3 + i))
    return [sudoku[i[0]][i[1]] for i in coords]

def possibilities(puzzle, row_num, col_num):
    possible = set(range(1,10))
    row = get_row(puzzle, row_num)
    col = get_column(puzzle, col_num)
    square = get_square(puzzle, row_num, col_num)
    not_possible = set(row + col + square)
    return possible - not_possible

print(possibilities(sudoku, 0, 2))
