import time
import json

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
    possible = set(range(1, 10))
    row = get_row(puzzle, row_num)
    col = get_column(puzzle, col_num)
    square = get_square(puzzle, row_num, col_num)
    not_possible = set(row + col + square)
    return possible - not_possible

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
            if solve(puzzle):
                return puzzle
            else:
                puzzle[row][col] = 0
    return False

def test():
    with open('puzzles.json') as f:
        puzzles = json.load(f)
    for i in puzzles:
        start = time.time()
        solved = solve(i)
        end = time.time()
        for j in range(9):
            if sum(get_row(solved, j)) != 45:
                return False
            if sum(get_column(solved, j)) != 45:
                return False
            if sum(get_square(solved, j, j)) != 45:
                return False
        print(f'Solved puzzle in {round(end - start, 2)} seconds')
        for i in solved:
            print(i)
    return True

if __name__ == '__main__':
    test()
