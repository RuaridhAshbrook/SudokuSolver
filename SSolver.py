import csv

def retrieveSudoku():
    filepath = input("What is the filepath for the sudoku - ")
    sudoku = csv.reader(filepath)
    return sudoku

def solver(sudoku, domain):
    # TODO: implement solver
    pass


def main():
    domain = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    sudoku = retrieveSudoku()
    solver(sudoku, domain)

main()