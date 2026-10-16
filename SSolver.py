import csv

def retrieveSudoku():
    sudoku = []
    filepath = "Sudokus/" + input("What is the filename for the sudoku puzzle - ") + '.csv'
    with open(filepath, 'r') as file:
        csvFile = csv.reader(file, delimiter=',')
        for row in csvFile:
            sudoku.append(row)
    return sudoku

def solver(sudoku, domain):
    # TODO: implement solver
    for row in sudoku:
        print(row)


def main():
    domain = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    sudoku = retrieveSudoku()
    solver(sudoku, domain)

main()