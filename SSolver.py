import csv
import sys
states = []
stateVal = []

def retrieveSudoku():
    sudoku = []                     # initialise the variable sudoku so each row of the sudoku can be appended as each row is iterated over
    filepath = "Sudokus/" + sys.argv[1] + '.csv'     # get the filepath for the sudoku, all sudoku must be stored in the folder, uses cmd line input for filename
    with open(filepath, 'r') as file:       # opens the file and reads it into the array sudoku
        csvFile = csv.reader(file, delimiter=',')
        for row in csvFile:
            sudoku.append(row)
    file.close()
    for x in range(9):              #this is to cast every value in the sudoku from a string to an integer so it can be used in the solver function
        for y in range(9):
            sudoku[x][y] = int(sudoku[x][y])
    return sudoku


def check(sudoku, x, y, val):       #thi schecks if the value assigned is currently valid, it does not know if the value will be valid in the future
    for i in range(9):              #this is how I check if the value is valid in its row
        if sudoku[x][i] == val:
            return False
    for i in range(9):              #this is how i check if the value is valid in its column
        if sudoku[i][y] == val:
            return False
    subRow = x - (x%3)              #this gives us the sub grid the position we are looking at belongs to by figuring out its sub row and sub column
    subCol = y - (y%3)
    for j in range(3):              #to check if the value is valid within the subgrid we want to iterate through each column in each row, since this is always 3 I cna hard code that in
        for k in range(3):
            if sudoku[k + subRow][j + subCol] == val:       #when I validate this, it takes the index of the subrow and subcolumn we are at and increments it by the looping value
                return False
    return True

#I decided to use a recursive function to implement the solver, this is because it allows me to easily backtrack when the is an invalid value
def solver(sudoku, domain, x, y):
    if y > 8:                       #we need to check if we have reached the end of the column in which case reset the position along the column and increment the row
        x += 1
        y = 0
    if x > 8:                       #when we have reached the end of the sudoku we can print the sudoku and return true to indicate that the sudoku has been solved
        for row in sudoku:
            print(" ".join(map(str, row)))      #this just formats the output of the sudoku in a more readable way        return True
    else:                           #this is if the i havent reached the end of the sudoku it keeps the program running
        if sudoku[x][y] == 0:       #empty cells are defined by 0 so if the cell is "empty" then we need to iterate until we find the valid value for this cell
            for num in domain:
                if check(sudoku, x, y, num):        #if the check comes back as successful then we assign the value to that cell and recursively call the function moving onto the next position
                    sudoku[x][y] = num          #this does not guarantee that the value in this position is valid, it could find later on that there is no valid solution with the value in this position at which point it will backtrack and change the number again
                    if solver(sudoku, domain, x, y+1):  #this returns true when the value at this position is definitely valid. this starts at the very final cell and works its way back
                        return True
                    sudoku[x][y] = 0        #if the value is not valid, then we need to reset the value back to 0, "empty", otherwise on the next iteration it will pass over the cell and we wont have a valid sudoku anymore
            return False        #this returns false when the check returns false so we know its not a valid value and it will try another one
        else:                   #this moves onto the next cell when we come across a cel which already has a value in it 
            return solver(sudoku, domain, x, y+1)

def main():
    domain = [1, 2, 3, 4, 5, 6, 7, 8, 9]        # defines the domain for the sudoku in main so it can be passed as a parameter into the solver function
    sudoku = retrieveSudoku()       # initialises a 2D array sudoku to store the initial sudoku grid, this then is passed as a parameter into the solver function
    #domainDict = calcDomains(sudoku, domain)
    if solver(sudoku, domain, 0, 0):        #this is our final message which prints to the terminal telling the user if the sudoku was solved or if it is an invalid sudoku
        print("Sudoku successfuly solved")
    else:
        print("Sudoku is not solvable")

main()

"""
peerDict = {
        (0,0): [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(1,1),(1,2),(2,1),(2,2)],
        (0,1): [(0,0),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,1),(2,1),(3,1),(4,1),(5,1),(6,1),(7,1),(8,1),(1,0),(1,2),(2,0),(2,2)],
        (0,2): [(0,0),(0,1),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),(8,2),(1,0),(1,1),(2,0),(2,1)],
        (0,3): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (0,4): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (0,5): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (0,6): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (0,7): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (0,8): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (1,0): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (1,1): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (1,2): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (1,3): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (1,4): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (1,5): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (1,6): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (1,7): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (1,8): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (2,0): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (2,1): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (2,2): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (2,3): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (2,4): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (2,5): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (2,6): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (2,7): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (2,8): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (3,0): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (3,1): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (3,2): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (3,3): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (3,4): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (3,5): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (3,6): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (3,7): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (3,8): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (4,0): [(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        (4,1): [],
        (4,2): [],
        (4,3): [],
        (4,4): [],
        (4,5): [],
        (4,6): [],
        (4,7): [],
        (4,8): [],
        (5,0): [],
        (5,1): [],
        (5,2): [],
        (5,3): [],
        (5,4): [],
        (5,5): [],
        (5,6): [],
        (5,7): [],
        (5,8): [],
        (6,0): [],
        (6,1): [],
        (6,2): [],
        (6,3): [],
        (6,4): [],
        (6,5): [],
        (6,6): [],
        (6,7): [],
        (6,8): [],
        (7,0): [],
        (7,1): [],
        (7,2): [],
        (7,3): [],
        (7,4): [],
        (7,5): [],
        (7,6): [],
        (7,7): [],
        (7,8): [],
        (8,0): [],
        (8,1): [],
        (8,2): [],
        (8,3): [],
        (8,4): [],
        (8,5): [],
        (8,6): [],
        (8,7): [],
        (8,8): [],
    }
"""