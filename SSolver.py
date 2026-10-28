import csv
import sys

def retrieveSudoku():
    sudoku = []                     # initialise the variable sudoku so each row of the sudoku can be appended as each row is iterated over
    filepath = "Sudokus/" + sys.argv[1] + '.csv'     # get the filepath for the sudoku, all sudoku must be stored in the folder, uses cmd line input for filename
    with open(filepath, 'r') as file:       # opens the file and reads it into the array sudoku
        csvFile = csv.reader(file, delimiter=',')
        for row in csvFile:
            if len(row) != 9:       #if a row has more or less than 9 values we know it cannot be valid so end
                sudoku = None
                return sudoku
            sudoku.append(row)
        if csvFile.line_num != 9:   # after iterating through the rows our line number is 9 if it is a valid sudoku, so if it isnt we know the sudoku inputted does not have 9 rows
            sudoku = None
            return sudoku
    file.close()
    for x in range(9):              #this is to cast every value in the sudoku from a string to an integer so it can be used in the solver function
        for y in range(9):
            try:                    # this is so when we cast the value to an int if it isnt a number we catch an exception so it doesnt error out
                sudoku[x][y] = int(sudoku[x][y])
            except ValueError:
                sudoku = -1
                return sudoku

    return sudoku

def initDomains(sudoku, domainDict):        # this function is used to set the initial state of the domains for each cell before starting the solve. this is needed for a valid forward checking algorithm to work
    for x in range(9):
        for y in range(9):      # we are looping htrough every possible cell so we can explore every domain
            if sudoku[x][y] != 0:       # when the cell isnt empty i.e. when it initially has a value we know we need to update all the peer cells which this affects
                domainDict[(x,y)] = []     # when the cell has a value it has no possible values so its domain is empty
                if not updateDomains(sudoku, domainDict, x, y, sudoku[x][y], []):   # this tells us if our sudoku is invalid from either from it initially having a conflicting value in 2 peer cells or if having the value in this cell causes another cell to have no possible values. This only works when we are initialising the domains, after this it tells us that the value we are placing in the cell is invalid
                    return None
    return domainDict

def updateDomains(sudoku, domainDict, x, y, val, removed):      # this is the function which checks if a value is valid and updates the peer domains accordingly
    for i in range(9):      # this is to check through every column in the current row
        if sudoku[x][i] == 0:
            if val in domainDict[(x,i)]:        # we can only remove a value from the domain dictionary if it is currently in it otherwise we get a ValueError
                domainDict[(x,i)].remove(val)
                removed.append  (((x,i), val))  # when we remove a value from the cells domain, we need to keep track of it so if we find the value is invalid we will backtrack and add the removed value back into the cells domain
                if domainDict[(x,i)] == [] :    # this checks if the cells domain is empty after removing the new value, if so we know this value is invalid so we backtrack and try the next value
                    return False
        elif sudoku[x][i] == val and i != y:    # we also need to check if the cells actual value is the same as the one we have added, however, we must skip over the position where we just added the cell otherwise it will always match and think all values are invalid
            return False
    for i in range(9):      # we are now checking through every row in the current column
        if sudoku[i][y] == 0:
            if val in domainDict[(i,y)]:        # again we are checking if the cell doesnt have a value, if the domain contains the value, if when we remove the value from the domain is it still valid and we compare actual cell values, skipping over the current position
                domainDict[(i,y)].remove(val)
                removed.append(((i,y), val))
                if domainDict[(i,y)] == []:
                    return False
        elif sudoku[i][y] == val and i != x:
            return False
    subRow = x - (x%3)      # this is when we move onto the subgrid, in order to find which subgrid we are in I use a basic moduli calculation for both the row and column
    subCol = y - (y%3)
    for j in range(3):
        for k in range(3):
            if sudoku[k + subRow][j + subCol] == 0: # this check if the cell does not have a value then we want to check the cell domains
                if val in domainDict[(k + subRow,j + subCol)]:      # again we must check if the domain contains the value so we dont get a ValueError by trying to remove a value which doesnt exist
                    domainDict[(k + subRow,j + subCol)].remove(val)
                    removed.append(((k + subRow, j + subCol), val)) # again we add the removed value at its cell position to the removed list for backtracking
                    if domainDict[(k + subRow,j + subCol)] == []:   # again if after removing hte domain is empty we know it is invalid so we backtrack
                        return False
            elif sudoku[k + subRow][j + subCol] == val and (k + subRow, j + subCol) != (x,y):   # again if there is a cell with the same value in the sub grid which isnt the cell we inserted the value, it is invalid and we must backtrack
                return False
    return True

#I decided to use a recursive function to implement the solver, this is because it allows me to easily backtrack when the is an invalid value
def solver(sudoku, domainDict, x, y):
    if y > 8:                       #we need to check if we have reached the end of the column in which case reset the position along the column and increment the row
        x += 1
        y = 0
    if x > 8:                       #when we have reached the end of the sudoku we can print the sudoku and return true to indicate that the sudoku has been solved
        for row in sudoku:
            print(" ".join(map(str, row)))      #this just formats the output of the sudoku in a more readable way
        return True
    # we continue if we have not reached the end of the sudoku
    if sudoku[x][y] == 0:       #empty cells are defined by 0 so if the cell is "empty" then we need to iterate until we find the valid value for this cell
        for num in list(domainDict[(x, y)]):
            sudoku[x][y] = num      #this does not guarantee that the value in this position is valid, it could find later on that there is no valid solution with the value in this position at which point it will backtrack and change the number again
            removed = []            #this is to track in this iteration what values we remove at what position so we can backtrack if neccessary
            if updateDomains(sudoku, domainDict, x, y, num, removed):        #if the check comes back as successful then we assign the value to that cell and recursively call the function moving onto the next position
                if solver(sudoku, domainDict, x, y+1):  #this returns true when the value at this position is definitely valid. this starts at the very final cell and works its way back
                    return True
            for cell, val in removed:   # this is where we backtrack the values we have removed from the "instance" domains if it is an invalid number in the current cell
                domainDict[cell].append(val)
            sudoku[x][y] = 0        #if the value is not valid, then we need to reset the value back to 0, "empty", otherwise on the next iteration it will pass over the cell and we wont have a valid sudoku anymore
        return False        #this returns false when the check returns false so we know its not a valid value and it will try another one
    else:                   #this moves onto the next cell when we come across a cel which already has a value in it 
        return solver(sudoku, domainDict, x, y+1)

def main():
    domainDict = {(p, vs): list(range(1, 10)) for p in range(9) for vs in range(9)}     # initially I manually defined every domain at each cell taking up 83 lines of code then I found this method to define dictionaries on geeksforgeeks: https://www.geeksforgeeks.org/python/create-dynamic-dictionary-in-python/
    sudoku = retrieveSudoku()       # initialises a 2D array sudoku to store the initial sudoku grid, this then is passed as a parameter into the solver function
    if sudoku is None:
        print("Sudoku is invalid")
        return
    elif sudoku == -1:
        print("value in sudoku is not a number")
        return
    domainDict = initDomains(sudoku, domainDict)
    if domainDict is None:
        print("Sudoku is not solvable")
        return
    if solver(sudoku, domainDict, 0, 0):        #this is our final message which prints to the terminal telling the user if the sudoku was solved or if it is an invalid sudoku
        print("Sudoku successfuly solved")
    else:
        print("Sudoku is not solvable")

main()