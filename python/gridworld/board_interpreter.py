import numpy as np
import os.path, sys, re

class MakeBoard:

    def __init__(self, boardName):

        """ Constructor for MakeBoard
        
        Arguments:

            boardName: filename to generate board.  Extension 
        is of form .board.

        """
        self.board = np.array([])
        self.chars = np.array([])
        self.comments = ['\"']

        # Check file name
        if os.path.exists(boardName):
            if os.path.splitext(boardName)[1] == '.board':
                self.file = boardName
            else:
                raise NameError('Wrong file type.')
        else:
            raise NameError('File does not exist.')

        self.vals = {'X': None, '0': -1, 'G': 10, 'B': -10}

        # Generate board and chars
        self.genBoard()

        # Find goal location
        self.goalLoc = None
        for i in range(len(self.chars)):
            for j in range(len(self.chars[i])):
                if self.chars[i][j] == 'G':
                    self.goalLoc = [j, i]


        return 

    def __str__(self):

        """ Printing Board """
        boardString = ''
        for row in range(len(self.chars)):
            # boardString += '|' + '-|' * len(self.chars[0]) + '\n'
            temp = ''
            for char in self.chars[row]:
                temp += char # + '|'
            temp += '\n'
            boardString += temp
        # boardString += '|' + '-|' * len(self.chars[0]) + '\n'

        return boardString

    def process(self, data):
        """ Process board data """
        # Remove comments
        for char in self.comments:
            commentFlag = 0
            init = 0
            for i, datum in enumerate(data):
                if char == datum:
                    if commentFlag == 0:
                        init = i
                        commentFlag = 1
                    else:
                        commentFlag = 0
                        data = data[:init] + data[i+1:]
                        init = 0

        # Generate board matrix
        board = []
        charboard = []
        lines = data.split('\n')
        for line in lines: 
            temp = []
            chartemp = line
            for char in line:
                if char in self.vals.keys():
                    temp.append(self.vals[char])
            if temp != []:
                board.append(temp)
                charboard.append(chartemp)

        self.chars = np.array(charboard)
        self.board = np.array(board)

    def genBoard(self):

        with open(self.file, 'r') as f:
            data = f.read()
            self.process(data)
        f.close()

# Testing

# mb = MakeBoard('gridworld/maps/map0.board')
# print(mb)
# print(mb.board)
