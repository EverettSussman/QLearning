import pygame as pg
import numpy as np 
import sys

from board_interpreter import MakeBoard

# Colors
WHITE = (255,255,255)
GREY = (209,209,209)
DARKGREY = (140,140,140)
BLACK = (0,0,0)

GREEN = (0,204,0)

RED = (204,0,0)
ORANGE = (255,128,0)

class GridWorld:

    def __init__(self, params=None, learner=None, board_file=None, epoch=None, speed=10):
        """
        Constructor for GridWorld class.

        Keyword Arguments:

        params: Dictionary containing game parameters.  We have 
            probs: Transition probabilities for game.  If None, then probabilities are 
                determinitstic.  If tuple of floats, then probabilities go by given 
                distribution where [action to left, action, action to right] is distribution.

        action_fn: Function for handling actions of game, takes in game state as input.

        reward_fn: Function for handling rewards of game.
        """
        self.WIDTH = 800
        self.HEIGHT = 600

        self.boardHEIGHT = self.HEIGHT
        self.boardWIDTH = .75 * self.WIDTH

        self.rest = speed
        if epoch is None:
            self.epoch = 0
        else:
            self.epoch = epoch

        self.obPenalty = -2
        self.score = 0

        self.actions = ["RIGHT", "LEFT", "UP", "DOWN"]
        self.actionDict = {"RIGHT":0, "LEFT":1, "UP":2, "DOWN":3}

        self.initializeLearner(learner)

        self.initializeParams(params)

        self.initializeBoard()

        self.initializeAgent()

        self.initializePygame()

        # To be updated during game
        self.state = None

    def initializePygame(self):
        # Initialize pygame
        pg.init()
        pg.font.init()

        # select font
        self.qfont = pg.font.SysFont('helvetica', 20)
        self.scorefont = pg.font.SysFont('helvetica', 30)

        # Set up the screen for rendering.
        self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT), 0, 32)

    def initializeLearner(self, learner):
        # determine action and reward functions
        if learner is None:
            self.learner = None
            self.action_fn = None
            self.reward_fn = None
        else:
            self.learner = learner
            self.action_fn = learner.action_fn
            self.reward_fn = learner.reward_fn

    def initializeParams(self, params):
        # Extract parameters for game
        if params is None:
            self.probs = np.eye(len(self.actions))
            self.board_file = None
        else:
            self.board_file = params['board_file']
            probs = params['probs']
            if len(probs) != 3:
                raise NameError("Invalid probability dist.")
            else:
                self.probs = np.array([[probs[1], 0, probs[0], probs[2]],
                                       [0, probs[1], probs[2], probs[0]],
                                       [probs[2], probs[0], probs[1], 0],
                                       [probs[0], probs[2], 0, probs[1]]])

    def initializeAgent(self):
        goodLoc = False
        while not goodLoc:
            x = np.random.randint(self.board.shape[1])
            y = np.random.randint(self.board.shape[0])
            if self.board[y][x] is not None and [x, y] != self.goal:
                goodLoc = True

        self.agentLoc = [x, y]

    def initializeBoard(self):
        # Generate board
        if self.board_file is None:
            self.mb = MakeBoard('gridworld/maps/map0.board')
        else:
            self.mb = MakeBoard(self.board_file)

        self.board = self.mb.board
        self.goal = self.mb.goalLoc

    def moveAgent(self, action):
        actionVal = self.actionDict[action]
        realAction = np.random.choice(self.actions, p=self.probs[actionVal])
        self.move(realAction)

    def move(self, realAction):
        x = self.agentLoc[0]
        y = self.agentLoc[1]
        if realAction == "DOWN":
            xnew, ynew = x, y + 1
        elif realAction == "UP":
            xnew, ynew = x, y - 1
        elif realAction == "RIGHT":
            xnew, ynew = x + 1, y
        elif realAction == "LEFT":
            xnew, ynew = x - 1, y

        if (xnew < 0 or xnew >= self.board.shape[1] 
                    or ynew < 0 or ynew >= self.board.shape[0] 
                    or self.board[ynew][xnew] is None):
            self.updateScore(xnew, ynew, ob=True)
        else:
            self.updateScore(xnew, ynew, ob=False)
            self.agentLoc[0], self.agentLoc[1] = xnew, ynew

    def updateScore(self, x, y, ob=False):
        """ Update reward function callback (if not None) 
            and score. """
        if ob == True:
            if self.reward_fn is not None:
                self.reward_fn(self.obPenalty)
            self.score += self.obPenalty
        else:
            if self.reward_fn is not None:
                self.reward_fn(self.board[y][x])
            self.score += self.board[y][x]

    def generateState(self, loc):
        """
        Update state information to give learner.
        Format is agentLoc (x,y), goal location (x,y), number rows in board,
        number columns in board, rewards of board 
        """
        state = (loc[0], loc[1], self.goal[0], self.goal[1])
        state += (len(self.board),)
        state += (len(self.board[0]),)
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                state += (self.board[i][j],)
        return state

    def drawTile(self, i, j, x, y):
        val = self.mb.chars[i][j]

        if val == 'X':
            pg.draw.rect(self.screen, BLACK, (x, y, self.tileWIDTH, self.tileHEIGHT), 0)
        else:
            if val == 'G':
                pg.draw.rect(self.screen, GREEN, (x, y, self.tileWIDTH, self.tileHEIGHT), 0)
            elif val == 'B':
                pg.draw.rect(self.screen, RED, (x, y, self.tileWIDTH, self.tileHEIGHT), 0)

            # draw angled lines
            pg.draw.line(self.screen, WHITE, [x, y], [x + self.tileWIDTH, y + self.tileHEIGHT], 3)
            pg.draw.line(self.screen, WHITE, [x + self.tileWIDTH, y], [x, y + self.tileHEIGHT], 3)

            # DRAW QVALUES HERE
            if self.learner is not None:
                # need agentLoc to be j, i
                state = self.generateState([j,i])

                bestAction, _ = self.learner.getAction(state)
                # draw qvals
                self.drawQVal("UP", bestAction, state, x + self.tileWIDTH / 2 - 20, y + 10)
                self.drawQVal("RIGHT", bestAction, state, x + 3 * self.tileWIDTH / 4 - 15, y + self.tileHEIGHT / 2 - 10)
                self.drawQVal("DOWN", bestAction, state, x + self.tileWIDTH / 2 - 20, y + self.tileHEIGHT - 30)
                self.drawQVal("LEFT", bestAction, state, x + 1 * self.tileWIDTH / 4 - 30, y + self.tileHEIGHT / 2 - 10)

    def drawQVal(self, action, bestAction, state, x, y):
        if bestAction == action:
            color = ORANGE
        else:
            color = WHITE
        text = self.qfont.render('{0:.2f}'.format(float(self.learner.getVal(state, action))), True, color)
        self.screen.blit(text, (x, y)) 

    def convertPixels(self, x, y):
        return int(x * self.tileWIDTH), int(y * self.tileHEIGHT)

    def drawAgent(self):
        x, y = self.convertPixels(self.agentLoc[0], self.agentLoc[1])

        circX, circY = int(x + self.tileWIDTH / 2), int(y + self.tileHEIGHT / 2)
        agentRad = 30
        pg.draw.circle(self.screen, ORANGE, [circX, circY], agentRad, 0)

    def drawBoard(self):
        # Figure out how many boxes 
        self.tileWIDTH = self.boardWIDTH / self.board.shape[1]
        self.tileHEIGHT = self.boardHEIGHT / self.board.shape[0]

        pg.draw.rect(self.screen, GREY, 
                    (0, 0, self.boardWIDTH, self.boardHEIGHT), 0)
        pg.draw.rect(self.screen, DARKGREY, 
                    (0, 0, self.boardWIDTH, self.boardHEIGHT), 5)

        # fill in tiles
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                x, y = self.convertPixels(j, i)
                self.drawTile(i, j, x, y)
                pg.draw.line(self.screen, DARKGREY, [0, y], [self.boardWIDTH, y], 5)
                pg.draw.line(self.screen, DARKGREY, [x, 0], [x, self.boardHEIGHT], 5)

    def drawScore(self):
        # Print epoch
        epochText = self.scorefont.render('Epoch: {}'.format(self.epoch), True, WHITE)
        self.screen.blit(epochText, (self.boardWIDTH + 20, self.HEIGHT * .1))

        # Print score
        score = self.scorefont.render('Score: {}'.format(self.score), True, WHITE)
        self.screen.blit(score, (self.boardWIDTH + 20, self.HEIGHT * .2))

        # Print speed
        speedText = self.scorefont.render('Speed: {}s'.format(self.rest / 1000.), True, WHITE)
        self.screen.blit(speedText, (self.boardWIDTH + 20, self.HEIGHT * .3))

    def game_loop(self):
        # Redraw screen every time
        self.screen.fill(BLACK)

        # Draw Board 
        self.drawBoard()

        # Draw Agent
        self.drawAgent()

        # Draw score
        self.drawScore()

        # Render the display.
        pg.display.update()

        # Update the state information
        self.state = self.generateState(self.agentLoc)

        # Process input events.
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()
            elif not hasattr(event, "key"):
                continue
            elif event.type == pg.KEYUP:
                if self.action_fn is None:  
                    if event.key == pg.K_DOWN:
                        self.moveAgent("DOWN")
                    elif event.key == pg.K_UP:
                        self.moveAgent("UP")
                    elif event.key == pg.K_RIGHT:
                        self.moveAgent("RIGHT")
                    elif event.key == pg.K_LEFT:
                        self.moveAgent("LEFT")
                # Check whether rest should change
                elif event.key == pg.K_MINUS:
                    if self.rest < 2000:
                        self.rest *= 2
                elif event.key == pg.K_EQUALS:
                    if self.rest <= 10:
                        self.rest = 1
                    else:
                        self.rest /= 2

        # Have agent move if learning
        if self.action_fn is not None:
            action = self.action_fn(self.state)
            self.moveAgent(action)  

        # Check whether game is over
        # NEED TO IMPLEMENT LEARNING OF FINAL STATE
        if self.agentLoc == self.goal:
            self.action_fn(self.state)
            return False   

        # Wait just a bit.
        pg.time.delay(self.rest)

        return True

    def bare_game_loop(self):

        self.state = self.generateState(self.agentLoc)

        # Have agent move if learning
        if self.action_fn is not None:
            action = self.action_fn(self.state)
            self.moveAgent(action) 

        # Check whether game is over
        # NEED TO IMPLEMENT LEARNING OF FINAL STATE
        if self.agentLoc == self.goal:
            # self.reward_fn()
            self.action_fn(self.state)
            return False   

        return True

if __name__ == '__main__':
    
    game = GridWorld(board_file='maps/map0.board', probs=[.1, .8, .1])

    while game.game_loop():
        pass

