import sys
import os
# add 'Carcassonne' directory to sys.path
sys.path.append(os.path.join(os.getcwd()))
print(sys.path)
print(os.getcwd())

# import local scripts
from Agents.human import HumanPlayer
from Agents.random import RandomPlayer
from Agents.vanilla_mcts import MCTS_Player
from Agents.siea_mcts import SIEA_MCTS_Player
from Games.Carcassonne.AvailableMove import AvailableMove
import datetime
import Utilities.logs_management as lm
import pandas as pd
#from player.MCTS_Player import MCTSPlayer
#from player.MCTS_RAVE_Player import MCTS_RAVEPlayer
#from player.MCTS_ES_Player import MCTS_ES_Player
#from player.Star1_Player import Star1
#from player.Star2_5_Player import Star2_5

from Carcassonne import CarcassonneState
from Tile import Tile

from pygameNextTile import nextTile
from pygameFunctions import playMove, drawGrid, diplayGameBoard, printScores, printTilesLeft
from pygameSettings import BLACK, WHITE, GRID, GRID_SIZE, GRID_BORDER, MENU_WIDTH
from pygameSettings import MEEPLE_SIZE
from pygameSettings import displayScreen

# Import and initialize the pygame library
import pygame
import pygame_menu

# Number Keys
NumKeys = [pygame.K_KP0, pygame.K_KP1, pygame.K_KP2, pygame.K_KP3, pygame.K_KP4, 
           pygame.K_KP5, pygame.K_KP6, pygame.K_KP7, pygame.K_KP8, pygame.K_KP9]

# list of player available to choose from
PLAYERS = [
        ("Human", HumanPlayer()),
        ("Random", RandomPlayer()),
        ("MCTS", MCTS_Player(max_iterations=5000)),
        ("SIEA_MCTS", SIEA_MCTS_Player(max_iterations=5000)),
        #("RAVE", MCTS_RAVEPlayer())
        ] 

PLAYER1 = [HumanPlayer()]
PLAYER2 = [HumanPlayer()]


# start menu
def startMenu():
    
    # initiate pygame
    pygame.init()
    surface = pygame.display.set_mode((600, 400))
    
    def selectPlayer1(value, player):
        PLAYER1[0] = player
    
    def selectPlayer2(value, player):
        PLAYER2[0] = player
    
    def start_the_game():
        p1 = PLAYER1[0]
        p2 = PLAYER2[0]
        PlayGame(p1, p2)
        

    # start menu
    menu = pygame_menu.Menu('Welcome', 600, 400,
                       theme=pygame_menu.themes.THEME_BLUE)
    
    # select players
    menu.add.selector('Player 1 :', PLAYERS, onchange=selectPlayer1)
    menu.add.selector('Player 2 :', PLAYERS, onchange=selectPlayer2)
    
    # play game
    menu.add.button('Play', start_the_game)
    
    # quit game
    menu.add.button('Quit', pygame_menu.events.EXIT)
    
    # start menu
    menu.mainloop(surface)


def FinalMenu(Carcassonne):
    FS = Carcassonne.FeatureScores
    Scores = Carcassonne.Scores
    
    # initiate pygame
    pygame.init()
    surface = pygame.display.set_mode((600, 800))
    
    # post game info
    winnerText = "Draw" if Carcassonne.winner is None else "Player " + str(Carcassonne.winner+1) + " Wins!"
    
    # feature info
    city = "City"
    road = "Road"
    monastery = "Monastery"
    farm = "Farm"
    total = "Total"
    p1 = "Player 1"
    p2 = "Player 2"
    gap = ""

    
    line0 = f'{p1}{gap:^20}{p2}'
    line1 = f'{Scores[2]}{total:^40}{Scores[3]}'
    line2 = f'{FS[0][0] + FS[0][3]}{city:^37}{FS[1][0] + FS[1][3]}'
    line3 = f'{FS[0][1] + FS[0][4]}{road:^37}{FS[1][1] + FS[1][4]}'
    line4 = f'{FS[0][2] + FS[0][5]}{monastery:^33}{FS[1][2] + FS[1][5]}'
    line5 = f'{FS[0][6]}{farm:^37}{FS[1][6]}'
    
    
    def nothingButton():
        pass
    
    def restart_the_game():
        startMenu()  # loop around to start menu
    
    # start menu
    menu = pygame_menu.Menu(title='Welcome', 
                            width=600, height=800,
                            theme=pygame_menu.themes.THEME_BLUE)
    
    # text
    menu.add.button(winnerText, nothingButton)
    menu.add.button(line0, nothingButton)
    menu.add.button(line1, nothingButton)
    menu.add.button(line2, nothingButton)
    menu.add.button(line3, nothingButton)
    menu.add.button(line4, nothingButton)
    menu.add.button(line5, nothingButton)
    
    # play game
    menu.add.button('Play Again', restart_the_game)
    
    # quit game
    menu.add.button('Quit', pygame_menu.events.EXIT)
    
    # start menu
    menu.mainloop(surface)



# main game loop
def PlayGame(p1, p2):
    """
    Main game logic
    """
    players = [p1,p2]
    player_index = 0
    # globals
    global GAME_DISPLAY, CLOCK
    
    # initiate pygame
    pygame.init()
    
    # manage how fast the screen updates
    CLOCK = pygame.time.Clock()
        
    # pack all settings into one object
    DisplayScreen = displayScreen(GRID, GRID_SIZE, GRID_BORDER, MENU_WIDTH, MEEPLE_SIZE)
    #game display window
    GAME_DISPLAY = DisplayScreen.pygameDisplay 
    pygame.display.set_caption('Carcassonne')  # windown name
    
    # background image
    background = pygame.image.load('Games/Carcassonne/pygame_images/table.jpg')
    background = pygame.transform.scale(background, (DisplayScreen.Window_Width, DisplayScreen.Window_Height))
    # add game title
    title = pygame.image.load('Games/Carcassonne/pygame_images/game_title.png')
    title = pygame.transform.scale(title, (274, 70))
    background.blit(title, (40, 7))
    
    # initiate game state  
    #Carcassonne = CarcassonneState(p1,p2)
    #Carcassonne = CarcassonneState()
    #Carcassonne = CarcassonneState(initial_tile_quantities=[1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0])
    #Carcassonne = CarcassonneState(initial_tile_quantities=[1 for _ in range(24)])
    Carcassonne = CarcassonneState(initial_tile_quantities=[1 for _ in range(24)], set_tile_sequence=True, set_tile_sequence_seed=1, initial_meeples=[2,2])
    Carcassonne.set_initial_state()

    # initialize the 'next tile' onject
    NT = nextTile(Carcassonne, DisplayScreen)
    
    # stay constant
    done = False  # loop until the user clicks the close button.
    isGameOver = False
    # isStartOfTurn - indicates whether it is the first tick of the next player's turn
    # hasSomethingNew - indicates an action has just happened
    isStartOfGame = isStartOfTurn = hasSomethingNew = True
    selectedMove = AvailableMove(16, 0, 0, 0, None) # first move of game
    
    # default settings
    rotation = 0
    newRotation = False
    numberSelected = 0  # default choice is no meeple

    #create game logs
    game_logs = pd.DataFrame()
    logs_path = os.path.join("Outputs", "Pygames", "pygame_at_" + str(datetime.datetime.now()))
    
    # main pygame loop
    while not done:
        
        # main event loop
        for event in pygame.event.get():
            
            # QUIT game
            if event.type == pygame.QUIT:
                pygame.quit()
                
            # show next tile in top right
            if not isGameOver:
                
                # check if player is 'AI' or is a 'Humnan' 
                # different game loops and allowable commands for each player
                if players[player_index].isAIPlayer:
                    
                    # if a keyboard button is pressed
                    if event.type == pygame.KEYDOWN:
                    
                        # play AI move when space bar is clicked
                        if event.key == pygame.K_SPACE:
                            player_choosing_action_index = player_index
                            player_index, selectedMove = playMove(NT, players[player_index], Carcassonne, NT.nextTileIndex, player_index, ManualMove = None)
                            NT = nextTile(Carcassonne, DisplayScreen)  # next tile
                            print(f'Scores: {Carcassonne.FeatureScores}')
                            isStartOfTurn = True  # new turn
                            hasSomethingNew = True  # action just happened
                            isStartOfGame = False # if move is made, the game has now started

                            #manage mcts logs
                            action_logs_path = os.path.join(logs_path, "Turn" + str(Carcassonne.turn-1) + "_pindex" + str(player_choosing_action_index))
                            #if mcts, then update logs
                            if "MCTS" in players[player_choosing_action_index].name:
                                players[player_choosing_action_index]._update_choose_action_logs()
                            players[player_choosing_action_index].dump_my_logs(path = action_logs_path)
                            action_logs = pd.concat([Carcassonne.logs_data(), players[player_choosing_action_index].choose_action_logs], axis=1)
                            lm.dump_data(action_logs, action_logs_path, "full_action_logs.csv")
                    # check if game is over
                    isGameOver = Carcassonne.is_terminal
                        
                # human player loop
                else:
                    
                    # if a keyboard button is pressed
                    if event.type == pygame.KEYDOWN:
                        
                        # tile rotations
                        if event.key == pygame.K_LEFT:
                            rotation = -1
                            newRotation = True
                        elif event.key == pygame.K_RIGHT:
                            rotation = 1
                            newRotation = True
                        
                        # number selection - for Meeple selection
                        if event.key in NumKeys:
                            numberStr = pygame.key.name(event.key)[1]
                            numberSelected = int(numberStr)  # selection of meeple location
                            if numberSelected == 0:
                                NT.Meeple = None
                            hasSomethingNew = True  # action just happened
                    
                    # is mouse is clicked
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        # get position of click
                        X, Y = NT.evaluate_click(pygame.mouse.get_pos(), DisplayScreen)
                        
                        # check is selection is valid
                        if (X, Y) in NT.possibleCoordsMeeples:
                            rotation = (90*NT.Rotated)  # rotation
                            ManualMove = AvailableMove(NT.nextTileIndex, X, Y, rotation, NT.Meeple)
                            player_index, selectedMove = playMove(NT, players[player_index], Carcassonne, NT.nextTileIndex, player_index, ManualMove=ManualMove)
                            NT = nextTile(Carcassonne, DisplayScreen)
                            print(f'Scores: {Carcassonne.FeatureScores}')
                            isStartOfTurn = True  # new turn
                            hasSomethingNew = True
                            isStartOfGame = False
                        elif (X,Y) in list(NT.Carcassonne.Board.keys()):
                            print("check")
                            text = NT.displayTextClickedTile(X,Y)
                            print(text)
                        else:
                            print(f'X: {X}')
                            print(f'Y: {Y}')
                            print("invalid")
                        
                # check if game is over
                isGameOver = Carcassonne.is_terminal
                
                if isGameOver:
                    isStartOfTurn = False
                    hasSomethingNew = False
                     
            # if game is over
            else:
                # print winner
                print(f'Winner: Player {Carcassonne.winner}, Scores:  P1: {Carcassonne.Scores[0]} - P2: {Carcassonne.Scores[1]}')
                FinalMenu(Carcassonne)
    
                    
        # display game screen
        GAME_DISPLAY.blit(background, (0, 0))  # wooden table background
        drawGrid(DisplayScreen)  # draw grid
        
        
        if hasSomethingNew:
            # meeple locations
            if players[player_index].name == "Human":
                # keep track of index
                    NT.resetImage()
                    i = 1
                    for location_key in NT.Tile.AvailableMeepleLocs:
                        location_value = NT.Tile.AvailableMeepleLocs[location_key]
                        NT.addMeepleLocations(location_key, location_value, i, numberSelected, NT.nextTileIndex)
                        NT.updateMeepleMenu(location_key, location_value, i, numberSelected)
                        i += 1
                    NT.rotate(NT.Rotated, newRotation)
            # instruction to press 'Space'   
            else:
                if not isGameOver:
                    NT.pressSpaceInstruction()
            
        
        if isStartOfTurn:
            # last play
            NT.updateMoveLabel(Carcassonne, selectedMove, isStartOfGame)
        
        # print scores
        printScores(Carcassonne, DisplayScreen)
        #displayTilesLeft
        printTilesLeft(Carcassonne, DisplayScreen)
        
        # show next tile in top right
        if not isGameOver:
            # show tile in top corner
            NT.showNextTile(DisplayScreen, rotation, newRotation)
            #print meeple info
            NT.showInfos(DisplayScreen)
            # possible moves
            NT.highlightPossibleMoves(DisplayScreen)
            
        # only one rotation per click
        newRotation = False
        numberSelected = 0
        
        # show all tiles in the board
        diplayGameBoard(Carcassonne, DisplayScreen)
        
        # update game screen
        pygame.display.flip()
        
        # first tick is over
        isStartOfTurn = False
        hasSomethingNew = False
        
        # FPS
        CLOCK.tick(10)
            
if __name__ == "__main__":
    startMenu()
    
    

    
