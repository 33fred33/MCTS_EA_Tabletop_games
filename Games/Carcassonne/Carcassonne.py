import random as rd
import itertools as it

from Games.Carcassonne.Tile import Tile, ROTATION_DICT, SIDE_CHANGE_DICT, AvailableMove
from Games.Carcassonne.GameFeatures import Monastery

from Games.Carcassonne.Carcassonne_CityUtils import cityConnections, cityClosures
from Games.Carcassonne.Carcassonne_RoadUtils import roadConnections, roadClosures
from Games.Carcassonne.Carcassonne_FarmUtils import farmConnections
import pandas as pd
import Games.base_games as base_games

SIDE_COMPARISON_DICT={
    2:0,
    3:1,
    0:2,
    1:3
    }
     
class CarcassonneState:
    
    def __init__(self, name = "Carcassonne", no_farms = False, reward_type = "Score_difference", losing_reward = -1, draw_reward = 0, winning_reward = 1, initial_tile_quantities = [1,3,1,1,2,3,2,2,2,3,1,3,2,5,3,2,4,3,3,4,4,9,8,1], initial_meeples = [7,7], set_tile_sequence = False, set_tile_sequence_seed = None, players = 2):

        assert len(initial_tile_quantities) == 24, "The initial tile quantities must be a list of 24 ints"
        assert initial_tile_quantities[16] > 0, "The initial tile quantities must have at least one tile of index 16"

        # players
        self.name = name
        self.no_farms = no_farms
        
        #Never changing:
        self.MatchingSide = [2,3,0,1]
        self.MatchingLine = [2,1,0]
        #Initialialize variables        
        self.Board = {}
        self.BoardCities = {}
        self.BoardRoads = {}
        self.BoardMonasteries = {}
        self.BoardFarms = {}
        self.MonasteryOpenings = {}
        self.AvailableSpots = set()
        self.AvailableSpots.add((0,0))  # first tile always placed in this position
        #self.AvailableMoves = []
        self.reward_type = reward_type
        self.losing_reward = losing_reward
        self.draw_reward = draw_reward
        self.winning_reward = winning_reward
        self.initial_tile_quantities = initial_tile_quantities
        self.initial_meeples = initial_meeples 
        self.set_tile_sequence = set_tile_sequence #has random events or tiles are calculated at first
        self.set_tile_sequence_seed = set_tile_sequence_seed #seed for initial tile sequence
        self.players = players

        self.TileQuantities = [n for n in self.initial_tile_quantities] #keeps independent count to the initla one
        self.TotalTiles = sum(self.TileQuantities)
        self.UniqueTilesCount = len(self.TileQuantities)
        self.TileIndexList = []
        for i in range(self.UniqueTilesCount):
            self.TileIndexList += [i for _ in range(self.TileQuantities[i])]
        
        
        self.max_possible_score = self.max_score()
        
    def set_initial_state(self):

        self.Meeples = self.initial_meeples
        self.winner = None
        self.result = None
        self.reward = [None, None]
        self.player_turn = 1 if self.players > 1 else 0 # player 2 "plays" the first tile 
        self.is_terminal = False
        self.random_events = []
        self.turn = 0  # number of turns played
        self.meeples_on_farms = [0,0]  # number of meeples on farms
        self.total_meeples_played = [0,0]  # total number of meeples played
        # scores
        self.Scores = [0,0,0,0]  #P1, P2, P1 virtual, P2 virtual
        self.FeatureScores = [   # [City, Road, Monastery, City(Incomplete), Road(Incomplete), Monastery(Incomplete), Farms]
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0]
            ]
        
        self.deck = self.TileIndexList.copy()
        if self.set_tile_sequence:
            rd.seed(self.set_tile_sequence_seed)
        rd.shuffle(self.deck)  # shuffle the deck
        
        # each game starts with the same tile (Tile16) being placed
        # PlayingTileIndex=16, X,Y = 0,0, Rotation=0, MeepleKey=None
        self.make_action(AvailableMove(16, 0, 0, 0, None))
        
        # code for running tests by arranging the order of the first few tiles
        
    def duplicate(self):
        """
        Clones the game state - quicker than using copy.deepcopy()
        """
        #Clone = copy.deepcopy(self)
        Clone = CarcassonneState(name=self.name, no_farms = self.no_farms, reward_type=self.reward_type, losing_reward=self.losing_reward, draw_reward=self.draw_reward, winning_reward=self.winning_reward, initial_tile_quantities=[x for x in self.initial_tile_quantities], initial_meeples=self.initial_meeples, set_tile_sequence=self.set_tile_sequence, set_tile_sequence_seed=self.set_tile_sequence_seed)
        Clone.Board = {k:v.CloneTile() for k,v in self.Board.items()}
        Clone.BoardCities = {k:v.CloneCity() for k,v in self.BoardCities.items()}
        Clone.BoardRoads = {k:v.CloneRoad() for k,v in self.BoardRoads.items()}
        Clone.BoardMonasteries = {k:v.CloneMonastery() for k,v in self.BoardMonasteries.items()}
        Clone.BoardFarms = {k:v.CloneFarm() for k,v in self.BoardFarms.items()}
        Clone.MonasteryOpenings = {k:[x for x in v] for k,v in self.MonasteryOpenings.items()}
        Clone.AvailableSpots = set([x for x in self.AvailableSpots])
        #Clone.AvailableMoves = [x for x in self.AvailableMoves] #Not needed?
        Clone.Meeples = [x for x in self.Meeples]
        Clone.winner = self.winner
        Clone.result = self.result
        Clone.reward = [x for x in self.reward]
        Clone.player_turn = self.player_turn
        Clone.is_terminal = self.is_terminal   
        Clone.turn = self.turn
        Clone.Scores = [x for x in self.Scores]
        Clone.FeatureScores = [x[:] for x in self.FeatureScores]
        Clone.TileQuantities = [x for x in self.TileQuantities]
        Clone.TotalTiles = self.TotalTiles
        Clone.UniqueTilesCount = self.UniqueTilesCount   
        Clone.TileIndexList = [x for x in self.TileIndexList]
        Clone.deck = [x for x in self.deck]
        Clone.random_events = [x.duplicate() for x in self.random_events]
        Clone.available_actions = [x for x in self.available_actions]
        Clone.next_tile_index = self.next_tile_index
        Clone.max_possible_score = self.max_possible_score
        Clone.meeples_on_farms = [x for x in self.meeples_on_farms]
        Clone.total_meeples_played = [x for x in self.total_meeples_played]
        Clone.set_tile_sequence = self.set_tile_sequence
        Clone.set_tile_sequence_seed = self.set_tile_sequence_seed
        Clone.initial_meeples = self.initial_meeples
        Clone.players = self.players
        return Clone 
    
    def AddMeeple(self, MeepleUpdate, MeepleKey, FeatureCharacter, i):
        """
        Add meeple to section of tile
        
        Inputs:
            - MeepleUpdate: [p1_Meeple, p2_Meeple] - List of Meeples (e.g [1,0] means P1 is adding a Meeple)
            - MeepleKey: Feature to which Meeple is added
            - FeatureCharacter: Possible Meeple Location
            - i: Possible feature index
        """
        AddedMeeples = [0,0]
        if MeepleKey is not None:
            # which feature the meeple is added to
                if MeepleKey[0] == FeatureCharacter and MeepleKey[1] == i:
                    AddedMeeples = MeepleUpdate
        return AddedMeeples
     
    def completeMonastery(self, AffectedMonastery):
        """
        Checks if the monastery is completed (completely surrounded)
        """
        AffectedMonastery.Value += 1
        # if monastery is complete
        if AffectedMonastery.Value == 9:
            self.Meeples[AffectedMonastery.Owner] += 1  # return meeple to player
            self.Scores[AffectedMonastery.Owner] += 9  # award points to player
            self.FeatureScores[AffectedMonastery.Owner][2] += 9 # update monastery feature score
            AffectedMonastery.Value = 0  
    
    def checkMonasteryCompleteness(self, X,Y, SurroundingSpots, MeepleUpdate, MeepleKey=None ):    
        """
        Monastery completeness check
        
        Inputs:
            - X,Y: tile coordinates of tile just played
            - SurroundingSpots: Coordinates of surrounding spots (left,right,above,below)
            - MeepleUpdate: [p1_Meeple, p2_Meeple] - List of Meeples (e.g [1,0] means P1 is adding a Meeple)
            - MeepleKey:Feature to which Meeple is added
        """
        # check if new tile is surrounding a monastery
        if (X,Y) in self.MonasteryOpenings:
            # check if monastery is completed
            [(self.completeMonastery(self.BoardMonasteries[AffectedMonasteryIndex]))  for AffectedMonasteryIndex in self.MonasteryOpenings[(X,Y)]]
            # spot is now filled, remove from possible locations
            del self.MonasteryOpenings[(X,Y)]  
            
        #Monastery logic
        if MeepleKey is not None:
            if MeepleKey[0] == "Monastery":
                # monastery surroundings include all 8 surrounding positions
                CompleteSurroundingSpots = SurroundingSpots + [(X-1,Y-1),(X+1,Y+1),(X+1,Y-1),(X-1,Y+1)]
                NextMonasteryID = len(self.BoardMonasteries)  # increment ID by 1
                
                self.BoardMonasteries[NextMonasteryID] = Monastery(NextMonasteryID,self.player_turn)
                AffectedMonastery = self.BoardMonasteries[NextMonasteryID]
                [self.monasterySurroundings(Spot, AffectedMonastery, NextMonasteryID) for Spot in CompleteSurroundingSpots]
                
    def monasterySurroundings(self, Spot, AffectedMonastery, NextMonasteryID):
        """
        Function for 'for loop'
        
        Inputs:
            - Spot: (X,Y)
            - AffectedMonastery: Monastery feature
            - NextMonasteryID: If Monastery is new, this will be its ID
        """
        if Spot in self.Board:
            self.completeMonastery(AffectedMonastery)
        elif Spot in self.MonasteryOpenings:
            self.MonasteryOpenings[Spot].append(NextMonasteryID)
        else:
            self.MonasteryOpenings[Spot] = [NextMonasteryID]
                   
    def checkCityCompleteness(self, PlayingTile, Surroundings, MeepleUpdate, MeepleKey=None):
        """
        Check if city has been completed
        
        Inputs:
            - PlayingTile: Tile just played 
            - Surroundings: List of surrounding tiles (left,right,above,below)
            - MeepleUpdate: [p1_Meeple, p2_Meeple] - List of Meeples (e.g [1,0] means P1 is adding a Meeple)
            - MeepleKey:Feature to which Meeple is added
        """

        if PlayingTile.HasCities:
            ClosingCities = []  # initialize list of closing cities
            ClosingCities = cityConnections(self, PlayingTile, Surroundings, ClosingCities, MeepleUpdate, MeepleKey)
            cityClosures(self, ClosingCities)
                             
    def checkRoadCompleteness(self, PlayingTile, Surroundings, MeepleUpdate, MeepleKey=None):            
        """
        Check if road has been completed
            
        Inputs:
            - PlayingTile: Tile just played 
            - Surroundings: List of surrounding tiles (left,right,above,below)
            - MeepleUpdate: [p1_Meeple, p2_Meeple] - List of Meeples (e.g [1,0] means P1 is adding a Meeple)
            - MeepleKey:Feature to which Meeple is added
        """
        if PlayingTile.HasRoads:
            ClosingRoads = [] # initialize list of closing cities
            ClosingRoads = roadConnections(self, PlayingTile, Surroundings, ClosingRoads, MeepleUpdate, MeepleKey)
            roadClosures(self, ClosingRoads)
    
    def checkFarmCompleteness(self, PlayingTile, Surroundings, MeepleUpdate, MeepleKey):
        """
        Check if farm has been completed
        
        Inputs:
            - PlayingTile: Tile just played 
            - Surroundings: List of surrounding tiles (left,right,above,below)
            - MeepleUpdate: [p1_Meeple, p2_Meeple] - List of Meeples (e.g [1,0] means P1 is adding a Meeple)
            - MeepleKey:Feature to which Meeple is added
        """
        if PlayingTile.HasFarms:
            farmConnections(self, PlayingTile, Surroundings, MeepleUpdate, MeepleKey)
                
    def UpdateVirtualScores(self):
        """
        Update virtual scores
        """
        self.Scores[2] = self.Scores[0]
        self.Scores[3] = self.Scores[1]
        # cities
        cityP1 = sum([v.Value for k,v in self.BoardCities.items() if v.Pointer == v.ID and v.Meeples[0] >= v.Meeples[1] and v.Meeples[0] > 0])        
        cityP2 = sum([v.Value for k,v in self.BoardCities.items() if v.Pointer == v.ID and v.Meeples[1] >= v.Meeples[0] and v.Meeples[1] > 0])        
        self.Scores[2] += cityP1
        self.Scores[3] += cityP2
        if self.is_terminal:
            self.FeatureScores[0][3] += cityP1
            self.FeatureScores[1][3] += cityP2
            
        # roads
        roadP1 = sum([v.Value for k,v in self.BoardRoads.items() if v.Pointer == v.ID and v.Meeples[0] >= v.Meeples[1] and v.Meeples[0] > 0])
        roadP2 = sum([v.Value for k,v in self.BoardRoads.items() if v.Pointer == v.ID and v.Meeples[1] >= v.Meeples[0] and v.Meeples[1] > 0])
        self.Scores[2] += roadP1
        self.Scores[3] += roadP2
        if self.is_terminal:
            self.FeatureScores[0][4] += roadP1
            self.FeatureScores[1][4] += roadP2
        
        # monasteries
        monP1 = sum([v.Value for k,v in self.BoardMonasteries.items() if v.Owner == 0])
        monP2 = sum([v.Value for k,v in self.BoardMonasteries.items() if v.Owner == 1])
        self.Scores[2] += monP1
        self.Scores[3] += monP2
        if self.is_terminal:
            self.FeatureScores[0][5] += monP1
            self.FeatureScores[1][5] += monP2
        
        # farms
        farmP1 = 3*sum([len([x for x in v.CityIndexes if self.BoardCities[x].ClosedFlag]) for k,v in self.BoardFarms.items() if v.Pointer == v.ID and v.Meeples[0] >= v.Meeples[1] and v.Meeples[0] > 0])
        farmP2 = 3*sum([len([x for x in v.CityIndexes if self.BoardCities[x].ClosedFlag]) for k,v in self.BoardFarms.items() if v.Pointer == v.ID and v.Meeples[1] >= v.Meeples[0] and v.Meeples[1] > 0])
        self.Scores[2] += farmP1
        self.Scores[3] += farmP2
        if self.is_terminal:
            self.FeatureScores[0][6] += farmP1
            self.FeatureScores[1][6] += farmP2
        #print(f'VIRTUAL POINTS: \nPlayer1: {self.Scores[2]}, Player2: {self.Scores[3]} \n')
        
    def make_action(self, Move = None):
        """
        Place a tile on the game board
        """
        PlayingTileIndex = Move.TileIndex#Move[0]
        X,Y = Move.X, Move.Y#Move[1], Move[2]
        Rotation = Move.Rotation#Move[3]
        MeepleKey = Move.MeepleInfo#Move[4]
        if Move.MeepleInfo is not None:
            self.total_meeples_played[self.player_turn] += 1
            if Move.MeepleInfo[0] == "G":
                self.meeples_on_farms[self.player_turn] += 1
        
        # fix for when the game chooses a tile after the end of game
        if PlayingTileIndex == -1:
            self.EndGameRoutine()
            return
        
        # remove played tile from list of remaining tiles
        self.AvailableSpots.remove((X,Y)) # position of placed tile is no longer available
        PlayingTile = Tile(PlayingTileIndex)
        #this is "discard tile". Not calling the function is faster
        self.deck.remove(PlayingTileIndex) # remove from deck
        self.TileQuantities[PlayingTileIndex] -= 1
        self.TotalTiles -= 1
        self.TileIndexList.remove(PlayingTileIndex) 
        
        #Surroundings analysis
        Surroundings = [None,None,None,None]  # initialization
        SurroundingSpots = [(X-1,Y),(X,Y+1),(X+1,Y),(X,Y-1)]  # left, above, right, below
        
        # check if there is a tile touching the newly placed tile
        for i in range(4):
            if not SurroundingSpots[i] in self.Board:
                self.AvailableSpots.add(SurroundingSpots[i])
            else:
                Surroundings[i] = self.Board[SurroundingSpots[i]]
        
        # add meeple info if one is played
        if not(MeepleKey is None):
            MeepleLocation = PlayingTile.AvailableMeepleLocs[MeepleKey]
            PlayingTile.Meeple = [MeepleKey[0], MeepleLocation, self.player_turn]
            
        # rotate tile to rotation specified and place tile on board
        PlayingTile.Rotate(Rotation)
        
        self.Board[(X,Y)] = PlayingTile  # add to board
        if MeepleKey is None:
            MeepleLoc = [0,0]
            MeepleUpdate = [0,0]
        else:
            player = self.player_turn
            MeepleUpdate = [0,0]
            MeepleUpdate[player] += 1
            self.Meeples[player] -= 1
            
       
        # run logic for each of the game features
        # with new move, it is important to check if any features have been completed
        self.checkMonasteryCompleteness(X,Y,SurroundingSpots, MeepleUpdate, MeepleKey)
        self.checkCityCompleteness(PlayingTile, Surroundings, MeepleUpdate, MeepleKey)
        self.checkRoadCompleteness(PlayingTile, Surroundings, MeepleUpdate, MeepleKey)
        self.checkFarmCompleteness(PlayingTile, Surroundings, MeepleUpdate, MeepleKey)
        
        # update virtual scores
        self.UpdateVirtualScores()
        # check if game is over
        if self.TotalTiles == 0:
            self.EndGameRoutine()
            return
        
        #Turn end routin
        if self.players > 1:
            self.player_turn = 1 - self.player_turn # switch turn
        self.turn += 1  # increment turns

        #Sample random event. Updates available moves
        if self.set_tile_sequence:
            there_are_moves = False
            for tile in self.deck:
                self.available_actions = self.availableMoves(TileIndexOther = tile)
                if self.available_actions != []:
                    there_are_moves = True
                    break
            if not there_are_moves:
                #set back turn change variables
                if self.players > 1:
                    self.player_turn = 1 - self.player_turn
                self.turn -= 1
                #end game
                for tile in self.deck: self.discardTile(tile)
                self.EndGameRoutine()
                return
            self.next_tile_index = tile
        else:
            self.sample_random_event(event_type="Draw random tile")

    def EndGameRoutine(self):
        """
        Logic to handle when game is finished
        Declares self.winner and self.result
        """
        
        # game is finished
        self.is_terminal = True  
        self.UpdateVirtualScores()
        
        # final scores
        self.Scores[0] = self.Scores[2]
        self.Scores[1] = self.Scores[3]
        
        # declare winner
        if self.Scores[0] > self.Scores[1]:
            self.winner = 0 #P1 wins
        elif self.Scores[1] > self.Scores[0]:
            self.winner = 1 #P2 wins
        else:
            self.winner = None #Draw       
            
        self.result = self.Scores[2] - self.Scores[3]

        #Kill variables
        self.available_actions = []
        self.random_events = []

        #Assign rewards
        if self.reward_type == "Score_difference":
            self.reward = [self.result/self.max_possible_score, -self.result/self.max_possible_score]
        elif self.reward_type == "Arbitrary":
            if self.winner == 1:
                self.reward = [self.losing_reward, self.winning_reward]
            elif self.winner == 0:
                self.reward = [self.winning_reward, self.losing_reward]
            else:
                self.reward = [self.draw_reward, self.draw_reward]
        
    def doesTileFit(self, EvaluatedTile, Rotation, SurroundingSpots):
        """
        Checks if the tile ('EvaluatedTile') can be placed with this 'Rotation' 
        and based on 'SurroundingSpots'
        """
        # change values for each rotation value
        SideChange = SIDE_CHANGE_DICT[Rotation]
        R = ROTATION_DICT[Rotation]
        TileLeft,TileUp,TileRight,TileDown = R[0],R[1],R[2],R[3]
        Sides = [TileLeft,TileUp,TileRight,TileDown]
                
        # assume tile fits, need to check if tile does not fit
        IsTileFitting = True
        while True:
            for i in range(4):
                TestingTile = self.Board.get(SurroundingSpots[i])
                if TestingTile is not None:
                    if TestingTile.Properties[(i+2)%4] != EvaluatedTile.Properties[Sides[i]]:
                        IsTileFitting = False
                        break
            break
        return IsTileFitting, SideChange
     
    def movesWithMeeples(self, EvaluatedTile, HasFeature, Openings, 
                         Feature, SideChange, SurroundingSpots, 
                         TempAvailableMoves,TileIndex,X,Y,Rotation):
        """
        Checks which moves are available if the player has Meeples remaining.
        """
        # check if this feature exists in this tile
        if HasFeature:
            if Feature == "G": 
                RotatedOpenings = [[(x+SideChange,y) if x+SideChange < 4 else (x+SideChange-4,y) for (x,y) in k] for k in Openings]
            else:
                RotatedOpenings = [[i+SideChange if i+SideChange < 4 else i+SideChange-4 for i in k] for k in Openings]
            
            for i in range(len(RotatedOpenings)):
                Openings = RotatedOpenings[i]
                MeepleFitting = True
            
                if Feature == "G":                    
                    for (FarmSide,FarmLine) in Openings:
                        if self.Board.get(SurroundingSpots[FarmSide]) is None:
                            pass
                        else:
                            MatchingFeature = self.matchingFeature(EvaluatedTile, self.BoardFarms, Feature, SurroundingSpots, FarmSide, FarmLine)
                            if MatchingFeature.Meeples[0] > 0 or MatchingFeature.Meeples[1] > 0:
                                MeepleFitting = False
                                break    
                else:
                    for Side in Openings:
                        if self.Board.get(SurroundingSpots[Side]) is None:
                            pass
                        else:
                            if Feature == "R":
                                BoardFeature = self.BoardRoads
                            elif Feature == "C":
                                BoardFeature = self.BoardCities
                            MatchingFeature = self.matchingFeature(EvaluatedTile, BoardFeature, Feature, SurroundingSpots, Side)
                            if MatchingFeature.Meeples[0] > 0 or MatchingFeature.Meeples[1] > 0:
                                MeepleFitting = False
                                break
                if MeepleFitting:
                    TempAvailableMoves.append( AvailableMove(TileIndex,X,Y,Rotation,(Feature,i)) )
                                        
        return TempAvailableMoves
                                                 
    def matchingFeature(self, EvaluatedTile, BoardFeature, Feature, SurroundingSpots, Side, FarmLine = None):
        # cities
        if Feature == "C":
            MatchingIndex = self.Board.get(SurroundingSpots[Side]).TileCitiesIndex[self.MatchingSide[Side]]
        #roads 
        elif Feature == "R":
            MatchingIndex = self.Board.get(SurroundingSpots[Side]).TileRoadsIndex[self.MatchingSide[Side]]
        # farms
        else:
            MatchingIndex = self.Board.get(SurroundingSpots[Side]).TileFarmsIndex[self.MatchingSide[Side]][self.MatchingLine[FarmLine]]
        
        while BoardFeature[MatchingIndex].Pointer != BoardFeature[MatchingIndex].ID:                            
                MatchingIndex = BoardFeature[MatchingIndex].Pointer
        MatchingFeature = BoardFeature[MatchingIndex]    
        
        return MatchingFeature
    
    def availableMoves(self, TilesOnly = False, TileIndexOther = None):
        """
        Create a list of all available moves based on the Tile the player just 
        flipped.
        """
        
        # if game is over, return empty list
        if self.is_terminal:
            return []
        
        # tile is the next tile in the deck
        TileIndex = self.next_tile_index if TileIndexOther is None else TileIndexOther
        if TileIndex == -1:
            print(f'\n\n\n(Carcassonne.availableMoves)  No Moves!!  -  TileIndex: {TileIndex}  -  isGameOver: {self.is_terminal}  -  Deck Length: {len(self.deck)}  -  Game Turn: {self.turn} \n\n\n')
            return [AvailableMove(-1,0,0,0,None)]  # game over
            
        # get the tile matching the index
        EvaluatedTile = Tile(TileIndex)
        
        allAvailableMoves = [(self.availableMovesForSpotRotations(Spot, Rotation, EvaluatedTile, TileIndex, TilesOnly)) 
                             for Spot in self.AvailableSpots for Rotation in EvaluatedTile.AvailableRotations] 
         
        allAvailableMoves = list(it.chain.from_iterable(allAvailableMoves))
        
        # check there are available moves
        amountMoves = len(allAvailableMoves)
        if amountMoves == 0:
            if TileIndexOther is not None:
                return []
            self.discardTile(TileIndex)
            allAvailableMoves = self.availableMoves()
        
        # list of all moves
        return allAvailableMoves
    
    def discardTile(self, TileIndex):
        """
        If there are no possible moves for a tile, then it should be discarded
        
        Input:
            - TileIndex: Index of tile to be removed
        """
        # discard tile
        self.deck.remove(TileIndex) # remove from deck
        self.TileQuantities[TileIndex] -= 1
        self.TotalTiles -= 1
        self.TileIndexList.remove(TileIndex)
     
    def availableMovesForSpotRotations(self, Spot, Rotation, EvaluatedTile, TileIndex, TilesOnly):
        """
        Checks available moves for each rotation
        """
        
        # X and Y values of possible spot
        X=Spot[0]
        Y=Spot[1]
        SurroundingSpots = [(X-1,Y),(X,Y+1),(X+1,Y),(X,Y-1)]
        availableMoves =  []
        
        IsTileFitting, SideChange = self.doesTileFit(EvaluatedTile, Rotation, SurroundingSpots)
        #TempAvailableMoves = []
        if IsTileFitting:            
            # add the possible move of adding the tile and placing no meeple
            availableMoves.append(AvailableMove(TileIndex,X,Y,Rotation,None))
            
            
            # now check for possible moves with adding meeples
            if not TilesOnly:
                    # check if current player has any meeples left
                    if self.Meeples[self.player_turn] > 0:
                        
                        # check which features have to be checked for
                        if EvaluatedTile.HasCities:
                            availableMoves = self.movesWithMeeples(EvaluatedTile, EvaluatedTile.HasCities, 
                                                                    EvaluatedTile.CityOpenings, "C", 
                                                                    SideChange, SurroundingSpots,availableMoves,
                                                                    TileIndex,X,Y,Rotation)
                        if EvaluatedTile.HasRoads:
                            availableMoves = self.movesWithMeeples(EvaluatedTile, EvaluatedTile.HasRoads, 
                                                                    EvaluatedTile.RoadOpenings, "R", 
                                                                    SideChange, SurroundingSpots,availableMoves,
                                                                    TileIndex,X,Y,Rotation)
                        if EvaluatedTile.HasFarms and not self.no_farms:
                            availableMoves = self.movesWithMeeples(EvaluatedTile, EvaluatedTile.HasFarms, 
                                                                    EvaluatedTile.FarmOpenings, "G", 
                                                                    SideChange, SurroundingSpots,availableMoves,
                                                                    TileIndex,X,Y,Rotation)
                        # monastery options
                        if EvaluatedTile.HasMonastery:
                            availableMoves.append( AvailableMove(TileIndex,X,Y,Rotation,("Monastery",0)) )
            #print(f'(Carcassonne.availableMovesForRotation) Available Moves: {availableMoves}')
        
        return availableMoves 
    
    def feature_vector(self): #To fix
        fv = {}
        fv["Meeples_p1"] = self.Meeples[0]
        fv["Meeples_p2"] = self.Meeples[1]
        fv["Score_p1"] = self.Scores[0]
        fv["Score_p2"] = self.Scores[1]
        fv["Virtual_score_p1"] = self.Scores[2]
        fv["Virtual_score_p2"] = self.Scores[3]
        fv["Player_turn"] = self.player_turn
        fv["Tile_index"] = self.next_tile_index
        return fv

    def logs_data(self):
        data = self.feature_vector()
        for i, player_reward in enumerate(self.reward):
            data["Reward_p"+str(i)] = player_reward
        data["Turn"] = self.turn
        data["Winner"] = self.winner
        data["Is_terminal"] = self.is_terminal
        data["N_available_actions"] = len(self.available_actions)
        data["Meeples_on_farms_p1"] = self.meeples_on_farms[0]
        data["Meeples_on_farms_p2"] = self.meeples_on_farms[1]
        data["Total_played_meeples_p1"] = self.total_meeples_played[0]
        data["Total_played_meeples_p2"] = self.total_meeples_played[1]
        return pd.DataFrame(data, index=[0])

    def game_definition_data(self):
        data = {"losing_reward":"Score_diff",#self.losing_reward,
                "draw_reward":"Score_diff",#self.draw_reward,
                "winning_reward":"Score_diff",#self.winning_reward,
                "total_tiles":sum(self.initial_tile_quantities),
                "unique_tile_configurations":self.UniqueTilesCount,
                "name":self.name,
                "initial_tile_quantities":str(self.initial_tile_quantities),
                "deck":str(self.deck),
                "initial_meeples": str(self.initial_meeples),
                "max_possible_score":self.max_possible_score,
                "no_farms":self.no_farms,
                "set_tile_sequence":self.set_tile_sequence,
                "reward_type":self.reward_type,
                "set_tile_sequence_seed":self.set_tile_sequence_seed}
        for tile_index, tile_count in enumerate(self.initial_tile_quantities):
            data["tile_"+str(tile_index)+"_count"] = tile_count
        return pd.DataFrame(data, index=[0])

    def sample_random_event(self, event_type):
        """Samples a random event of the game"""
        if event_type == "Draw random tile":
            chosen_tiles = []
            self.available_actions = []
            while self.available_actions == []:
                untested_tile_indexes = [tile_index for tile_index in self.TileIndexList if tile_index not in chosen_tiles]
                if untested_tile_indexes == []:
                    #set back turn change variables
                    if self.players > 1:
                        self.player_turn = 1 - self.player_turn
                    self.turn -= 1
                    #end game
                    for tile in chosen_tiles: self.discardTile(tile)
                    self.EndGameRoutine()
                    return
                else:
                    random_tile = rd.choice(untested_tile_indexes)
                    chosen_tiles.append(random_tile)
                    self.available_actions = self.availableMoves(TileIndexOther = random_tile)
        else:
            raise ValueError("Unknown event type: "+str(event_type))
        
        chosen_tile_probability = self.TileIndexList.count(random_tile)/len(self.TileIndexList)
        random_event = base_games.RandomEvent(id = random_tile, probability = chosen_tile_probability, event_type="Draw random tile")
        self.random_events = [random_event]
        self.next_tile_index = random_tile
        
        return random_event

    def max_score(self):

        #initialise variables
        city_points = 0
        single_city_end = 0
        single_city_values = []
        pizza_city = 0
        pizza_city_values = []
        triple_city = 0
        triple_city_values = []
        quad_city = 0
        quad_city_values = []
        road_points = 0
        monastery_points = 0
        not_closed_city_values = []
        closed_cities = 0

        #iterate tiles
        for i, quantity in enumerate(self.initial_tile_quantities):
            tile = Tile(i)
            if tile.HasCities:
                for city_opening in tile.CityOpenings:
                    if len(city_opening) == 1:
                        single_city_values += [tile.CityValues for _ in range(quantity)]
                    elif len(city_opening) == 2:
                        pizza_city_values += [tile.CityValues for _ in range(quantity)]
                    elif len(city_opening) == 3:
                        triple_city_values += [tile.CityValues for _ in range(quantity)]
                    else:
                        quad_city_values += [tile.CityValues for _ in range(quantity)]
            if tile.HasRoads: #missing: check if pair road ends, or take a meeple
                for road_opening in tile.RoadOpenings:
                    road_points += quantity
            if tile.HasMonastery:
                monastery_points += quantity * 9
        
        #sort score stacks
        sorted_single_city_values = sorted(single_city_values, reverse=True)
        sorted_pizza_city_values = sorted(pizza_city_values, reverse=True)
        sorted_triple_city_values = sorted(triple_city_values, reverse=True)
        sorted_quad_city_values = sorted(quad_city_values, reverse=True)

        #How many single cities are closed?
        closed_cities += int(len(sorted_single_city_values) / 2)
        if len(sorted_single_city_values) % 2 == 1:
            not_closed_city_values.append(sorted_single_city_values.pop(-1))
        city_points += sum(sorted_single_city_values) * 2

        #How many double cities are closed?
        closed_cities += int(len(sorted_pizza_city_values) / 4)
        for _ in range(len(sorted_pizza_city_values) % 4):
            not_closed_city_values.append(sorted_pizza_city_values.pop(-1))
        city_points += sum(sorted_pizza_city_values) * 2

        #Find how many triple cities can be closed with pizzas
        if len(sorted_pizza_city_values) >= 4 and len(sorted_triple_city_values) >= 2:
                if len(sorted_triple_city_values) % 2 == 1:
                    not_closed_city_values.append(sorted_triple_city_values.pop(-1))
                city_points += sum(sorted_triple_city_values) *2
        else:
            not_closed_city_values += sorted_triple_city_values

        #Find how many quad cities can be closed with pizzas
        quads_can_be_closed = int(min(int(len(sorted_pizza_city_values)) / 4 , int(len(sorted_triple_city_values) / 4)))
        quads_cannot_be_closed = int(min(len(sorted_quad_city_values) - quads_can_be_closed,0))
        for _ in range(quads_cannot_be_closed):
            not_closed_city_values += sorted_quad_city_values.pop(-1)
        city_points += sum(sorted_quad_city_values) * 2 

        #Use meeples to score unfinished cities
        max_meeples = max(self.initial_meeples) - 1 #minus one used for a hypothetical unique farm
        city_points += sum(sorted(not_closed_city_values, reverse=True)[:max_meeples])
        
        closed_cities = int(single_city_end / 2 + pizza_city / 4)
        return city_points + road_points + monastery_points + closed_cities * 3

    def __repr__(self):
        #Str = str(self.TileIndexList) + "\n" + str(self.Board) + "\n" + str(self.BoardCities) + "\n" + str(self.BoardRoads) + "\n" + str(self.BoardMonasteries) + "\n" + str(self.BoardFarms)
        Str = "Tiles left:" + str(self.TileIndexList)
        return Str


             
            