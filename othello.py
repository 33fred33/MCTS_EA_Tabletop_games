from collections import defaultdict
import board_2d as b2d
import operator
import random as rd
from enum import Enum, auto
from typing import List
from dataclasses import dataclass

#Game Othello

class Players(Enum):

    PLAYER1 = auto()
    PLAYER2 = auto()
    
    def succ(self):

        if self == Players.PLAYER1:
            return Players.PLAYER2
        else:
            return Players.PLAYER1

@dataclass
class Action():
    
    location : b2d.Location
    content : Players
    swaps : List[b2d.Location]

    def __str__(self):
        return str(self.content) + ", " + str(self.location.coordinates) + ", swaps:" + str(len(self.swaps))

class GameState():

    player_turn : Players
    ply : int
    board : b2d.Board

    def __init__(self):
        pass
    
    def set_initial_state(self, width=8, height=8) -> None:

        self.board = b2d.Board()
        for x in (int(width/2)-1, int(width/2)):
            for y in (int(height/2)-1, int(height/2)):
                if (x+y)%2==0: 
                    content = Players.PLAYER1
                else: 
                    content = Players.PLAYER2
                location = self.board.make_location_available(x, y)
                location.update_content({"P":content})
                self.board.fill_location(location)

        for location in self.board.filled_locations.values():
            for direction in b2d.Directions:
                adjacent_coordinate = location.surrounding_coordinates[direction]
                if adjacent_coordinate not in self.board.filled_locations and adjacent_coordinate not in self.board.available_locations:
                    self.board.make_location_available(adjacent_coordinate[0], adjacent_coordinate[1])
        
        self.ply = 0
        self.player_turn = Players.PLAYER1

    def _get_enemy_directions(self, location) -> List[b2d.Directions]:

        enemy_directions = []
        for direction in b2d.Directions:
            adjacent_coordinate = location.surrounding_coordinates[direction]
            if adjacent_coordinate in self.board.filled_locations:
                if self.board.filled_locations[adjacent_coordinate].content["P"] != self.player_turn:
                    enemy_directions.append(direction)

        return enemy_directions

    def _get_outflanked_locations(self, location, direction, content_to_stop, outflanked_locations=[], outflanked = False):

        if location.surrounding_coordinates[direction] in self.board.filled_locations:
            next_location = self.board.filled_locations[location.surrounding_coordinates[direction]]
            if next_location.content["P"] == content_to_stop:
                return True, outflanked_locations
            else:
                outflanked, locations = self._get_outflanked_locations(next_location, direction, content_to_stop)
                if outflanked:
                    outflanked_locations = outflanked_locations + [next_location]
                else:
                    outflanked_locations = []
        return outflanked, outflanked_locations

    def get_available_actions(self) -> List[Action]:

        actions = []
        for location in self.board.available_locations.values():
            enemy_directions = self._get_enemy_directions(location)
            swaps = []
            outflanked_locations = []
            for direction in enemy_directions:
                
                _, outflanked_locations = self._get_outflanked_locations(location, direction, self.player_turn)
                swaps = swaps + outflanked_locations
            if len(swaps) > 0: 
                actions.append(Action(location = location, content = self.player_turn, swaps = swaps))
            
        return actions

    def duplicate(self):

        the_duplicate = GameState()
        the_duplicate.player_turn = self.player_turn
        the_duplicate.ply = self.ply
        the_duplicate.board = self.board.duplicate()