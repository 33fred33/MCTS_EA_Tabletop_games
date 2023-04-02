from collections import defaultdict
import Games.board_2d as b2d
import operator
import random as rd
from typing import List, Tuple
from dataclasses import dataclass
import Games.base_games as base_games

#Game Othello

@dataclass
class Action(base_games.BaseAction):
    
    coordinates : Tuple[int]
    content : int
    swaps : List[Tuple[int]]

    def duplicate(self):
        return Action(self.coordinates, self.content, [x for x in self.swaps])

    def __str__(self):
        return str(self.content) + ", " + str(self.coordinates) + ", swaps:" + str(len(self.swaps))

    def __eq__(self, other):
        return other.coordinates == self.coordinates and other.content == self.content and other.swaps == self.swaps

    #def __hash__(self):
    #    return hash((self.coordinates, self.content, self.swaps))

    def __ne__(self, other):
        return not(self == other)

class GameState(base_games.BaseGameState):

    
    ply : int
    board : b2d.Board
    content_name : str = "P"
    width : int
    height : int
    available_actions : List[Action] = []
    make_action_calls : int = 0
    player_turn : int = 0
    
    def set_initial_state(self, width=8, height=8) -> None:

        self.width = width
        self.height = height
        self.board = b2d.Board(x_min=0, x_max=width-1, y_min=0, y_max=height-1)
        for x in (int(width/2)-1, int(width/2)):
            for y in (int(height/2)-1, int(height/2)):
                if (x+y)%2==0: 
                    content = 0
                else: 
                    content = 1
                location = self.board.make_location_available(x, y)
                location.update_content({"P":content})
                self.board.fill_location(location)

        for location in self.board.filled_locations.values():
            self._find_new_available_locations(location)
        
        self.ply = 0
        self.player_turn = 0
        self.available_actions = self._get_available_actions()

    def _get_enemy_directions(self, location) -> List[int]:

        enemy_directions = []
        for direction in b2d.directions:
            adjacent_coordinate = location.surrounding_coordinates[direction]
            if adjacent_coordinate in self.board.filled_locations:
                if self.board.filled_locations[adjacent_coordinate].content[self.content_name] != self.player_turn:
                    enemy_directions.append(direction)

        return enemy_directions

    def _get_available_actions(self) -> List[Action]:

        actions = []
        for location in self.board.available_locations.values():
            #enemy_directions = self._get_enemy_directions(location)
            swaps = []
            for direction in b2d.directions:
                temp_location = location
                outflanked_locations = []
                while temp_location.surrounding_coordinates[direction] in self.board.filled_locations:
                    next_location = self.board.filled_locations[temp_location.surrounding_coordinates[direction]]
                    if next_location.content[self.content_name] == self.player_turn:
                        swaps = swaps + outflanked_locations
                        break
                    else:
                        outflanked_locations.append(next_location.coordinates)
                        temp_location = next_location
            if len(swaps) > 0: 
                actions.append(Action(coordinates = location.coordinates, content = self.player_turn, swaps = swaps))
            
        return actions

    def _find_new_available_locations(self, location) -> None:

        for direction in b2d.directions:
            adjacent_coordinate = location.surrounding_coordinates[direction]
            if adjacent_coordinate not in self.board.filled_locations and adjacent_coordinate not in self.board.available_locations:
                self.board.make_location_available(adjacent_coordinate[0], adjacent_coordinate[1])

    def make_action(self, action) -> None:
        
        self.make_action_calls = self.make_action_calls + 1

        location = self.board.available_locations[action.coordinates]
        self.board.fill_location(location)
        location.update_content({self.content_name : action.content})
        self._find_new_available_locations(location)

        #Swaps
        for swap_coordinate in action.swaps:
            location_to_swap = self.board.filled_locations[swap_coordinate]
            new_content = base_games.next_player(location_to_swap.content[self.content_name])
            location_to_swap.update_content({self.content_name:new_content})
        
        #Variables update
        self.player_turn = base_games.next_player(self.player_turn)
        self.available_actions = self._get_available_actions()
        if len(self.available_actions) > 0:
            self.ply = self.ply + 1
        else: 
            if len(self.board.available_locations) > 0:
                self.player_turn = base_games.next_player(self.player_turn)
                self.available_actions = self._get_available_actions()
        
    def view_game_state(self) -> None:
        top_string = "   " + "-"*(self.width*2 + 2)
        print(top_string)
        for y in reversed(range(self.height)):
            string = f"{y} |"
            for x in range(self.width):
                coordinates = (x,y)
                if coordinates in self.board.filled_locations:
                    filling = self.board.filled_locations[coordinates].content[self.content_name]
                    if filling == 0:
                        string += " X"
                    else: string += " O"
                else:
                    string += " ."
            print(string + " |")
        print(top_string)
        last_string = "    "
        for x in range(self.width): last_string += f"{x} "
        print(last_string)

    def is_terminal(self) -> bool:

        if len(self.board.available_locations) == 0:
            return True
        elif len(self.available_actions) == 0:
            return True
        return False

    def winner(self) -> int:

        if not self.is_terminal():
            return None
        else:
            scores = self.scores()
            return max(scores, key=scores.get)

    def scores(self) -> dict:

        scores = {0:0,1:0}
        for location in self.board.filled_locations.values():
            scores[location.content[self.content_name]] = scores[location.content[self.content_name]] + 1
        return scores

    def duplicate(self):

        the_duplicate = GameState()
        the_duplicate.player_turn = self.player_turn
        the_duplicate.ply = self.ply
        the_duplicate.board = self.board.duplicate()
        the_duplicate.content_name = self.content_name
        the_duplicate.width = self.width
        the_duplicate.height = self.height
        the_duplicate.available_actions = [a.duplicate() for a in self.available_actions]
        return the_duplicate

    def __eq__(self, other):
        return other.board == self.board and other.player_turn == self.player_turn

    #def __hash__(self):
    #    return hash((self.board, self.player_turn))

    def __ne__(self, other):
        return not(self == other)