
from typing import List, Tuple
import numpy as np
import Games.base_games as base_games
import pandas as pd

class Meeple():
    def __init__(self, owner_id):
        self.meeple_owner_id = owner_id
        
        

class Feature():
    def __init__(self, index:int, feature_type:str, tiles_extension:int = 1, meeples:List[Meeple] = [], shields:int = 0, contact_cities = [], openings = []):
        """
        feature_type: str in road, city, monastery, field
        """
        self.index = index
        self.feature_type = feature_type
        self.tiles_extension = tiles_extension
        self.meeples = meeples
        self.is_completed = False
        self.shields = shields
        self.contact_cities = contact_cities
        self.openings = openings
    
    def add_meeple(self, meeple):
        self.meeples.append(meeple)

    def add_contact_city(self, city):
        self.contact_cities.append(city)
    
    def complete(self):
        self.is_completed = True

    def score(self, end_of_game=False, adjacent_tile_count = 0):
        if self.feature_type == "road":
            return self.tiles_extension
        elif self.feature_type == "city":
            if end_of_game or self.is_completed:
                return (self.tiles_extension + self.shields) * 2
            else:
                return self.tiles_extension + self.shields
        elif self.feature_type == "monastery":
            return self.tiles_extension + adjacent_tile_count
        elif self.feature_type == "field":
            return 3 * len([city for city in self.contact_cities if city.is_completed])


class Action():

    x : int
    y : int
    features : List[Feature]

    def __init__(self, player_index, x, y):
        self.player_index = player_index
        self.x = x
        self.y = y

    def __str__(self):
        return f"(x{self.x},y{self.y},t{self.player_index})"
    
    def __members(self):
        return (self.x, self.y, self.player_index)
    
    def __eq__(self, other):
        return self.__members() == other.__members()

    def __hash__(self):
        return hash(self.__members())
    
