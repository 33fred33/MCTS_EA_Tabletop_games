import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List

"""
Rotation -->    0
----------------------------------
|  ul           u            ur  |
|lu                            ru|
|                                |
|                                |
|                                |
|l              m               r|  90
|                                |
|                                |
|                                |
|ld                            rd|
|  dl           d            dr  |
----------------------------------
               180
"""

class Sides(Enum):
    """
    Sides of a quadrangular location on the board
    """
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()

class Directions(Enum):
    """
    Directions where you can move from a location
    """
    DIR0 = auto()
    DIR45 = auto()
    DIR90 = auto()
    DIR135 = auto()
    DIR180 = auto()
    DIR225 = auto()
    DIR270 = auto()
    DIR315 = auto()

class PartialSides(Enum):
    """
    Partial sides of a quadrangular location on the board
    """
    UP_LEFT = auto()
    UP_RIGHT = auto()
    DOWN_LEFT = auto()
    DOWN_RIGHT = auto()
    RIGHT_UP = auto()
    RIGHT_DOWN = auto()
    LEFT_UP = auto()
    LEFT_DOWN = auto()

class Rotations(Enum):
    """
    Available quadrangular rotations
    """
    NULL = auto()
    CLOCKWISE = auto()
    UPSIDE_DOWN = auto()
    ANTICLOCKWISE = auto()

adjacent_sides = {Sides.LEFT : Sides.RIGHT
                ,Sides.RIGHT : Sides.LEFT
                ,Sides.UP : Sides.DOWN
                ,Sides.DOWN : Sides.UP
                ,PartialSides.DOWN_LEFT : PartialSides.UP_LEFT
                ,PartialSides.DOWN_RIGHT : PartialSides.UP_RIGHT
                ,PartialSides.UP_LEFT : PartialSides.DOWN_LEFT
                ,PartialSides.UP_RIGHT : PartialSides.DOWN_RIGHT
                ,PartialSides.LEFT_UP : PartialSides.RIGHT_UP
                ,PartialSides.LEFT_DOWN : PartialSides.RIGHT_DOWN
                ,PartialSides.RIGHT_UP : PartialSides.LEFT_UP
                ,PartialSides.RIGHT_DOWN : PartialSides.LEFT_DOWN
                }

grouped_sides = {Sides.LEFT : (PartialSides.LEFT_UP,PartialSides.LEFT_DOWN)
                ,Sides.RIGHT : (PartialSides.RIGHT_UP,PartialSides.RIGHT_DOWN)
                ,Sides.UP : (PartialSides.UP_LEFT,PartialSides.UP_RIGHT)
                ,Sides.DOWN : (PartialSides.DOWN_LEFT,PartialSides.DOWN_RIGHT)
                }

rotated_sides = {(Sides.LEFT,Rotations.CLOCKWISE) : Sides.UP
                ,(Sides.RIGHT,Rotations.CLOCKWISE) : Sides.DOWN
                ,(Sides.UP,Rotations.CLOCKWISE) : Sides.RIGHT
                ,(Sides.DOWN,Rotations.CLOCKWISE) : Sides.LEFT
                ,(PartialSides.LEFT_DOWN,Rotations.CLOCKWISE): PartialSides.UP_LEFT
                ,(PartialSides.LEFT_UP,Rotations.CLOCKWISE): PartialSides.UP_RIGHT
                ,(PartialSides.RIGHT_DOWN,Rotations.CLOCKWISE): PartialSides.DOWN_LEFT
                ,(PartialSides.RIGHT_UP,Rotations.CLOCKWISE): PartialSides.DOWN_RIGHT
                ,(PartialSides.DOWN_LEFT,Rotations.CLOCKWISE): PartialSides.LEFT_UP
                ,(PartialSides.DOWN_RIGHT,Rotations.CLOCKWISE): PartialSides.LEFT_DOWN
                ,(PartialSides.UP_LEFT,Rotations.CLOCKWISE): PartialSides.RIGHT_UP
                ,(PartialSides.UP_RIGHT,Rotations.CLOCKWISE): PartialSides.RIGHT_DOWN
                ,(Sides.LEFT,Rotations.UPSIDE_DOWN) : Sides.RIGHT
                ,(Sides.RIGHT,Rotations.UPSIDE_DOWN) : Sides.LEFT
                ,(Sides.UP,Rotations.UPSIDE_DOWN) : Sides.DOWN
                ,(Sides.DOWN,Rotations.UPSIDE_DOWN) : Sides.UP
                ,(PartialSides.LEFT_DOWN,Rotations.UPSIDE_DOWN): PartialSides.RIGHT_UP
                ,(PartialSides.LEFT_UP,Rotations.UPSIDE_DOWN): PartialSides.RIGHT_DOWN
                ,(PartialSides.RIGHT_DOWN,Rotations.UPSIDE_DOWN): PartialSides.LEFT_UP
                ,(PartialSides.RIGHT_UP,Rotations.UPSIDE_DOWN): PartialSides.LEFT_DOWN
                ,(PartialSides.DOWN_LEFT,Rotations.UPSIDE_DOWN): PartialSides.UP_RIGHT
                ,(PartialSides.DOWN_RIGHT,Rotations.UPSIDE_DOWN): PartialSides.UP_LEFT
                ,(PartialSides.UP_LEFT,Rotations.UPSIDE_DOWN): PartialSides.DOWN_RIGHT
                ,(PartialSides.UP_RIGHT,Rotations.UPSIDE_DOWN): PartialSides.DOWN_LEFT
                ,(Sides.LEFT,Rotations.ANTICLOCKWISE) : Sides.DOWN
                ,(Sides.RIGHT,Rotations.ANTICLOCKWISE) : Sides.UP
                ,(Sides.UP,Rotations.ANTICLOCKWISE) : Sides.LEFT
                ,(Sides.DOWN,Rotations.ANTICLOCKWISE) : Sides.RIGHT
                ,(PartialSides.LEFT_DOWN,Rotations.ANTICLOCKWISE): PartialSides.DOWN_RIGHT
                ,(PartialSides.LEFT_UP,Rotations.ANTICLOCKWISE): PartialSides.DOWN_LEFT
                ,(PartialSides.RIGHT_DOWN,Rotations.ANTICLOCKWISE): PartialSides.UP_RIGHT
                ,(PartialSides.RIGHT_UP,Rotations.ANTICLOCKWISE): PartialSides.UP_LEFT
                ,(PartialSides.DOWN_LEFT,Rotations.ANTICLOCKWISE): PartialSides.RIGHT_DOWN
                ,(PartialSides.DOWN_RIGHT,Rotations.ANTICLOCKWISE): PartialSides.RIGHT_UP
                ,(PartialSides.UP_LEFT,Rotations.ANTICLOCKWISE): PartialSides.LEFT_DOWN
                ,(PartialSides.UP_RIGHT,Rotations.ANTICLOCKWISE): PartialSides.LEFT_UP
            }

orthogonal_directions = (Directions.DIR0, Directions.DIR90, Directions.DIR180, Directions.DIR270)
diagonal_directions = (Directions.DIR45, Directions.DIR135, Directions.DIR225, Directions.DIR315)

class Location():
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coordinates = (x,y)
        self.surrounding_coordinates = {
                                    Directions.DIR0 : (x,y+1)
                                    ,Directions.DIR45 : (x+1,y+1)
                                    ,Directions.DIR90 : (x+1,y)
                                    ,Directions.DIR135 : (x+1,y-1)
                                    ,Directions.DIR180 : (x,y-1)
                                    ,Directions.DIR225 : (x-1,y-1)
                                    ,Directions.DIR270 : (x-1,y)
                                    ,Directions.DIR315 : (x-1,y+1)
                                    }
        self.content = {}

    def update_content(self, content) -> None:
        """
        Updates "content" variable in the Location class
        Args:
            content: (dicitonary) {side:required_value}
        """
        self.content.update(content)

    def duplicate(self):
        """
        Creates an independent duplicate snapshot of the class
        Returns:
            Location instance
        """
        #Crete the instance
        the_duplicate = Location(self.x, self.y)
        the_duplicate.content = {}
        for k,v in self.content.items():
            if hasattr(v,"duplicate"):
                the_duplicate.content[k] = v.duplicate()
            else:
                the_duplicate.content[k] = v

        return the_duplicate

    def __str__(self):
        return str(self.coordinates) + ": " + str({k:str(v) for k,v in self.content.items()})

class Board():

    filled_locations : dict
    available_locations : dict  
    limits : dict

    def __init__(self, x_min=-np.inf, x_max=np.inf, y_min=-np.inf, y_max=np.inf):

        self.filled_locations = {} #Keys are tuples of ints (coordinates)
        self.available_locations = {}
        self.limits = {"x-":x_min, "x+":x_max, "y-":y_min, "y+":y_max}           

    def create_empty_squared_board(self, width, height, start_x=0, start_y=0) -> None:

        for i in range(width):
            for j in range(height):
                x = start_x + i
                y = start_y + j
                self.available_locations[(x,y)] = self.make_location_available(x, y)

    def fill_location(self, location) -> None:
        """
        Updates "filled_locations" and "available_locations" in the board class
        Args:
            location: (Location instance)
        """

        self.available_locations.pop(location.coordinates)
        self.filled_locations[location.coordinates] = location

    def make_location_available(self, x, y) -> Location:
        """
        Updates "available_locations" variable in the board class
        Args:
            x: (int)
            y: (int)
        """
        if x >= self.limits["x-"] and x <= self.limits["x+"] and y >= self.limits["y-"] and y <= self.limits["y+"]:
            new_loc = Location(x,y)
            self.available_locations[(x,y)] = new_loc
            return new_loc
        return None

    def duplicate(self):
        """
        Creates an independent duplicate snapshot of the class
        Returns:
            Board instance
        """
        #Crete the instance
        the_duplicate = Board()

        #Creating independent copies of every relevant variable in the class
        the_duplicate.available_locations = {k:v.duplicate() for k,v in self.available_locations.items()}
        the_duplicate.filled_locations = {k:v.duplicate() for k,v in self.filled_locations.items()}
        the_duplicate.limits = {k:v for k,v in self.limits.items()}

        return the_duplicate

    def __str__(self):
        string = "Board: "
        for coordinates, location in self.filled_locations.items():
            string = string + str(coordinates) + str(location.content)
        return string
