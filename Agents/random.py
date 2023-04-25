import random as rd

class RandomPlayer():
    
    player : int = 0

    def choose_action(self, state): #game interface dependencies
        return rd.choice(state.available_actions)