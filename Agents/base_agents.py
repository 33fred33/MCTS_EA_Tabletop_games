from typing import Protocol
from enum import Enum, auto
from typing import List
from Games.base_games import BaseGameState


class BaseAgent(Protocol):

    def choose_action(self, state: BaseGameState) -> None: raise NotImplementedError