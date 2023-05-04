from typing import Protocol
from enum import Enum, auto
from typing import List
from Games.base_games import BaseGameState
import pandas as pd


class BaseAgent(Protocol):

    name : str
    choose_action_logs : pd.DataFrame
    logs : bool
    def choose_action(self, state: BaseGameState) -> None: raise NotImplementedError