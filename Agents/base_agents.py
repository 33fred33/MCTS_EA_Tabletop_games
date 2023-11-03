from typing_extensions import Protocol
from enum import Enum, auto
from typing import List
from Games.base_games import BaseGameState
import pandas as pd


class BaseAgent(Protocol):

    name : str
    choose_action_logs : pd.DataFrame
    logs : bool
    all_my_logs : List[pd.DataFrame]
    def choose_action(self, state: BaseGameState) -> None: raise NotImplementedError
    def dump_my_logs(self, file_path, file_name): raise NotImplementedError