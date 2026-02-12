import json
import random
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime,UTC
from pathlib import Path
import pandas as pd

class Backend(ABC):

    @abstractmethod
    def load(self,limit:int):
        pass
    # load production data from backend
    #arguments: limit: int - number of samples to load from backend

    @abstractmethod
    def save(self,model_input:pd.DataFrame,model_output:list):
        pass
    # save inference results to backend
    # save data and model ouput to the database 

    @abstractmethod
    def label(self, ground_truth_quality: float=0.8):
        pass
