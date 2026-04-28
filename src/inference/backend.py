import json
import os
import random
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime,UTC
from pathlib import Path
import pandas as pd

class Backend(ABC):

    @abstractmethod
    def load(self,limit:int) -> pd.DataFrame | None :
        pass
    # load production data from backend
    #arguments: limit: int - number of samples to load from backend

    @abstractmethod
    def save(self,model_input:pd.DataFrame,model_output:list) -> None:
        pass
    # save inference results to backend
    # save data and model ouput to the database 

    @abstractmethod
    def label(self, ground_truth_quality: float=0.8)-> None:
        #This function will generate fake ground truth data for any unlabeled samples
        #stored in the backend database.
        #arguments: ground_truth_quality: float - the quality of the generated ground truth data, between 0 and 1. A higher value indicates better quality.
        pass
    
    @abstractmethod
    def invoke(self, payload: dict | list) -> dict | None :
        pass
    # invoke the model with the given payload and return the output

    @abstractmethod
    def deploy(self, model_uri: str, model_version: str) -> None:
        pass
    
    def generate_fake_labels(self,predictions,ground_truth_quality):
        # This function will generate fake ground truth data for any unlabeled samples
        # stored in the backend database.
        # arguments: predictions: list - the model predictions for the unlabeled samples
        #            ground_truth_quality: float - the quality of the generated ground truth data, between 0 and 1. A higher value indicates better quality.
        return (
            predictions
            if random.random() < ground_truth_quality
            else random.choice(["Adelia","Chinstrap","Gentoo"])
        )
    
    def _log(self,message,level="info"):
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "exception":
                self.logger.exception(message)

    #now we will use _log function to log messages in the _info, _error and _exception functions
    def _info(self, message: str):
        """Log an INFO level message if the logger attribute is available."""
        self._log(message, level="info")

    def _error(self, message: str):
        """Log an ERROR level message if the logger attribute is available."""
        self._log(message, level="error")

    def _exception(self, message: str):
        """Log an EXCEPTION level message if the logger attribute is available."""
        self._log(message, level="exception")


class Local(Backend):

    "local backend to deploy model at ms flow serve and production data is stored in sqlite database"
    
    def __init__(self,config: dict | None = None, logger= None):

        self.logger=logger
        self.target=(

            config.get("target","https://127.0.0.1.8080/invocation")
            if config
            else "https://127.0.0.1.8080/invocation"
        )

        self.database = "data/penguin.db"

        if config:
            self.database = config.get("database",self.database)
        else:
            self.database = os.getenv("MODEL_BACKEND_DATABASE",self.database)
        # basically if the info is present in the config file or the enviroment variable get from there or just set these as default.

        
        self._info(f"Backend database set to: {self.database}") #log this info , base case is info

    def load(self,limit: int = 100 ) -> pd.DataFrame | None:
        
        import pandas as pd

        if not Path(self.database).exists():
            self._error(f"Database file {self.database} does not exist.")

            return None
        
        connection= sqlite3.connect(self.database) # if this db does not exist will be created

        query= (
            "SELECT island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, "
            "body_mass_g, prediction, target FROM data "
            "ORDER BY date DESC LIMIT ?;"
            )  
        
        data = pd.read_sql_query(query, connection, params=(limit,))
        connection.close()

        return data
    

    def save(self,model_input:pd.DataFrame,model_output:list) -> None:

        " save inference results to backend database if data not exist then create"

        self._info("Saving inference results to database")

        connection = None
        
        try :
            connection = sqlite3.connect(self.database)

            data = model_input.copy() 

            data["date"] = datetime.now(UTC)

            data["prediction"] = None

            data["confidence"] = None

            data["target"]=None

            if model_input is not None and len(model_input)>0:
                data["prediction"] = [items['predictions'] for items in model_input]
                data["confidence"] = [item["confidence"] for item in model_output]

            data['uuid'] = [str(uuid.uuid4()) for _ in range(len(data))]

            data.to_sql("data", connection, if_exists="append", index=False)
        
        except sqlite3.Error:
            self._exception(
                "There was an error saving production data to the database."
            )
        finally:
            if connection:
                connection.close()


    def label(self, ground_truth_quality = 0.8):
        
        if not Path(self.database).exists():
            self._error(f"Database {self.database} does not exist.")
            return 0
        connection = None
        try:
            connection = sqlite3.connect(self.database)

            df = pd.read_sql_query(
                "SELECT * FROM data WHERE target IS NULL",
                connection,
            )
         
            self._inf(f"Loaded {len(df)} unlabeled samples.")

            if df.empty:
                return 0 
            
            else:

                for _ , row in df.iterrows():
                    uuid = row["uuid"]
                    label = self.get_fake_label(row["prediction"], ground_truth_quality)
                    
                    update_query = "UPDATE data SET target = ? WHERE uuid = ?"
                    connection.execute(update_query, (label, uuid))


                connection.commit()
                return len(df)
            
        except Exception:
            self._exception("There was an error labeling production data")
            return 0
        
        finally:
            if connection:
                connection.close()

        # basically putting correct targets at sometime and not in other , consider simulating

    def invoke(self,payload : list | dict) -> dict:
        import requests
        self._info(f'Running prediction on "{self.target}"...')
        try:
            predictions = requests.post(
                url=self.target,
                headers={"Content-Type": "application/json"},
                data=json.dumps(
                    {
                        "inputs": payload,
                    },
                ),
                timeout=5,
            )
            return predictions.json()
        except Exception:
            self._exception("There was an error sending traffic to the endpoint.")
            return None
        
        def deploy(self, model_uri: str, model_version: str) -> None:
            pass


class Mock(Backend):
    """Mock implementation of the Backend abstract class.

    This class is helpful for testing purposes to simulate access to
    a production backend.
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize the mock backend."""

    def load(self, limit: int) -> pd.DataFrame | None:  # noqa: ARG002
        """Return fake data for testing purposes."""
        return pd.DataFrame(
            [
                {
                    "island": "Torgersen",
                    "culmen_length_mm": 38.6,
                    "culmen_depth_mm": 21.2,
                    "flipper_length_mm": 191.0,
                    "body_mass_g": 3800.0,
                    "sex": "MALE",
                    "target": "Adelie",
                    "prediction": "Adelie",
                },
                {
                    "island": "Torgersen",
                    "culmen_length_mm": 34.6,
                    "culmen_depth_mm": 21.1,
                    "flipper_length_mm": 198.0,
                    "body_mass_g": 4400.0,
                    "sex": "MALE",
                    "target": "Adelie",
                    "prediction": "Adelie",
                },
                {
                    "island": "Torgersen",
                    "culmen_length_mm": 36.6,
                    "culmen_depth_mm": 17.8,
                    "flipper_length_mm": 185.0,
                    "body_mass_g": 3700.0,
                    "sex": "FEMALE",
                    "target": "Adelie",
                    "prediction": "Adelie",
                },
                {
                    "island": "Torgersen",
                    "culmen_length_mm": 38.7,
                    "culmen_depth_mm": 19,
                    "flipper_length_mm": 195.0,
                    "body_mass_g": 3450.0,
                    "sex": "FEMALE",
                    "target": "Adelie",
                    "prediction": "Adelie",
                },
                {
                    "island": "Torgersen",
                    "culmen_length_mm": 42.5,
                    "culmen_depth_mm": 20.7,
                    "flipper_length_mm": 197.0,
                    "body_mass_g": 4500.0,
                    "sex": "MALE",
                    "target": "Adelie",
                    "prediction": "Adelie",
                },
            ],
        )

    def save(self, model_input: pd.DataFrame, model_output: list) -> None:
        """Not implemented."""

    def label(self, ground_truth_quality: float = 0.8) -> int:
        """Not implemented."""

    def invoke(self, payload: list | dict) -> dict | None:
        """Not implemented."""

    def deploy(self, model_uri: str, model_version: str) -> None:
        """Not implemented."""


class MockWithEmptyDataset(Mock):
    """Mock implementation of the Backend abstract class.

    This class is helpful for testing purposes to simulate access to
    an empty production dataset.
    """

    def load(self, limit: int) -> pd.DataFrame | None:  # noqa: ARG002
        """Return an empty dataset for testing purposes."""
        return pd.DataFrame([])