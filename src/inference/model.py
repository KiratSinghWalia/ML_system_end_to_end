import importlib
import json
import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import pydantic
from mlflow.models import set_model
from mlflow.pyfunc.model import PythonModelContext

class Input(pydantic.BaseModel):
    island: str | None = None
    culmen_length_mm: float | None = None
    culmen_depth_mm: float | None = None
    flipper_length_mm: float | None = None
    body_mass_g: float | None = None
    sex: str | None = None


class Output(pydantic.BaseModel):
    prediction: str | None = None
    confidence: float | None = None

class Model(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.backend = None

    def load_context(self, context :PythonModelContext | None) -> None:
        self._configure_logging()
        self._initialize_backend()
        self._load_artifacs(context)
        self.logger.info("Model is ready to receive requests")
        #uses context to initialize the model

    def predict(self, context, model_input : list[Input], params: dict[str, Any] | None = None) -> Output:

        model_input = pd.DataFrame([sample.model_dump() for sample in model_input])
        #use pydantic to make a list of the dicts and then convert to dataframe
        #from model_input(abc->,xyz->) to [{abc:,xyz:}]

        if model_input.empty:
            self.logging.warning("Received an empty request")
            return []
        
        self.logger.info(
            "Received prediction request with %d %s",
            len(model_input),
            "samples" if len(model_input) > 1 else "sample",
        )
        model_output = []
        transformed_payload = self.process_input(model_input)
        #need to transform the model same way we did in training period
        if transformed_payload is not None:
            self.logger.info("Making a prediction using the transformed payload...")
            predictions = self.model.predict(transformed_payload, verbose=0)

            model_output = self.process_output(predictions)

        if self.backend is not None:
            self.backend.save(model_input, model_output)

        self.logger.info("Returning prediction to the client")
        self.logger.debug("%s", model_output)

        return model_output

    def process_input(self, payload: pd.DataFrame) -> pd.DataFrame | None:
        self.logger.info("Transforming payload...")
        # convert input data to a format which can be handeled by the model

        try:
            result = self.features_transformer.transform(payload)
        except Exception:
            self.logger.exception("There was an error processing the payload.")
            return None

        return result

    def process_output(self,output:np.ndarray) -> list[dict[str, Any]]:
        self.logger.info("Processing prediction received from the model...")
        result=[]
        if output is not None:
            prediction = np.argmax(output, axis=1)
            confidence = np.max(output, axis=1)
            #prediction get you the index number
            #conf gets you logits

            classes = self.target_transformer.named_transformers_[
                "species"
            ].categories_[0]
            prediction = np.vectorize(lambda x: classes[x])(prediction)

            #output = [1,2] -> ["gento","adeliat"]

            result = [{"prediction": p.item(), "confidence": c.item()} #.item to get out of np.array
                for p, c in zip(prediction, confidence, strict=True)]
        return result
    
    def _initialize_backend(self):
        with suppress(ImportError):
            import inference.backend

        self.logger.info("Initializing model backend...")
        backend_class = os.getenv("MODEL_BACKEND", "backend.Local")

        if backend_class is not None:
            # We can optionally load a JSON configuration file and use it to initialize
            # the backend instance.
            backend_config = os.getenv("MODEL_BACKEND_CONFIG", None)

            try:
                if backend_config is not None:
                    backend_config = Path(backend_config)
                    backend_config = (
                        json.loads(backend_config.read_text())
                        if backend_config.exists()
                        else None
                    )

                    module , cls = backend_class.rsplit(".",1)
                    module = importlib.import_module(module)
                    self.backend = getattr(module,cls)(config=backend_config)

            except Exception:
                self.logger.exception(
                    'There was an error initializing backend "%s".',
                    backend_class,
                )

        self.logger.info("Backend: %s", backend_class if self.backend else None)

    def _load_artifacts(self,context:PythonModelContext | None):
        if context is None:
            self.logger.warning("No model context was provided.")
            return  
        
        if not os.getenv("KERAS_BACKEND"):
            os.environ["KERAS_BACKEND"] = "tensorflow"

        import keras

        self.features_transformer = joblib.load(
            context.artifacts["features_transformer"],
        )
        self.target_transformer = joblib.load(context.artifacts["target_transformer"])

        # Then, we can load the Keras model we trained.
        self.model = keras.saving.load_model(context.artifacts["model"])

    def _configure_logging(self):
        """Configure how the logging system will behave."""
        import sys

        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )

        self.logger = logging.getLogger("model")


set_model(Model())