import os
import time
import sys
import re
import importlib

from contextlib import suppress
from pathlib import Path


import yaml
import pandas as pd
from metaflow import (
    Config,
    FlowMutator,
    FlowSpec,
    Parameter,
    config_expr,
    current,
    project,
    user_step_decorator,
)

@user_step_decorator
def dataset(step_name,flow,inputs=None,attr=None):
    #Transforming the data
    #flow basically - which pipeline
    import numpy as np
    if not Path(flow.dataset).exists():
        #will search the flow for a dataset artifact 
        flow.data=None
        yield
    
    else:
        data=pd.read_csv(flow.dataset)

        data["sex"]=data["sex"].replace({".",np.nan})

        #shuffle the data 
        seed=int(time.time()*1000) if current.is_production else 42 #using curent from metaflow to check if we are in production and set seed accordingly
        generator=np.random.default_rng(seed)
        data=data.sample(frac=1,random_state=generator)#using pandas sample to shuffle the data
        #using logger from metaflow to log info
        flow.logger.info("Loaded data with %d samples",len(data))
        flow.data=data
        yield

@user_step_decorator
def logging(step_name,flow,inputs=None,attr=None):
        
    import logging
    import logging.config

    # Let's get the logging configuration file from the project settings.
    logging_file = flow.project.get("logging", "logging.conf")
    # If the logging configuration file exists else we will use basic configuration , basically logging_file = your config file path
    
    if Path(logging_file).exists():
        #Now if the file in your directory exists we will use that file for logging configuration
        logging.config.fileConfig(logging_file)
        #will go into the file path and configure logging as per the file
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )
    # if path exists use that file else make a basic config

    flow.logger = logging.getLogger("main_application")
    yield # at yeild code will flow from decorator to the step function.
        



@user_step_decorator
def mlflow(step_name,flow,inputs=None,attr=None):
    import mlflow

    mlflow.set_tracking_uri(flow.mlflow_tracking_uri)
    yield

@user_step_decorator
def backend(step_name,flow,inputs=None,attr=None):
    
    with suppress(ImportError):
        import inference.backend

    try:
        module,cls= flow.backend.rsplit(".",1)
        modeule = importlib.import_module(module)

        backend_impl=getattr(module,cls)(
            Config=flow.project.backend, #get from pipeline config
            logger=flow.logger) #get from pipeline logger
        #inferecence backend file and loading the class dynamically.
    except Exception as e:
        message = f"There was an error instantiating class {flow.backend}"
        flow.logger.exception(message)
        raise RuntimeError(message) from e
    
    else:
        flow.logger.info("Backend: %s", flow.backend)
        flow.backend_impl = backend_impl
        yield

def parse_project_configuration(yamlfile):

    config= yaml.full_load(yamlfile)

    if "mlflow_tracking_uri" not in config:
        config["mlflow_tracking_uri"]= os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")
        #checking if mlflow_tracking_uri is in config else set to env variable or default localhost

    if "backend" not in config:
        config["backend"]={"modeule":"backend.local"}

        #we need config as a dictionary to set default values if not present in yaml file

    pattern = re.compile(r"\$\{(\w+)\}") #regex pattern to match ${VAR_NAME}

    def replacer(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0)) #return env variable value or original string if not found
    
    for key, value in config["backend"].items():
        if isinstance(value, str):#check if value is string 
            config["backend"][key] = pattern.sub(replacer, value) #replace ${VAR_NAME} with env variable value

    return config

class pipeline(FlowMutator):
    def mutate(self,mutable_flow):

        for _ , step in mutable_flow.steps:
            
            step.add_decorator(logging,duplicates=step.IGNORE)

            step.add_decorator(mlflow,duplicates=step.IGNORE)


@pipeline
@project(name=config_expr("project.project"))
class Pipeline(FlowSpec):

    #first step parse the project configuration file and make a config object 
    project = Config(
        "project",
        help="Project configuration settings.",
        default="config/local.yml",
        parser=parse_project_configuration,
    )

    backend = Parameter(
        "backend",
        help="Backend module implementation.",
        default=project.backend["module"],
    )

    dataset = Parameter(
        "dataset",
        help="Project dataset that will be used to train and evaluate the model.",
        default="data/penguins.csv",
    )

    mlflow_tracking_uri = Parameter(
        "mlflow-tracking-uri",
        help="MLflow tracking URI.",
        default=project.mlflow_tracking_uri,
    )



   