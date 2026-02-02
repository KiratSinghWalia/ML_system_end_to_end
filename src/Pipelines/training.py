import os
from pathlib import Path

from metaflow import Parameter,card,current,step,environment


from src.common.pipeline import Pipeline, dataset

environment_variables = {
    "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "tensorflow"),
}


def build_features_transformer():

    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler


    numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
    )

    return ColumnTransformer(
        transformers=[("numeric", numeric_transformer, make_column_selector(dtype_include="object")),
                        ("categorical", categorical_transformer, ["island","sex"])])


def build_target_transformer():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    return ColumnTransformer(
        transformers=[("species", OrdinalEncoder(), ["species"])],
    )
    # giving the targets ordinial values 0,1,2 for the 3 species , MLP will use sparse_categorical_crossentropy


def build_model(input_shape,learning_rate=0.01):
    from keras import Input, layers, models, optimizers

    model=models.Sequential(
        [
            Input(shape=(input_shape,)),
            layers.Dense(10,activation="relu"),
            layers.Dense(8,activation="relu"),
            layers.Dense(3,activation="softmax"),

        ]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


class Training(Pipeline):
    training_epochs = Parameter(
        "training-epochs",
        help="Number of epochs to train the model.",
        default=50,
    )

    training_batch_size = Parameter(
        "training-batch-size",
        help="Batch size for training the model.",
        default=32,
    )

    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help="Accuracy threshold for model evaluation.",
        default=0.85,
    )   



    @dataset
    @card
    @step

    def start(self):
        " Start the traiining pipeline. "

        import mlflow

        self.logging.info("MLflow Tracking URI: %s", mlflow.get_tracking_uri())
        self.mode ="production" if current.is_production else "development"

        self.logging.info("Pipeline started in %s mode.", self.mode)

        try:

            run = mlflow.start_run(run_name=current.run_id)
            #using metaflow run id as mlflow run name by using current object
            self.mlflow_run_id = run.info.run_id
            # store mlflow run id in self object for later use

        except Exception as e:
            message= f"Failed to start MLflow run: {self.mlflow_run_id}"
            raise RuntimeError(message) from e
        
        self.next(self.cross_validation,self.transform)

    @card
    @step

    def cross_validation(self):
        from sklearn.model_selection import KFold

        kflod= KFold(n_splits=5, shuffle=True)

        self.folds=list(enumerate(kflod.split(self.data)))
        
        self.next(self.transform_fold,foreach="folds")

    @card
    @step

    def transform_fold(self):

        self.fold,(self.train_indices,self.test_indices)=self.input
        self.logger.info("Transforming fold %d ..",self.fold)

        train_data=self.data.iloc[self.train_indices]

        test_data=self.data.iloc[self.test_indices]

        feature_transformation=build_features_transformer()
        self.x_train=feature_transformation.fit_transform(train_data)
        self.x_test=feature_transformation.transform(test_data)

        target_transformer = build_target_transformer()
        self.y_train=target_transformer.fit_transform(train_data)
        self.y_test=target_transformer.transform(test_data)

        self.next(self.train_fold)

    @card
    @environment(vars=environment_variables) #important to set it , cuz when running in diiferent env , metaflow must know
    @step

    def train_fold(self):

        import mlflow

        self.logger.info("Training model for fold %d ..",self.fold)
        #giving logging info for the respective fold being trained now we will
        #configure a nested mlflow run for each fold under a parent run 

        with (mlflow.start_run(run_id=self.mlflow_run_id),
              mlflow.start_run(
                  run_name=f"cross val fold {self.fold}",nested=True) as run ,):
            
            self.mlflow_run_id=run.info.run_id # update mlflow run id to current fold run id
            #first it was the parent run id now it is the fold run id .

            mlflow.autolog(log_models=False)

            self.model = build_model(self.x_train.shape[1]) #.shape[1] gives the number of features

            history = self.model.fit(
                self.x_train,
                self.y_train,
                epochs=self.training_epochs,
                batch_size=self.training_batch_size,
                verbose=0,
            )

        self.logging.info(
            "Fold %d - training loss: %.4f , training accuracy: %.4f",
            self.fold,
            history.history["loss"][-1],
            history.history["accuracy"][-1],
        )

        self.next(self.evaluate_fold)


    @card
    @environment(vars=environment_variables)
    @step

    def evaluate_fold(self):
        #evaluating and logging metrics for the each fold

        import mlflow

        self.logger.info("Evaluating fold %d...", self.fold)

        self.test_loss, self.test_accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=0,
        )

        self.logger.info(
            "Fold %d - test_loss: %f - test_accuracy: %f",
            self.fold,
            self.test_loss,
            self.test_accuracy,
        )

        mlflow.log_metric(
            {
                "test_loss": self.test_loss,
                "test_accuracy": self.test_accuracy,
            }, run_id=self.mlflow_run_id
        )


        self.next(self.average_scores)

    @card
    @step

    def average_scores(self,inputs):
        import mlflow
        import numpy as np

        self.merge_artifacts(inputs,include=["mlflow_run_id"])#merging artifacts from all folds

        metrics=[[input.test_accuracy,input.test_loss] for input in inputs] #iterating over inputs to get test accuracy and loss for each fold

        self.test_accuracy,self.test_loss=np.mean(metrics,axis=0) #calculating mean across folds
        self.test_accuracy_std,self.test_loss_std=np.std(metrics,axis=0) #calculating std deviation across folds

        mlflow.log_metrics( #logging average metrics to the parent mlflow run
            {"test_accuracy": self.test_accuracy,
             "test_loss": self.test_loss,
             "test_accuracy_std": self.test_accuracy_std,
             "test_loss_std": self.test_loss_std,},
            run_id=self.mlflow_run_id,

        )

        self.next(self.register)

    @card
    @step   

    def transfom(self): #feature and target transformation for the entire dataset

        self.features_transformer=build_features_transformer()
        self.target_transformer=build_target_transformer()

        self.x=self.features_transformer.fit_transform(self.data)
        self.y=self.target_transformer.fit_transform(self.data)

        self.next(self.train)


    @card
    @environment(vars=environment_variables)
    @step

    def train(self,inputs):
        import mlflow
        
        self.logger.info("Training final model on entire dataset...")

        with mlflow.start_run(run_id=self.mlflow_run_id) as run:
            mlflow.autolog(log_models=False)

            self.model=build_model(self.x.shape[1])

            history=self.model.fit(
                self.x,
                self.y,
                epochs=self.training_epochs,
                batch_size=self.training_batch_size,
                verbose=2,
            )
    
        self.next(self.register)

    @environment(vars=environment_variables)
    @step

    def register(self,inputs):
        import mlflow
        import tempfile

        self.merge_artifacts(inputs)
        if self.test_accuracy >= self.accuracy_threshold:
            self.logger.info("Registering model ...")

            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                self.artifacts= self._get_model_artifacts(directory)
                self.pip_requirements=self._get_model_pip_requirements()

                root = Path(__file__).parent.parent #parent of parent is src/

                self.code_paths = [(root / "inference" / "backend.py").as_posix()]

                mlflow.pyfunc.log_model(
                    name="model",
                    python_model=root / "inference" / "model.py",
                    registered_model_name="penguins",
                    code_paths=self.code_paths,
                    artifacts=self.artifacts,
                    pip_requirements=self.pip_requirements,
                )

        else:

            self.register= False
            self.logger.info(
                "The accuracy of the model (%.2f) is lower than the accuracy threshold "
                "(%.2f). Skipping model registration.",
                self.test_accuracy,
                self.accuracy_threshold,
            )

        self.next(self.end)

    @step
    def end(self):
        self.logger.info("Pipeline finished.")


    def _get_model_artifacts(self, directory):
        import joblib

        model_path=(Path(directory)/"model.keras").as_posix()
        self.model.save(model_path)

        features_transformer_path = (Path(directory) / "features.joblib").as_posix()
        target_transformer_path = (Path(directory) / "target.joblib").as_posix()
        joblib.dump(self.features_transformer, features_transformer_path)
        joblib.dump(self.target_transformer, target_transformer_path)

        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }

    def _get_model_pip_requirements(self):
        import keras
        import numpy as np
        import pandas as pd
        import sklearn
        import tensorflow as tf

        return [
            f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}",
            f"keras=={keras.__version__}",
            f"tensorflow=={tf.__version__}",
        ]


if __name__ == "__main__":
    Training()