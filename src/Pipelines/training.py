import os
from pathlib import Path

from metaflow import parameters,cards,current,step,environment


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


