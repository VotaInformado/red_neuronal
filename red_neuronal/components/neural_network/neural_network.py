import os
import numpy as np
from django.conf import settings

# Sklearn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# from tensorflow.math import reduce_mean
import keras
from keras.models import model_from_json
from keras.callbacks import History

# Other
import pandas as pd
from red_neuronal.components.encoder import PartiesEncoder, LegislatorsEncoder, VotesEncoder

# Project
from red_neuronal.components.embedding import UniversalEmbedding
from red_neuronal.utils.exceptions.exceptions import UntrainedNeuralNetwork


# import tensorflow as tf

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# # No funciona para desactivar los warnings de tensorflow


class NeuralNetwork:
    class Meta:
        abstract = True

    MODEL_FILE_SAVING_DIR = f"{settings.MODEL_SAVING_DIR}/model.json"
    WEIGHTS_SAVING_DIR = f"{settings.MODEL_SAVING_DIR}/model.h5"
    HISTORY_SAVING_DIR = f"{settings.MODEL_SAVING_DIR}/history.json"
    REPORT_SAVING_DIR = f"{settings.MODEL_SAVING_DIR}/report.txt"

    def __init__(self):
        # TODO: comentarios: no se usa el partido del votante, tal vez podríamos usarlo
        # Lo que se necesita es un DataFrame con las columnas:
        # - project: id del proyecto
        # - project_title
        # - project_text
        # - voter_id
        # - voter_party --> TODO: esto no se estaría usando, entraría como voter_party
        # - party_authors
        # - vote
        # - project_year

        self.embedder = UniversalEmbedding()
        self.model = None

    def _load_encoders(self):
        self.votes_encoder = VotesEncoder(is_training=False)
        self.legislators_encoder = LegislatorsEncoder(is_training=False)
        self.parties_encoder = PartiesEncoder(is_training=False)

    def _fit_encoders(self):
        self.votes_encoder = VotesEncoder(is_training=True)
        self.legislators_encoder = LegislatorsEncoder(is_training=True)
        self.parties_encoder = PartiesEncoder(is_training=True)

        self.votes_encoder.fit(self.df["vote"].to_frame())
        self.legislators_encoder.fit(self.df["voter_id"].to_frame())
        authors_dict = self.df["party_authors"].apply(lambda x: {"party_authors": x.split(";") if x else []}).tolist()
        self.parties_encoder.fit(authors_dict)

    def _normalize_years(self, df: pd.DataFrame):
        max_year = df["project_year"].max()
        min_year = df["project_year"].min()
        df["project_year_cont"] = (df["project_year"] - min_year) / (max_year - min_year)
        return df

    def _load_model(self):
        if self.model is not None:
            # The model is still loaded in memory, no need to load it again
            return
        try:
            with open(self.MODEL_FILE_SAVING_DIR, "r") as json_file:
                loaded_model_json = json_file.read()

            # Load weights into new model
            loaded_model: keras.Model = model_from_json(loaded_model_json)
            loaded_model.load_weights(self.WEIGHTS_SAVING_DIR)
            self.model = loaded_model
        except FileNotFoundError:
            raise UntrainedNeuralNetwork()

    def _save_history(self, history: History):
        pass  # Mongo DB?

    def _evaluate(self):
        # Make predictions on the validation data
        y_pred = self.prediction

        # Convert the predictions and test data to class labels
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test_labels = np.argmax(self.y_test, axis=1)
        # Generate the classification report
        report: str = classification_report(
            y_test_labels, y_pred_labels, target_names=self.votes_encoder.get_categories()
        )
        self._save_report(report)

        ConfusionMatrixDisplay.from_predictions(
            y_test_labels, y_pred_labels, display_labels=self.votes_encoder.get_categories()
        )  # normalize?

    def _save_report(self, report: str):
        os.makedirs(os.path.dirname(self.REPORT_SAVING_DIR), exist_ok=True)
        with open(self.REPORT_SAVING_DIR, "w") as file:
            file.write(report)

    def _get_legislators_input(self, df: pd.DataFrame):
        transformed = self.legislators_encoder.transform(df["voter_id"].to_frame())
        return pd.DataFrame(np.array(transformed), columns=self.legislators_encoder.get_feature_names())

    def _get_authors_input(self, df: pd.DataFrame):
        authors_dict = df["party_authors"].apply(lambda x: {"party_authors": x.split(";") if x else []}).tolist()
        transformed = self.parties_encoder.transform(authors_dict)
        return pd.DataFrame(np.array(transformed), columns=self.parties_encoder.get_feature_names())

    def _get_embeddings(self, df: pd.DataFrame, embeddings: pd.DataFrame):
        embeddings = pd.DataFrame.merge(df["project"], embeddings, how="left", on="project")
        embeddings.drop(columns=["project"], inplace=True)
        return embeddings

    # def save_results(self):
    #     self.model.save("/kaggle/working/model")
    #     self.model.save("/kaggle/working/model-tf", save_format="tf")
    #     self.model.save("/kaggle/working/model-keras", save_format="keras")
    #     self.model.save("/kaggle/working/model-keras2/model.keras")
