import os
import numpy as np
from django.conf import settings

# Sklearn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# from tensorflow.math import reduce_mean
import keras
from keras import layers
from keras.models import model_from_json
from keras.callbacks import History

# Other
import pandas as pd
import matplotlib.pyplot as plt
import collections
from red_neuronal.components.encoder import PartiesEncoder, LegislatorsEncoder, VotesEncoder

# Project
from red_neuronal.components.embedding import UniversalEmbedding
from red_neuronal.utils.exceptions.exceptions import UntrainedNeuralNetwork
from red_neuronal.utils.logger import logger


import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# No funciona para desactivar los warnings de tensorflow


class NeuralNetwork:
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
        # - Partido Legislador --> TODO: esto no se estaría usando, entraría como voter_party
        # - party_authors
        # - vote
        # - project_year

        self.embedder = UniversalEmbedding()
        self.model = None

    def train(self, df: pd.DataFrame):
        """Trains the model from scratch, using the database saved in the class."""
        self.df: pd.DataFrame = self._normalize_years(df)
        self._fit_encoders()
        self._split_dataframe()
        self._generate_inputs()
        self._create_embeddings()
        self._create_neuronal_network()
        self._compile_model()
        self._fit_model()
        self._save_model()

    def fit(self, df: pd.DataFrame):
        """Fits the model using new data saved in the class."""
        # TODO: verificar si cambia al menos la dimensión de una de las capas, de ser así, volver a entrenar
        self._load_model()  # Will raise an exception if the model is not trained
        self.df: pd.DataFrame = self._normalize_years(df)
        self._load_encoders()
        self._split_dataframe()
        self._generate_inputs()
        self._create_embeddings()
        self._compile_model()
        self._fit_model()
        self._save_model()

    def predict(self, df: pd.DataFrame):
        """Predicts the votes for the given data"""
        self._load_model()
        self.df: pd.DataFrame = self._normalize_years(df)
        self._load_encoders()
        self._generate_inputs_for_prediction()
        self._create_embeddings_for_prediction()
        predictions = self._predict()
        return predictions
        # self._evaluate()

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

    def _split_dataframe(self):
        df = self.df
        laws = df["project"].unique()
        laws_train, laws_test = train_test_split(laws, train_size=0.7)
        laws_val, laws_test = train_test_split(laws_test, train_size=0.66)

        self.df_train = df.loc[df["project"].isin(laws_train)]
        self.df_val = df.loc[df["project"].isin(laws_val)]
        self.df_test = df.loc[df["project"].isin(laws_test)]

        self.y_train = self.df_train["vote"]
        self.y_val = self.df_val["vote"]
        self.y_test = self.df_test["vote"]

        self.df_train.drop(columns=["vote"])
        self.df_val.drop(columns=["vote"])
        self.df_test.drop(columns=["vote"])

    def _normalize_years(self, df: pd.DataFrame):
        max_year = df["project_year"].max()
        min_year = df["project_year"].min()
        df["project_year_cont"] = (df["project_year"] - min_year) / (max_year - min_year)
        return df

    def _get_legislators_input(self, df: pd.DataFrame):
        transformed = self.legislators_encoder.transform(df["voter_id"].to_frame())
        return pd.DataFrame(np.array(transformed), columns=self.legislators_encoder.get_feature_names())

    def _get_authors_input(self, df: pd.DataFrame):
        authors_dict = df["party_authors"].apply(lambda x: {"party_authors": x.split(";") if x else []}).tolist()
        transformed = self.parties_encoder.transform(authors_dict)
        return pd.DataFrame(np.array(transformed), columns=self.parties_encoder.get_feature_names())

    def _generate_inputs(self):
        # One hot encode votos
        self.y_train, self.y_val, self.y_test = [
            self.votes_encoder.transform(y.to_frame()) for y in [self.y_train, self.y_val, self.y_test]
        ]

        self.legislators_train = self._get_legislators_input(self.df_train)
        self.legislators_val = self._get_legislators_input(self.df_val)
        self.legislators_test = self._get_legislators_input(self.df_test)

        self.authors_train = self._get_authors_input(self.df_train)
        self.authors_train = self.authors_train.applymap(lambda x: int(bool(x)))
        self.authors_val = self._get_authors_input(self.df_val)
        self.authors_test = self._get_authors_input(self.df_test)

        self.year_train = self.df_train["project_year_cont"]
        self.year_val = self.df_val["project_year_cont"]
        self.year_test = self.df_test["project_year_cont"]

    def _generate_inputs_for_prediction(self):
        self.legislators = self._get_legislators_input(self.df)
        self.authors = self._get_authors_input(self.df)
        self.authors = self.authors.applymap(lambda x: int(bool(x)))
        self.years = self.df["project_year_cont"]

    def _create_embeddings(self):
        self._create_text_embeddings()
        self._create_title_embeddings()

    def _create_embeddings_for_prediction(self):
        self._create_text_embeddings_for_prediction()
        self._create_title_embeddings_for_prediction()

    def _get_embeddings(self, df: pd.DataFrame, embeddings: pd.DataFrame):
        embeddings = pd.DataFrame.merge(df["project"], embeddings, how="left", on="project")
        embeddings.drop(columns=["project"], inplace=True)
        return embeddings

    def _create_text_embeddings_for_prediction(self):
        law_and_text = self.df.drop_duplicates(subset=["project"])[["project", "project_text"]]
        law_and_text["project_text"] = law_and_text["project_text"].map(
            lambda x: self.embedder.create_law_text_embedding(x)
        )
        text_and_embedding = pd.DataFrame(
            data=law_and_text["project_text"].tolist(), index=law_and_text["project"]
        ).reset_index()

        self.texts = self._get_embeddings(self.df, text_and_embedding)

    def _create_title_embeddings_for_prediction(self):
        law_and_text = self.df.drop_duplicates(subset=["project"])[["project", "project_title"]]
        law_and_text["project_title"] = law_and_text["project_title"].map(
            lambda x: self.embedder.create_law_text_embedding(x)
        )
        title_and_embedding = pd.DataFrame(
            data=law_and_text["project_title"].tolist(), index=law_and_text["project"]
        ).reset_index()

        self.titles = self._get_embeddings(self.df, title_and_embedding)

    def _create_text_embeddings(self):
        law_and_text = self.df.drop_duplicates(subset=["project"])[["project", "project_text"]]
        law_and_text["project_text"] = law_and_text["project_text"].map(
            lambda x: self.embedder.create_law_text_embedding(x)
        )
        text_and_embedding = pd.DataFrame(
            data=law_and_text["project_text"].tolist(), index=law_and_text["project"]
        ).reset_index()

        self.texts_train = self._get_embeddings(self.df_train, text_and_embedding)
        self.texts_val = self._get_embeddings(self.df_val, text_and_embedding)
        self.texts_test = self._get_embeddings(self.df_test, text_and_embedding)

    def _create_title_embeddings(self):
        law_and_text = self.df.drop_duplicates(subset=["project"])[["project", "project_title"]]
        law_and_text["project_title"] = law_and_text["project_title"].map(
            lambda x: self.embedder.create_law_text_embedding(x)
        )
        title_and_embedding = pd.DataFrame(
            data=law_and_text["project_title"].tolist(), index=law_and_text["project"]
        ).reset_index()

        self.titles_train = self._get_embeddings(self.df_train, title_and_embedding)
        self.titles_val = self._get_embeddings(self.df_val, title_and_embedding)
        self.titles_test = self._get_embeddings(self.df_test, title_and_embedding)

    def _get_input_dimensions(self):
        self.law_texts_input_dim = self.texts_train.shape[1]
        self.law_titles_input_dim = self.titles_train.shape[1]
        self.legislators_input_dim = len(self.legislators_encoder.get_feature_names())
        party_categories = len(self.parties_encoder.get_feature_names())
        max_parties_value = self.authors_train.max().max()
        self.authors_input_dim = int(max(party_categories, max_parties_value))
        self.authors_input_dim = party_categories

    def _create_network_inputs(self):
        self.law_texts_input = keras.Input(
            shape=(self.law_texts_input_dim,), name="law_texts"
        )  # Variable-length sequence of ints
        self.legislators_input = keras.Input(shape=(self.legislators_input_dim,), name="legislators")
        self.authors_input = keras.Input(shape=(self.authors_input_dim,), name="authors")
        self.years_input = keras.Input(shape=(1,), name="years")
        self.law_titles_input = keras.Input(shape=(self.law_titles_input_dim,), name="law_titles")

    def _create_embeddings_layers(self):
        self.law_features = layers.Embedding(
            self.law_texts_input_dim, int(self.law_texts_input_dim / 10), name="law_embedding"
        )(self.law_texts_input)
        self.legislators_features = layers.Embedding(self.legislators_input_dim, 10, name="legislators_embedding")(
            self.legislators_input
        )
        self.authors_features = layers.Embedding(
            self.authors_input_dim, int(self.authors_input_dim / 10), name="authors_embedding"
        )(self.authors_input)
        self.title_features = layers.Embedding(
            self.law_titles_input_dim, int(self.law_titles_input_dim / 10), name="title_embedding"
        )(self.law_titles_input)

    def _create_flattened_layers(self):
        self.flatten_law_features = layers.Flatten()(self.law_features)
        self.flatten_legislators_features = layers.Flatten()(self.legislators_features)
        self.flatten_authors_features = layers.Flatten()(self.authors_features)
        self.flatten_title = layers.Flatten()(self.title_features)

    def _create_concatenated_layer(self):
        self.features = layers.Concatenate(axis=-1, name="concatenados")(
            [
                self.flatten_law_features,
                self.flatten_legislators_features,
                self.flatten_authors_features,
                self.flatten_title,
                self.years_input,
            ]
        )

    def _add_extra_dense_layers(self):
        self.features = layers.Dense(128, activation="relu", name="relu_1")(self.features)
        self.features = layers.Dense(128, activation="relu", name="relu_2")(self.features)
        self.features = layers.Dense(128, activation="relu", name="relu_3")(self.features)

    def _create_output_layer(self):
        self.output_dim = len(self.votes_encoder.get_feature_names())
        self.features = layers.Dense(self.output_dim, name="vote")(self.features)
        self.features = layers.Activation("softmax", name="softmax_vote")(self.features)

    def _create_model(self):
        self.model = keras.Model(
            inputs=[
                self.law_texts_input,
                self.legislators_input,
                self.authors_input,
                self.years_input,
                self.law_titles_input,
            ],
            outputs=[self.features],
        )

    def _create_neuronal_network(self):
        self._get_input_dimensions()
        self._create_network_inputs()
        self._create_embeddings_layers()
        self._create_flattened_layers()
        self._create_concatenated_layer()
        self._add_extra_dense_layers()
        self._create_output_layer()
        self._create_model()

        keras.utils.plot_model(self.model, "my_first_model.png", show_shapes=True)

    def _compile_model(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss={
                "softmax_vote": keras.losses.CategoricalCrossentropy(),
            },
            metrics=["accuracy"]
            # loss_weights={"softmax_vote": 1.0},
        )

    def _fit_model(self):
        history: History = self.model.fit(
            {
                "authors": self.authors_train,
                "law_texts": self.texts_train,
                "legislators": self.legislators_train,
                "years": self.year_train,
                "law_titles": self.titles_train,
            },
            {"softmax_vote": self.y_train},
            epochs=4,
            batch_size=32,
            validation_data=(
                {
                    "authors": self.authors_val,
                    "law_texts": self.texts_val,
                    "legislators": self.legislators_val,
                    "years": self.year_val,
                    "law_titles": self.titles_val,
                },
                {"softmax_vote": self.y_val},
            ),
        )
        self._save_history(history)

    def _save_model(self):
        model_json = self.model.to_json()
        os.makedirs(os.path.dirname(self.MODEL_FILE_SAVING_DIR), exist_ok=True)
        with open(self.MODEL_FILE_SAVING_DIR, "w") as json_file:
            json_file.write(model_json)
        os.makedirs(os.path.dirname(self.WEIGHTS_SAVING_DIR), exist_ok=True)
        self.model.save_weights(self.WEIGHTS_SAVING_DIR)
        logger.info("Model saved successfully")

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

    def _predict(self):
        self.prediction = self.model.predict(
            {
                "authors": self.authors,
                "legislators": self.legislators,
                "years": self.years,
                "law_texts": self.texts,
                "law_titles": self.titles,
            },
            batch_size=2,
        )
        POSSIBLE_VOTES = self.votes_encoder.get_categories()
        max_probs_index = np.argmax(self.prediction, axis=1)
        vote_predictions = [POSSIBLE_VOTES[i] for i in max_probs_index]

        freq = collections.Counter(vote_predictions)
        plt.figure(figsize=(10, 6))
        plt.bar(freq.keys(), freq.values())
        plt.title("Cantidad de votos predecidos por categoría")
        plt.show()
        return vote_predictions

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

    # def save_results(self):
    #     self.model.save("/kaggle/working/model")
    #     self.model.save("/kaggle/working/model-tf", save_format="tf")
    #     self.model.save("/kaggle/working/model-keras", save_format="keras")
    #     self.model.save("/kaggle/working/model-keras2/model.keras")
