import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import numpy as np
from django.conf import settings

# Sklearn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction import DictVectorizer
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

# Project
from red_neuronal.components.universal_embedding import UniversalEmbedding


class NeuralNetwork:
    def __init__(self, df: pd.DataFrame):
        # Lo que se necesita es un DataFrame con las columnas:
        # - Ley
        # - Titulo
        # - Texto
        # - Legislador
        # - Partido Legislador
        # - Autores
        # - Voto
        # - Año

        self.embedder = UniversalEmbedding()
        self.df = self.normalize_years(df)

    def compile(self):
        # Whole compilation process
        self.fit_encoders()
        self.generate_inputs()
        self.get_embeddings()
        self.create_neuronal_network()
        self.fit_model()
        self.save_model()

    def fit(self):
        # Just fitting new data
        self.generate_inputs()
        self.fit_model()
        self.save_model()

    def fit_encoders(self):
        df = self.df
        self.encoded_votes = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(df["voto"].to_frame())
        self.encoded_deputies = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(
            df["diputado_nombre"].to_frame()
        )
        autores_dict = df["partido"].apply(lambda x: {"partido": x.split(";")}).tolist()
        self.encoded_authors = DictVectorizer(sparse=False, separator="_").fit(autores_dict)

        leyes = df["ley"].unique()
        leyes_train, leyes_test = train_test_split(leyes, train_size=0.7)
        leyes_val, leyes_test = train_test_split(leyes_test, train_size=0.66)

        self.df_train = df.loc[df["ley"].isin(leyes_train)]
        self.df_val = df.loc[df["ley"].isin(leyes_val)]
        self.df_test = df.loc[df["ley"].isin(leyes_test)]

        self.y_train = self.df_train["voto"]
        self.y_val = self.df_val["voto"]
        self.y_test = self.df_test["voto"]

        self.df_train.drop(columns=["voto"])
        self.df_val.drop(columns=["voto"])
        self.df_test.drop(columns=["voto"])

        print(
            f"Porcentaje leyes train: {leyes_train.shape[0]/leyes.shape[0]:.2%} --> {leyes_train.shape[0]} leyes = {self.df_train.shape[0]} votaciones"
        )
        print(
            f"Porcentaje leyes validation: {leyes_val.shape[0]/leyes.shape[0]:.2%} --> {leyes_val.shape[0]} leyes = {self.df_val.shape[0]} votaciones"
        )
        print(
            f"Porcentaje leyes test-holdout: {leyes_test.shape[0]/leyes.shape[0]:.2%} --> {leyes_test.shape[0]} leyes = {self.df_test.shape[0]} votaciones"
        )

    def normalize_years(self, df: pd.DataFrame):
        max_year = df["anio_exp"].max()
        min_year = df["anio_exp"].min()
        df["anio_exp_cont"] = (df["anio_exp"] - min_year) / (max_year - min_year)
        return df

    def get_deputies_input(self, df: pd.DataFrame):
        transformed = self.encoded_deputies.transform(df["diputado_nombre"].to_frame())
        return pd.DataFrame(
            np.array(transformed), columns=self.encoded_deputies.get_feature_names_out(["diputado_nombre"])
        )

    def get_authors_input(self, df: pd.DataFrame):
        autores_dict = df["partido"].apply(lambda x: {"partido": x.split(";")}).tolist()
        transformed = self.encoded_authors.transform(autores_dict)
        return pd.DataFrame(np.array(transformed), columns=self.encoded_authors.get_feature_names_out(["partido"]))

    def generate_inputs(self):
        # One hot encode votos
        self.y_train, self.y_val, self.y_test = [
            self.encoded_votes.transform(y.to_frame()) for y in [self.y_train, self.y_val, self.y_test]
        ]

        self.deputies_train = self.get_deputies_input(self.df_train)
        self.deputies_val = self.get_deputies_input(self.df_val)
        self.deputies_test = self.get_deputies_input(self.df_test)

        self.authors_train = self.get_authors_input(self.df_train)
        self.authors_val = self.get_authors_input(self.df_val)
        self.authors_test = self.get_authors_input(self.df_test)

        self.year_train = self.df_train["anio_exp_cont"]
        self.year_val = self.df_val["anio_exp_cont"]
        self.year_test = self.df_test["anio_exp_cont"]

    def get_embeddings(self):
        self.get_text_embeddings()
        self.get_title_embeddings()

    def get_text_embeddings(self):
        law_and_text = self.df.drop_duplicates(subset=["ley"])[["ley", "texto"]]
        law_and_text["texto"] = law_and_text["texto"].map(lambda x: self.embedder.create_law_text_embedding(x))
        self.law_and_embedding = pd.DataFrame(
            data=law_and_text["texto"].tolist(), index=law_and_text["ley"]
        ).reset_index()

        self.laws_train = self._get_text_embeddings(self.df_train)
        self.laws_val = self._get_text_embeddings(self.df_val)
        self.laws_test = self._get_text_embeddings(self.df_test)

    def _get_text_embeddings(self, df: pd.DataFrame):
        embeddings = pd.DataFrame.merge(df["ley"], self.law_and_embedding, how="left", on="ley")
        embeddings.drop(columns=["ley"], inplace=True)
        return embeddings

    def get_title_embeddings(self):
        # Obtener embedding para cada titulo
        law_and_text = self.df.drop_duplicates(subset=["ley"])[["ley", "titulo"]]
        law_and_text["titulo"] = law_and_text["titulo"].map(lambda x: self.embedder.create_law_text_embedding(x))
        self.title_and_embedding = pd.DataFrame(
            data=law_and_text["titulo"].tolist(), index=law_and_text["ley"]
        ).reset_index()

        self.titles_train = self._get_title_embeddings(self.df_train)
        self.titles_val = self._get_title_embeddings(self.df_val)
        self.titles_test = self._get_title_embeddings(self.df_test)

    def _get_title_embeddings(self, df: pd.DataFrame):
        embeddings = pd.DataFrame.merge(df["ley"], self.title_and_embedding, how="left", on="ley")
        embeddings.drop(columns=["ley"], inplace=True)
        return embeddings

    def create_neuronal_network(self):
        # Dimensiones
        laws_input_dim = self.laws_train.shape[1]
        deputies_input_dim = len(self.encoded_deputies.get_feature_names_out())
        authors_input_dim = len(self.encoded_authors.get_feature_names_out())
        titles_input_dim = self.titles_train.shape[1]
        output_dim = len(self.encoded_votes.get_feature_names_out())

        ### INPUTS

        law_input = keras.Input(shape=(laws_input_dim,), name="ley")  # Variable-length sequence of ints
        politician_input = keras.Input(shape=(deputies_input_dim,), name="politico")
        authors_input = keras.Input(shape=(authors_input_dim,), name="autores")
        anio_input = keras.Input(shape=(1,), name="anio")
        title_input = keras.Input(shape=(titles_input_dim,), name="titulos")

        ### CAPAS

        # parties_sigmoid = layers.Dense(units=cant_partidos, activation="sigmoid")(parties_input)

        # EMBEDDING

        # Embed each word in the title into a 64-dimensional vector
        law_features = layers.Embedding(laws_input_dim, int(laws_input_dim / 10), name="law_embedding")(law_input)
        politician_features = layers.Embedding(deputies_input_dim, 10, name="politician_embedding")(politician_input)
        authors_features = layers.Embedding(authors_input_dim, int(authors_input_dim / 10), name="authors_embedding")(
            authors_input
        )
        title_features = layers.Embedding(titles_input_dim, int(titles_input_dim / 10), name="title_embedding")(
            title_input
        )

        # Reduce sequence of embedded words in the title into a single 128-dimensional vector
        # text_features = layers.LSTM(128)(text_features)

        # FLATENATION
        flatten_law_features = layers.Flatten()(law_features)
        flatten_politician_features = layers.Flatten()(politician_features)
        flatten_authors_features = layers.Flatten()(authors_features)
        flatten_title = layers.Flatten()(title_features)
        # CONCATENACION

        # features = layers.concatenate([law_features, politician_features, authors_features], axis=2 )
        features = layers.Concatenate(axis=-1, name="concatenados")(
            [flatten_law_features, flatten_politician_features, flatten_authors_features, flatten_title, anio_input]
        )
        # DENSAS

        features = layers.Dense(128, activation="relu", name="relu_1")(features)
        features = layers.Dense(128, activation="relu", name="relu_2")(features)
        features = layers.Dense(128, activation="relu", name="relu_3")(features)

        # Stick a department classifier on top of the features
        features = layers.Dense(output_dim, name="voto")(features)
        features = layers.Activation("softmax", name="voto_softmax")(features)

        # Instantiate an end-to-end model predicting both priority and department
        self.model = keras.Model(
            inputs=[law_input, politician_input, authors_input, anio_input, title_input],
            outputs=[features],
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss={
                "voto_softmax": keras.losses.CategoricalCrossentropy(),
            },
            metrics=["accuracy"]
            # loss_weights={"voto_softmax": 1.0},
        )

        # keras.utils.plot_model(self.model, "my_first_model.png", show_shapes=True)

    def save_model(self):
        model_json = self.model.to_json()
        os.makedirs(os.path.dirname(settings.MODEL_SAVING_DIR), exist_ok=True)
        with open(settings.MODEL_SAVING_DIR, "w") as json_file:
            json_file.write(model_json)
        os.makedirs(os.path.dirname(settings.WEIGHTS_SAVING_DIR), exist_ok=True)
        self.model.save_weights(settings.WEIGHTS_SAVING_DIR)

    def load_model(self):
        with open(settings.MODEL_SAVING_DIR, "r") as json_file:
            loaded_model_json = json_file.read()

        # Load weights into new model
        loaded_model: keras.Model = model_from_json(loaded_model_json)
        loaded_model.load_weights(settings.WEIGHTS_SAVING_DIR)
        self.model = loaded_model

    def save_history(self, history: History):
        pass  # Mongo DB?

    def fit_model(self):
        history: History = self.model.fit(
            {
                "autores": self.authors_train,
                "ley": self.laws_train,
                "politico": self.deputies_train,
                "anio": self.year_train,
                "titulos": self.titles_train,
            },
            {"voto_softmax": self.y_train},
            epochs=4,
            batch_size=32,
            validation_data=(
                {
                    "autores": self.authors_val,
                    "ley": self.laws_val,
                    "politico": self.deputies_val,
                    "anio": self.year_val,
                    "titulos": self.titles_val,
                },
                {"voto_softmax": self.y_val},
            ),
        )
        self.save_history(history)

    def predict(self):
        self.prediction = self.model.predict(
            {
                "autores": self.authors_test,
                "ley": self.laws_test,
                "politico": self.deputies_test,
                "anio": self.year_test,
                "titulos": self.titles_test,
            },
            batch_size=2,
        )
        POSSIBLE_VOTES = self.encoded_votes.categories_[0]
        max_probs_index = np.argmax(self.prediction, axis=1)
        vote_predictions = [POSSIBLE_VOTES[i] for i in max_probs_index]

        # real_votes = y[int(len(y)*0.8):]
        # total_votes = len(real_votes)
        # correct_predictions = 0
        # for i in range(total_votes):
        #    if real_votes[i] == vote_predictions[i]:
        #        correct_predictions += 1
        # print(f"Porcentaje de acierto: {correct_predictions/total_votes}")

        freq = collections.Counter(vote_predictions)
        plt.figure(figsize=(10, 6))
        plt.bar(freq.keys(), freq.values())
        plt.title("Cantidad de votos predecidos por categoría")
        plt.show()

    def evaluate(self):
        # Make predictions on the validation data
        y_pred = self.prediction

        # Convert the predictions and test data to class labels
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test_labels = np.argmax(self.y_test, axis=1)
        # Generate the classification report
        report: str = classification_report(
            y_test_labels, y_pred_labels, target_names=self.encoded_votes.categories_[0]
        )
        self.save_report(report)

        ConfusionMatrixDisplay.from_predictions(
            y_test_labels, y_pred_labels, display_labels=self.encoded_votes.categories_[0]
        )  # normalize?

    def save_report(self, report: str):
        os.makedirs(os.path.dirname(settings.REPORT_SAVING_DIR), exist_ok=True)
        with open(settings.REPORT_SAVING_DIR, "w") as file:
            file.write(report)

    # def save_results(self):
    #     self.model.save("/kaggle/working/model")
    #     self.model.save("/kaggle/working/model-tf", save_format="tf")
    #     self.model.save("/kaggle/working/model-keras", save_format="keras")
    #     self.model.save("/kaggle/working/model-keras2/model.keras")
