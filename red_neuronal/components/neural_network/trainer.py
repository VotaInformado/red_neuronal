import os

# Sklearn
from sklearn.model_selection import train_test_split

# from tensorflow.math import reduce_mean
import keras
from keras import layers
from keras.callbacks import History

# Other
import pandas as pd

# Project
from red_neuronal.components.neural_network.neural_network import NeuralNetwork

# Logging
from red_neuronal.utils.logger import logger


class Trainer(NeuralNetwork):
    def train(
        self,
        df: pd.DataFrame,
        votes: pd.DataFrame,
        parties: pd.DataFrame,
        legislators: pd.DataFrame,
    ):
        """Trains the model from scratch, using the database saved in the class."""
        self.df: pd.DataFrame = self._normalize_years(df)
        logger.info("Fitting encoders...")
        self._fit_encoders(votes, parties, legislators)
        logger.info("Encoders fitted successfully")
        self._split_dataframe()
        self._generate_inputs()
        logger.info("Generating embeddings...")
        self._create_embeddings()
        logger.info("Creating neural network...")
        self._create_neuronal_network()
        logger.info("Compiling model...")
        self._compile_model()
        logger.info("Fitting model...")
        self._fit_model()
        logger.info("Saving model...")
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

    def _create_text_embeddings(self):
        logger.info("Creating text embeddings...")
        law_and_text = self.df.drop_duplicates(subset=["project"])[
            ["project", "project_text"]
        ]
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
        logger.info("Creating title embeddings...")
        law_and_text = self.df.drop_duplicates(subset=["project"])[
            ["project", "project_title"]
        ]
        law_and_text["project_title"] = law_and_text["project_title"].map(
            lambda x: self.embedder.create_law_text_embedding(x)
        )
        title_and_embedding = pd.DataFrame(
            data=law_and_text["project_title"].tolist(), index=law_and_text["project"]
        ).reset_index()

        self.titles_train = self._get_embeddings(self.df_train, title_and_embedding)
        self.titles_val = self._get_embeddings(self.df_val, title_and_embedding)
        self.titles_test = self._get_embeddings(self.df_test, title_and_embedding)

    def _create_embeddings(self):
        self._create_text_embeddings()
        self._create_title_embeddings()

    def _split_dataframe(self):
        df = self.df
        laws = df["project"].unique()
        laws_train, laws_test = train_test_split(laws, train_size=0.95)
        laws_val, laws_test = train_test_split(laws_test, train_size=0.99)

        self.df_train = df.loc[df["project"].isin(laws_train)]
        self.df_val = df.loc[df["project"].isin(laws_val)]
        self.df_test = df.loc[df["project"].isin(laws_test)]

        self.y_train = self.df_train["vote"]
        self.y_val = self.df_val["vote"]
        self.y_test = self.df_test["vote"]

        self.df_train.drop(columns=["vote"])
        self.df_val.drop(columns=["vote"])
        self.df_test.drop(columns=["vote"])

    def _generate_inputs(self):
        # One hot encode votos
        self.y_train, self.y_val, self.y_test = [
            self.votes_encoder.transform(y.to_frame())
            for y in [self.y_train, self.y_val, self.y_test]
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

    def _get_input_dimensions(self):
        self.law_texts_input_dim = self.texts_train.shape[1]
        self.law_titles_input_dim = self.titles_train.shape[1]
        self.legislators_input_dim = len(self.legislators_encoder.get_feature_names())
        party_categories = len(self.parties_encoder.get_feature_names())
        self.authors_input_dim = party_categories

    def _create_network_inputs(self):
        self.law_texts_input = keras.Input(
            shape=(self.law_texts_input_dim,), name="law_texts"
        )  # Variable-length sequence of ints
        self.legislators_input = keras.Input(
            shape=(self.legislators_input_dim,), name="legislators"
        )
        self.authors_input = keras.Input(
            shape=(self.authors_input_dim,), name="authors"
        )
        self.years_input = keras.Input(shape=(1,), name="years")
        self.law_titles_input = keras.Input(
            shape=(self.law_titles_input_dim,), name="law_titles"
        )

    def _create_embeddings_layers(self):
        self.law_features = layers.Embedding(
            self.law_texts_input_dim,
            int(self.law_texts_input_dim / 10),
            name="law_embedding",
        )(self.law_texts_input)
        self.legislators_features = layers.Embedding(
            self.legislators_input_dim, 10, name="legislators_embedding"
        )(self.legislators_input)
        self.authors_features = layers.Embedding(
            self.authors_input_dim,
            int(self.authors_input_dim / 10),
            name="authors_embedding",
        )(self.authors_input)
        self.title_features = layers.Embedding(
            self.law_titles_input_dim,
            int(self.law_titles_input_dim / 10),
            name="title_embedding",
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
        self.features = layers.Dense(128, activation="relu", name="relu_1")(
            self.features
        )
        self.features = layers.Dense(128, activation="relu", name="relu_2")(
            self.features
        )
        self.features = layers.Dense(128, activation="relu", name="relu_3")(
            self.features
        )

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
        batch_size = 32
        logger.info(f"Fitting model with batchsize={batch_size}")
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
            batch_size=batch_size,
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
        os.makedirs(os.path.dirname(self.MODEL_KERAS_SAVING_DIR), exist_ok=True)
        self.model.save(self.MODEL_KERAS_SAVING_DIR)
