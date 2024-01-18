import numpy as np
# Other
import pandas as pd
# Project
from red_neuronal.components.neural_network.neural_network import NeuralNetwork


class Predictor(NeuralNetwork):
    def predict(self, df: pd.DataFrame):
        """Predicts the votes for the given data"""
        self._load_model()
        self.df: pd.DataFrame = self._normalize_years(df)
        self._load_encoders()
        self._generate_inputs_for_prediction()
        self._create_embeddings_for_prediction()
        predictions = self._predict()
        return predictions

    def _generate_inputs_for_prediction(self):
        self.legislator_ids = self.df["voter_id"]
        self.legislators = self._get_legislators_input(self.df)
        self.authors = self._get_authors_input(self.df)
        self.authors = self.authors.applymap(lambda x: int(bool(x)))
        self.years = self.df["project_year_cont"]

    def _create_text_embeddings_for_prediction(self):
        law_and_text = self.df.drop_duplicates(subset=["project"])[
            ["project", "project_text"]
        ]
        law_and_text["project_text"] = law_and_text["project_text"].map(
            lambda x: self.embedder.create_law_text_embedding(x)
        )
        text_and_embedding = pd.DataFrame(
            data=law_and_text["project_text"].tolist(), index=law_and_text["project"]
        ).reset_index()

        self.texts = self._get_embeddings(self.df, text_and_embedding)

    def _create_title_embeddings_for_prediction(self):
        law_and_text = self.df.drop_duplicates(subset=["project"])[
            ["project", "project_title"]
        ]
        law_and_text["project_title"] = law_and_text["project_title"].map(
            lambda x: self.embedder.create_law_text_embedding(x)
        )
        title_and_embedding = pd.DataFrame(
            data=law_and_text["project_title"].tolist(), index=law_and_text["project"]
        ).reset_index()

        self.titles = self._get_embeddings(self.df, title_and_embedding)

    def _create_embeddings_for_prediction(self):
        self._create_text_embeddings_for_prediction()
        self._create_title_embeddings_for_prediction()

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

        result = []
        for legislator, vote_predictions in zip(self.legislator_ids, vote_predictions):
            result.append({"legislator": legislator, "vote": vote_predictions})
        return result
