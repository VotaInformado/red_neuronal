import random
import pandas as pd
from django.conf import settings

# Project
from red_neuronal.tests.test_helpers.test_case import CustomTestCase
import red_neuronal.tests.test_helpers.mocks as mck
from red_neuronal.components.neural_network import NeuralNetwork
from red_neuronal.tests.test_helpers.faker import create_fake_df
from red_neuronal.utils.exceptions.exceptions import UntrainedNeuralNetwork
from red_neuronal.utils.enums import VoteChoices


class NeuralNetworkTestCase(CustomTestCase):
    COLUMNS = {
        "project": "int",
        "party_authors": "parties",
        "vote": "vote",
        "voter_id": "int",
        "project_text": "text",
        "project_title": "short_text",
        "project_year": "year",
    }

    def create_fit_df(self, training_df: pd.DataFrame, fit_df_length=100):
        votes = list(training_df["vote"].unique())
        fit_votes = [random.choice(votes) for _ in range(fit_df_length)]
        legislators = list(training_df["voter_id"].unique())
        fit_legislators = [random.choice(legislators) for _ in range(fit_df_length)]
        parties = list(training_df["party_authors"].unique())
        fit_parties = [random.choice(parties) for _ in range(fit_df_length)]
        df_columns = {
            "project": "int",
            "project_text": "text",
            "project_title": "short_text",
            "project_year": "year",
        }
        fit_df = create_fake_df(df_columns, fit_df_length, as_dict=False)
        fit_df["vote"] = fit_votes
        fit_df["voter_id"] = fit_legislators
        fit_df["party_authors"] = fit_parties
        return fit_df

    def create_prediction_df(self, training_df: pd.DataFrame, prediction_df_length=100):
        df = self.create_fit_df(training_df, prediction_df_length)
        df.pop("vote")
        return df

    def test_neural_network_training(self):
        df: pd.DataFrame = create_fake_df(self.COLUMNS, n=500, as_dict=False)
        neural_network = NeuralNetwork()
        neural_network.train(df)

    def test_neural_network_fit_with_in_memory_model(self):
        train_df: pd.DataFrame = create_fake_df(self.COLUMNS, n=500, as_dict=False)
        fit_df: pd.DataFrame = self.create_fit_df(train_df)
        neural_network = NeuralNetwork()
        neural_network.train(train_df)
        neural_network.fit(fit_df)

    def test_neural_network_fit_with_persisted_model(self):
        train_df: pd.DataFrame = create_fake_df(self.COLUMNS, n=500, as_dict=False)
        fit_df: pd.DataFrame = self.create_fit_df(train_df)
        neural_network = NeuralNetwork()
        neural_network.train(train_df)
        neural_network = NeuralNetwork()
        neural_network.fit(fit_df)

    def test_neural_network_fitting_without_training_raises_exception(self):
        df: pd.DataFrame = create_fake_df(self.COLUMNS, n=100, as_dict=False)
        neural_network = NeuralNetwork()
        with self.assertRaises(UntrainedNeuralNetwork):
            neural_network.fit(df)

    def test_legislator_neural_network_prediction(self):
        DF_LEN = 1
        train_df: pd.DataFrame = create_fake_df(self.COLUMNS, n=500, as_dict=False)
        neural_network = NeuralNetwork()
        neural_network.train(train_df)
        predict_columns = self.COLUMNS.copy()
        predict_columns.pop("vote")
        prediction_df: pd.DataFrame = self.create_prediction_df(train_df, DF_LEN)
        predictions = neural_network.predict(prediction_df)
        self.assertEqual(len(predictions), DF_LEN)
        prediction = predictions[0]
        possible_predictions = [choice.value for choice in VoteChoices]
        self.assertIn(prediction, possible_predictions)

    def test_project_neural_network_prediction(self):
        DF_LEN = 72
        train_df: pd.DataFrame = create_fake_df(self.COLUMNS, n=500, as_dict=False)
        neural_network = NeuralNetwork()
        neural_network.train(train_df)
        predict_columns = self.COLUMNS.copy()
        predict_columns.pop("vote")
        prediction_df: pd.DataFrame = self.create_prediction_df(train_df, DF_LEN)
        predictions = neural_network.predict(prediction_df)
        self.assertEqual(len(predictions), DF_LEN)
        possible_predictions = [choice.value for choice in VoteChoices]
        for prediction in predictions:
            self.assertIn(prediction, possible_predictions)

    def test_prediction_after_fit(self):
        DF_LEN = 1
        train_df: pd.DataFrame = create_fake_df(self.COLUMNS, n=500, as_dict=False)
        neural_network = NeuralNetwork()
        neural_network.train(train_df)
        predict_columns = self.COLUMNS.copy()
        predict_columns.pop("vote")
        fit_df: pd.DataFrame = self.create_fit_df(train_df)
        neural_network.fit(fit_df)
        prediction_df: pd.DataFrame = self.create_prediction_df(train_df, DF_LEN)
        predictions = neural_network.predict(prediction_df)
        self.assertEqual(len(predictions), DF_LEN)
        prediction = predictions[0]
        possible_predictions = [choice.value for choice in VoteChoices]
        self.assertIn(prediction, possible_predictions)
