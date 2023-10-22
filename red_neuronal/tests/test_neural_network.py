import os
import shutil
import random
import pandas as pd
from red_neuronal.tests.test_helpers.test_case import CustomTestCase
from django.conf import settings

# Project
from red_neuronal.components.neural_network import NeuralNetwork
from red_neuronal.tests.test_helpers.faker import create_fake_df
from red_neuronal.utils.exceptions.exceptions import UntrainedNeuralNetwork


class NeuralNetworkTestCase(CustomTestCase):
    COLUMNS = {
        "project_id": "int",
        "party_authors": "str",
        "vote": "vote",
        "voter_id": "str",
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
            "project_id": "int",
            "project_text": "text",
            "project_title": "short_text",
            "project_year": "year",
        }
        fit_df = create_fake_df(df_columns, fit_df_length, as_dict=False)
        fit_df["vote"] = fit_votes
        fit_df["voter_id"] = fit_legislators
        fit_df["party_authors"] = fit_parties
        return fit_df

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
