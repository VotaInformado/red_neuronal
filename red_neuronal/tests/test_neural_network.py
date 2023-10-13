import os
import random
import pandas as pd
from django.test import TestCase
from django.conf import settings

# Project
from red_neuronal.components.neural_network import NeuralNetwork
from red_neuronal.tests.utils.faker import create_fake_df
from red_neuronal.utils.exceptions.exceptions import UntrainedNeuralNetwork


class NeuralNetworkTestCase(TestCase):
    def setUp(self):
        self.COLUMNS = {
            "ley": "int",
            "expediente_inicial": "exp",
            "partido": "str",
            "voto": "vote",
            "diputado_nombre": "str",
            "texto": "text",
            "titulo": "short_text",
            "anio_exp": "year",
        }
        self.remove_persistence_files()

    def create_fit_df(self, training_df: pd.DataFrame, fit_df_length=100):
        votes = list(training_df["voto"].unique())
        fit_votes = [random.choice(votes) for _ in range(fit_df_length)]
        legislators = list(training_df["diputado_nombre"].unique())
        fit_legislators = [random.choice(legislators) for _ in range(fit_df_length)]
        parties = list(training_df["partido"].unique())
        fit_parties = [random.choice(parties) for _ in range(fit_df_length)]
        df_columns = {
            "ley": "int",
            "expediente_inicial": "exp",
            "texto": "text",
            "titulo": "short_text",
            "anio_exp": "year",
        }
        fit_df = create_fake_df(df_columns, fit_df_length, as_dict=False)
        fit_df["voto"] = fit_votes
        fit_df["diputado_nombre"] = fit_legislators
        fit_df["partido"] = fit_parties
        return fit_df

    def remove_persistence_files(self):
        files = os.listdir(settings.MODEL_SAVING_DIR)
        # Iterate through the files and remove them
        for file in files:
            file_path = os.path.join(settings.MODEL_SAVING_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def test_neural_network_training(self):
        df: pd.DataFrame = create_fake_df(self.COLUMNS, n=500, as_dict=False)
        neural_network = NeuralNetwork()
        neural_network.train(df)

    def test_neural_network_fit_with_in_memory_modely(self):
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
