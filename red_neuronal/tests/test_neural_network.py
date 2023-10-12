from django.test import TestCase
import pandas as pd

# Project
from red_neuronal.components.neural_network import NeuralNetwork
from red_neuronal.tests.utils.faker import create_fake_df

# Create your tests here.


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

    def test_neural_network(self):
        df: pd.DataFrame = create_fake_df(self.COLUMNS, n=500, as_dict=False)
        neural_network = NeuralNetwork(df)
        neural_network.compile()
