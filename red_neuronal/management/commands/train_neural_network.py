import pandas as pd

# Base command
from django.core.management.base import BaseCommand

# Project
from red_neuronal.components.data_handler import DataHandler
from red_neuronal.components.neural_network import NeuralNetwork
from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO


class Command(BaseCommand):
    def handle(self, *args, **options):
        endpoints_info = EndpointsDTO()
        data_handler = DataHandler()
        df: pd.DataFrame = data_handler.get_data(endpoints_info)
        neural_network = NeuralNetwork()
        neural_network.train(df)
