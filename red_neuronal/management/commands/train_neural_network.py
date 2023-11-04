import pandas as pd

# Base command
from django.core.management.base import BaseCommand

# Project
from red_neuronal.components.data_handler import TrainDataHandler
from red_neuronal.components.neural_network import NeuralNetwork
from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--starting-date', nargs='+', type=str, help='Starting date (YYYY-MM-DD)')

    def handle(self, *args, **options):
        starting_date = options.get('starting_date')
        data_handler = TrainDataHandler(starting_date=starting_date)
        df: pd.DataFrame = data_handler.get_data()
        neural_network = NeuralNetwork()
        neural_network.train(df)
        data_handler._update_last_fetched_date()
