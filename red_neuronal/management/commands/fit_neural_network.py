import pandas as pd

# Base command
from django.core.management.base import BaseCommand

# Project
from red_neuronal.components.data_handler import FitDataHandler
from red_neuronal.components.neural_network.trainer import Trainer
from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO


class Command(BaseCommand):
    def handle(self, *args, **options):
        data_handler = FitDataHandler()
        df: pd.DataFrame = data_handler.get_data()
        trainer = Trainer()
        trainer.train(df)
        data_handler._update_last_fetched_date()
