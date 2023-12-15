import pandas as pd

# Base command
from django.core.management.base import BaseCommand

# Project
from red_neuronal.components.data_handler import FitDataHandler
from red_neuronal.components.neural_network.trainer import Trainer
from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO
from red_neuronal.utils.logger import logger


class Command(BaseCommand):
    def handle(self, *args, **options):
        logger.info("Starting FIT_NEURAL_NETWORK command.")
        logger.info("Retrieving data...")
        data_handler = FitDataHandler()
        votes = data_handler.get_votes()
        if votes.empty:
            logger.info("No new votes to fit the neural network. Exiting...")
            return
        authors = data_handler.get_authors()
        projects = data_handler.get_law_projects()
        df: pd.DataFrame = data_handler.merge_data(votes, authors, projects)
        logger.info("The DataFrame has been created. Training neural network...")
        trainer = Trainer()
        trainer.fit(df)
        data_handler._update_last_fetched_date()
