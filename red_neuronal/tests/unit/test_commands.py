import time
import os
from freezegun import freeze_time
from django.core.management import call_command
from django.conf import settings

# Project
from red_neuronal.tests.test_helpers.test_case import CustomTestCase
import red_neuronal.tests.test_helpers.mocks as mck
from red_neuronal.components.neural_network.neural_network import NeuralNetwork
from red_neuronal.components.data_handler import DataHandler


class TrainingCommandTestcase(CustomTestCase):
    def setUp(self):
        self.data_handler = DataHandler()
        super().setUp()
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = 100  # there could be repetitions in the generated data
        mck.create_person_ids(self)
        mck.create_project_ids(self)

    def files_created(self) -> bool:
        dirs = [NeuralNetwork.MODEL_KERAS_SAVING_DIR]
        for dir in dirs:
            if not os.path.exists(dir):
                return False
        return True

    def test_neural_network_training_command_without_filters(self):
        with mck.mock_recoleccion_data(self):
            call_command("train_neural_network")
        self.assertTrue(self.files_created())

    def test_neural_network_training_command_with_filters(self):
        STARTING_DATE = "2020-01-01"
        with mck.mock_recoleccion_data(self):
            call_command("train_neural_network", f"--starting-date={STARTING_DATE}")
        self.assertTrue(self.files_created())

    def test_neural_network_fit_command(self):
        today = "2020-01-01"
        tommorrow = "2020-01-02"
        with freeze_time(today):
            with mck.mock_recoleccion_data(self):
                call_command("train_neural_network")
        self.assertTrue(self.files_created())
        first_last_fetched_date = self.data_handler._get_last_fetched_date()
        with freeze_time(tommorrow):
            with mck.mock_recoleccion_data(self, use_existing_data=True):
                call_command("fit_neural_network")
        second_last_fetched_date = self.data_handler._get_last_fetched_date()
        self.assertGreater(second_last_fetched_date, first_last_fetched_date)
