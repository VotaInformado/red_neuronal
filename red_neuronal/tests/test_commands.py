import time
import os
from faker import Faker
import pandas as pd
from django.core.management import call_command
from django.conf import settings

# Project
from red_neuronal.tests.test_helpers.test_case import CustomTestCase
import red_neuronal.tests.test_helpers.mocks as mck
from red_neuronal.components.neural_network import NeuralNetwork


class TrainingCommandTestcase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = 100  # there could be repetitions in the generated data
        mck.create_person_ids(self)
        mck.create_project_ids(self)

    def files_created(self) -> bool:
        dirs = [NeuralNetwork.MODEL_FILE_SAVING_DIR, NeuralNetwork.WEIGHTS_SAVING_DIR]
        for dir in dirs:
            if not os.path.exists(dir):
                return False
        return True

    def test_neural_network_training_command(self):
        with mck.mock_recoleccion_data(self):
            call_command("train_neural_network")
        self.assertTrue(self.files_created())
