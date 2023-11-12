import random
import os
import pandas as pd
from datetime import datetime
from django.conf import settings
from django.core.management import call_command


# Project
from red_neuronal.tests.test_helpers.test_case import CustomTestCase
from red_neuronal.components.neural_network.trainer import Trainer
import red_neuronal.tests.test_helpers.mocks as mck
from red_neuronal.tests.test_helpers.faker import create_fake_df
from red_neuronal.components.data_handler import FitDataHandler, TrainDataHandler


class DataRetrievalTestCase(CustomTestCase):
    def test_retrieving_paginated_data(self):
        settings.TOTAL_RESULTS = settings.MAX_TOTAL_PROJECTS = 3000
        settings.MAX_TOTAL_PERSONS = 100
        settings.PAGE_SIZE = 1000
        settings.total_votes = 0
        mck.create_person_ids(settings)
        mck.create_project_ids(settings)
        with mck.mock_method_paginated_data_retrieval(self):
            df: pd.DataFrame = TrainDataHandler()._get_votes()
        self.assertEqual(len(df), settings.TOTAL_RESULTS)

    def test_retrieving_non_paginated_data(self):
        settings.TOTAL_RESULTS = settings.MAX_TOTAL_PROJECTS = 3000
        settings.MAX_TOTAL_PERSONS = 100
        settings.PAGE_SIZE = 1000
        settings.total_votes = 0
        mck.create_person_ids(settings)
        mck.create_project_ids(settings)
        with mck.mock_method_paginated_data_retrieval(self):
            df: pd.DataFrame = TrainDataHandler()._get_votes()
        self.assertEqual(len(df), settings.TOTAL_RESULTS)


class TrainDataHandlerTestCase(CustomTestCase):
    def test_authors_flattening(self):
        # Checks that author compression (grouped by [project_id, person_id]) works correctly
        self.project_ids = [15, 20]
        self.party_authors = [1, 2]
        df_info = []
        for party in self.party_authors:
            for project_id in self.project_ids:
                df_info.append([project_id, party])

        df: pd.DataFrame = pd.DataFrame(df_info, columns=["project", "party"])
        data_handler = TrainDataHandler()
        flattened_df = data_handler._flatten_party_authors(df)
        for _, row in flattened_df.iterrows():
            self.assertEqual(row["party_authors"], ";".join(map(str, self.party_authors)))

    def test_retrieving_training_data(self):
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = 100  # there could be repetitions in the generated data
        mck.create_person_ids(self)
        mck.create_project_ids(self)
        self.total_votes = 0
        with mck.mock_recoleccion_data(self):
            merged_df: pd.DataFrame = TrainDataHandler().get_data()
        expected_df_length = self.total_votes
        # The amount of authors per project should not affect the length of the DF,
        # They are joined into a single row per [project_id, person_id]
        self.assertEqual(len(merged_df), expected_df_length)

    def test_neural_network_can_train_with_merged_data(self):
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = 100  # there could be repetitions in the generated data
        mck.create_person_ids(self)
        mck.create_project_ids(self)
        with mck.mock_recoleccion_data(self):
            merged_df: pd.DataFrame = TrainDataHandler().get_data()
        trainer = Trainer()
        trainer.train(merged_df)
        # We just want to check that the training does not fail

    def test_neural_network_cannot_train_with_incorrect_merged_data(self):
        POSSIBLE_EXPECTED_EXCEPTIONS = [KeyError, ValueError]
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = 100  # there could be repetitions in the generated data
        mck.create_person_ids(self)
        mck.create_project_ids(self)
        with mck.mock_recoleccion_data(self):
            merged_df: pd.DataFrame = TrainDataHandler().get_data()
        column_to_drop = random.choice(merged_df.columns)
        print(f"Column to drop: {column_to_drop}")
        merged_df = merged_df.drop(column_to_drop, axis=1)
        trainer = Trainer()
        with self.assertRaises(Exception) as context:
            trainer.train(merged_df)
        self.assertIn(type(context.exception), POSSIBLE_EXPECTED_EXCEPTIONS)


class FitDataHandlerTestCase(CustomTestCase):
    def setUp(self):
        today = datetime.today()
        os.makedirs(os.path.dirname(settings.LAST_FETCHED_DATA_DIR), exist_ok=True)
        with open(settings.LAST_FETCHED_DATA_DIR, "w") as json_file:
            today_str = today.strftime("%Y-%m-%d")
            json_file.write(today_str)

    def train_neural_network(self):
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = 100  # there could be repetitions in the generated data
        mck.create_person_ids(self)
        mck.create_project_ids(self)
        with mck.mock_recoleccion_data(self):
            call_command("train_neural_network")

    def test_retrieving_fitting_data(self):
        self.train_neural_network()
        self.total_votes = 0
        with mck.mock_recoleccion_data(self):
            merged_df: pd.DataFrame = FitDataHandler().get_data()
        expected_df_length = self.total_votes
        # The amount of authors per project should not affect the length of the DF,
        # They are joined into a single row per [project_id, person_id]
        self.assertEqual(len(merged_df), expected_df_length)

    def test_neural_network_can_be_fit_with_merged_data(self):
        self.train_neural_network()
        self.NEW_VOTES = 100
        self.NEW_PROJECTS = 100
        self.NEW_AUTHORS = 100
        with mck.mock_recoleccion_data(self, use_existing_data=True):
            merged_df: pd.DataFrame = FitDataHandler().get_data()
        trainer = Trainer()
        trainer.fit(merged_df)
        # We just want to check that the training does not fail

    def test_neural_network_cannot_be_fit_with_incorrect_merged_data(self):
        self.NEW_VOTES = 50
        self.NEW_PROJECTS = 50
        self.NEW_AUTHORS = 50
        POSSIBLE_EXPECTED_EXCEPTIONS = [KeyError, ValueError]
        self.train_neural_network()
        with mck.mock_recoleccion_data(self, use_existing_data=True):
            merged_df: pd.DataFrame = FitDataHandler().get_data()
        column_to_drop = random.choice(merged_df.columns)
        merged_df = merged_df.drop(column_to_drop, axis=1)
        trainer = Trainer()
        with self.assertRaises(Exception) as context:
            trainer.fit(merged_df)
        self.assertIn(type(context.exception), POSSIBLE_EXPECTED_EXCEPTIONS)
