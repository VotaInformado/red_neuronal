import random
from faker import Faker
import pandas as pd
from red_neuronal.tests.test_helpers.test_case import CustomTestCase


# Project
from red_neuronal.components.neural_network import NeuralNetwork
import red_neuronal.tests.test_helpers.mocks as mck
from red_neuronal.tests.test_helpers.faker import create_fake_df
from red_neuronal.components.data_handler import DataHandler
from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO
from red_neuronal.utils.enums import VoteChoices


class DataHandlerTestCase(CustomTestCase):
    def test_authors_flattening(self):
        # Checks that author compression (grouped by [project_id, person_id]) works correctly
        self.project_ids = [15, 20]
        self.party_authors = [1, 2]
        df_info = []
        for party in self.party_authors:
            for project_id in self.project_ids:
                df_info.append([project_id, party])

        df: pd.DataFrame = pd.DataFrame(df_info, columns=["project_id", "party"])
        data_handler = DataHandler()
        flattened_df = data_handler._flatten_party_authors(df)
        for _, row in flattened_df.iterrows():
            self.assertEqual(row["party_authors"], ";".join(map(str, self.party_authors)))

    def test_retrieving_votes_data(self):
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = 100  # there could be repetitions in the generated data
        mck.create_person_ids(self)
        mck.create_project_ids(self)
        endpoints_data = mck.create_fake_endpoints_data(self)
        with mck.mock_recoleccion_data(self):
            merged_df: pd.DataFrame = DataHandler().get_data(endpoints_data)
        expected_df_length = self.total_votes
        # The amount of authors per project should not affect the length of the DF,
        # They are joined into a single row per [project_id, person_id]
        self.assertEqual(len(merged_df), expected_df_length)

    def test_neural_network_can_train_with_merged_data(self):
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = 100  # there could be repetitions in the generated data
        mck.create_person_ids(self)
        mck.create_project_ids(self)
        endpoints_data = mck.create_fake_endpoints_data(self)
        with mck.mock_recoleccion_data(self):
            merged_df: pd.DataFrame = DataHandler().get_data(endpoints_data)
        neural_network = NeuralNetwork()
        neural_network.train(merged_df)
        # We just want to check that the training does not fail

    def test_neural_network_cannot_train_with_incorrect_merged_data(self):
        POSSIBLE_EXPECTED_EXCEPTIONS = [KeyError, ValueError]
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = 100  # there could be repetitions in the generated data
        mck.create_person_ids(self)
        mck.create_project_ids(self)
        endpoints_data = mck.create_fake_endpoints_data(self)
        with mck.mock_recoleccion_data(self):
            merged_df: pd.DataFrame = DataHandler().get_data(endpoints_data)
        column_to_drop = random.choice(merged_df.columns)
        print(f"Column to drop: {column_to_drop}")
        merged_df = merged_df.drop(column_to_drop, axis=1)
        neural_network = NeuralNetwork()
        with self.assertRaises(Exception) as context:
            neural_network.train(merged_df)
        self.assertIn(type(context.exception), POSSIBLE_EXPECTED_EXCEPTIONS)
