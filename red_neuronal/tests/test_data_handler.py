import shutil
import os
import random
from faker import Faker
import pandas as pd
from red_neuronal.tests.test_helpers.test_case import CustomTestCase
from django.conf import settings
from contextlib import contextmanager


# Project
from red_neuronal.components.neural_network import NeuralNetwork
import red_neuronal.tests.test_helpers.mocks as mck
from red_neuronal.tests.test_helpers.faker import create_fake_df
from red_neuronal.components.data_handler import DataHandler
from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO
from red_neuronal.utils.enums import VoteChoices


class DataHandlerTestCase(CustomTestCase):
    def create_project_ids(self):
        columns = {"project_id": "int"}
        df: pd.DataFrame = create_fake_df(columns, n=self.MAX_TOTAL_PROJECTS, as_dict=False)
        self.project_ids = list(df["project_id"].unique())

    def create_person_ids(self):
        columns = {"person_id": "int"}
        df: pd.DataFrame = create_fake_df(columns, n=self.MAX_TOTAL_PERSONS, as_dict=False)
        self.person_ids = list(df["person_id"].unique())

    def mock_votes_data(self) -> pd.DataFrame:
        MAX_VOTES = 5

        # Generate random votes for each project and person combination
        votes_data = []
        self.total_votes = 0
        for project_id in self.project_ids:
            for _ in range(random.randint(1, MAX_VOTES)):
                person_id = random.choice(self.person_ids)
                vote = random.choice(VoteChoices.values)
                votes_data.append([project_id, person_id, vote])
                self.total_votes += 1

        # Create a dataframe from the generated data
        df = pd.DataFrame(votes_data, columns=["project_id", "person_id", "vote"])

        total_votes = len(votes_data)

        columns = {
            "date": "date",
            "party": "str",
        }
        second_df: pd.DataFrame = create_fake_df(columns, n=total_votes, as_dict=False)
        df["party"] = second_df["party"]
        df["date"] = second_df["date"]
        return df

    def mock_authors_data(self) -> pd.DataFrame:
        MAX_AUTHORS = 3
        authors_data = []
        for project_id in self.project_ids:
            for _ in range(random.randint(1, MAX_AUTHORS)):
                party = Faker().name()
                authors_data.append([project_id, party])

        # Create a dataframe from the generated data
        df = pd.DataFrame(authors_data, columns=["project_id", "party"])
        return df

    def mock_legislators_data(self) -> pd.DataFrame:
        # TODO: de los legisladores sacamos nada más los nombres, podríamos intentar no usarlo pero hay que ver si funciona en la red
        n = len(self.person_ids)
        columns = {
            "person_full_name": "str",
        }
        df: pd.DataFrame = create_fake_df(columns, n=n, as_dict=False)
        df["person_id"] = self.person_ids
        return df

    def mock_projects_data(self) -> pd.DataFrame:
        total_projects = len(self.project_ids)
        columns = {
            "project_title": "short_text",
            "project_text": "text",
            "project_year": "year",
        }
        df: pd.DataFrame = create_fake_df(columns, n=total_projects, as_dict=False)
        df["project_id"] = self.project_ids
        return df

    @contextmanager
    def mock_data(self):
        votes_data = self.mock_votes_data()
        authors_data = self.mock_authors_data()
        legislators_data = self.mock_legislators_data()
        projects_data = self.mock_projects_data()
        with mck.mock_method(DataHandler, "_get_votes", return_value=votes_data):
            with mck.mock_method(DataHandler, "_get_authors", return_value=authors_data):
                with mck.mock_method(DataHandler, "_get_legislators", return_value=legislators_data):
                    with mck.mock_method(DataHandler, "_get_law_projects", return_value=projects_data):
                        yield

    def create_fake_endpoints_data(self):
        endpoints_info = {
            "votes": "",
            "legislators": "",
            "projects": "",
            "authors": "",
        }
        return EndpointsDTO(endpoints_info)

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
        self.create_person_ids()
        self.create_project_ids()
        endpoints_data = self.create_fake_endpoints_data()
        with self.mock_data():
            merged_df: pd.DataFrame = DataHandler().get_data(endpoints_data)
        expected_df_length = self.total_votes
        # The amount of authors per project should not affect the length of the DF,
        # They are joined into a single row per [project_id, person_id]
        self.assertEqual(len(merged_df), expected_df_length)

    def test_neural_network_can_train_with_merged_data(self):
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = 100  # there could be repetitions in the generated data
        self.create_person_ids()
        self.create_project_ids()
        endpoints_data = self.create_fake_endpoints_data()
        with self.mock_data():
            merged_df: pd.DataFrame = DataHandler().get_data(endpoints_data)
        neural_network = NeuralNetwork()
        neural_network.train(merged_df)
        # We just want to check that the training does not fail

    def test_neural_network_cannot_train_with_incorrect_merged_data(self):
        POSSIBLE_EXPECTED_EXCEPTIONS = [KeyError, ValueError]
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = 100  # there could be repetitions in the generated data
        self.create_person_ids()
        self.create_project_ids()
        endpoints_data = self.create_fake_endpoints_data()
        with self.mock_data():
            merged_df: pd.DataFrame = DataHandler().get_data(endpoints_data)
        column_to_drop = random.choice(merged_df.columns)
        print(f"Column to drop: {column_to_drop}")
        merged_df = merged_df.drop(column_to_drop, axis=1)
        neural_network = NeuralNetwork()
        with self.assertRaises(Exception) as context:
            neural_network.train(merged_df)
        self.assertIn(type(context.exception), POSSIBLE_EXPECTED_EXCEPTIONS)
