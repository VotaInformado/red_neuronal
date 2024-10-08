import random
import os
import pandas as pd
from datetime import datetime
from django.conf import settings
from django.core.management import call_command
from red_neuronal.components.neural_network.predictor import Predictor


# Project
from red_neuronal.tests.test_helpers.test_case import CustomTestCase
from red_neuronal.components.neural_network.trainer import Trainer
import red_neuronal.tests.test_helpers.mocks as mck
from red_neuronal.components.data_handler import (
    FitDataHandler,
    PredictionDataHandler,
    TrainDataHandler,
    DataHandler,
)
from red_neuronal.tests.test_helpers.faker import create_fake_df, reset_fake_data


class NeuralNetworkTestCase(CustomTestCase):
    def tearDown(self) -> None:
        reset_fake_data()

    def train_neural_network(self):
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = (
            100  # there could be repetitions in the generated data
        )
        mck.create_person_ids(self)
        mck.create_project_ids(self)
        with mck.mock_recoleccion_data(self):
            call_command("train_neural_network")


class DataHandlerTestCase(CustomTestCase):
    def test_ignores_votes_with_null_or_empty_vote_value(self):
        expected_columns = ["project_id", "person", "vote", "date", "party"]
        rows = [
            {
                "project_id": 1,
                "person": "John",
                "vote": "POSITIVE",
                "date": "2022-01-01",
                "party": "ABC",
            },
            {
                "project_id": 2,
                "person": "Jane",
                "vote": None,
                "date": "2022-01-02",
                "party": "XYZ",
            },
            {
                "project_id": 2,
                "person": "Jane",
                "vote": "",
                "date": "2022-01-02",
                "party": "XYZ",
            },
        ]

        expected_df = pd.DataFrame.from_records(data=rows, columns=expected_columns)
        data_handler = DataHandler()

        with mck.mock_method(
            DataHandler, "_get_data_from_source", return_value=expected_df
        ):
            result = data_handler.get_votes()

        self.assertEqual(result.columns.tolist(), expected_columns)
        self.assertEqual(result.shape, (1, 5))
        self.assertIn("POSITIVE", result["vote"].tolist())


class DataRetrievalTestCase(CustomTestCase):
    def test_retrieving_paginated_data(self):
        settings.TOTAL_RESULTS = settings.MAX_TOTAL_PROJECTS = 3000
        settings.MAX_TOTAL_PERSONS = 100
        settings.PAGE_SIZE = 1000
        settings.total_votes = 0
        mck.create_person_ids(settings)
        mck.create_project_ids(settings)
        with mck.mock_method_paginated_data_retrieval(self):
            df: pd.DataFrame = TrainDataHandler().get_votes()
        self.assertEqual(len(df), settings.TOTAL_RESULTS)

    def test_retrieving_non_paginated_data(self):
        settings.TOTAL_RESULTS = settings.MAX_TOTAL_PROJECTS = 3000
        settings.MAX_TOTAL_PERSONS = 100
        settings.PAGE_SIZE = 1000
        settings.total_votes = 0
        mck.create_person_ids(settings)
        mck.create_project_ids(settings)
        with mck.mock_method_paginated_data_retrieval(self):
            df: pd.DataFrame = TrainDataHandler().get_votes()
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
            self.assertEqual(
                row["party_authors"], ";".join(map(str, self.party_authors))
            )

    def test_retrieving_training_data(self):
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = (
            100  # there could be repetitions in the generated data
        )
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
        self.MAX_TOTAL_PROJECTS = (
            100  # there could be repetitions in the generated data
        )
        mck.create_person_ids(self)
        mck.create_project_ids(self)
        with mck.mock_recoleccion_data(self):
            votes = TrainDataHandler().get_votes()
            parties = TrainDataHandler().get_parties()
            authors = TrainDataHandler().get_authors()
            projects = TrainDataHandler().get_law_projects()
            legislators = TrainDataHandler().get_legislators()
            merged_df = TrainDataHandler().merge_data(votes, authors, projects)
        trainer = Trainer()
        trainer.train(merged_df, votes, parties, legislators)
        # We just want to check that the training does not fail

    def test_neural_network_cannot_train_with_incorrect_merged_data(self):
        POSSIBLE_EXPECTED_EXCEPTIONS = [KeyError, ValueError]
        self.MAX_TOTAL_PERSONS = 50  # there could be repetitions in the generated data
        self.MAX_TOTAL_PROJECTS = (
            100  # there could be repetitions in the generated data
        )
        mck.create_person_ids(self)
        mck.create_project_ids(self)
        with mck.mock_recoleccion_data(self):
            votes = TrainDataHandler().get_votes()
            authors = TrainDataHandler().get_authors()
            projects = TrainDataHandler().get_law_projects()
            merged_df = TrainDataHandler().merge_data(votes, authors, projects)

        column_to_drop = random.choice(merged_df.columns)
        merged_df = merged_df.drop(column_to_drop, axis=1)
        trainer = Trainer()
        with self.assertRaises(Exception) as context:
            trainer.train(merged_df, votes, authors, projects)
        self.assertIn(type(context.exception), POSSIBLE_EXPECTED_EXCEPTIONS)


class FitDataHandlerTestCase(NeuralNetworkTestCase):
    def setUp(self):
        today = datetime.today()
        os.makedirs(os.path.dirname(settings.LAST_FETCHED_DATA_DIR), exist_ok=True)
        with open(settings.LAST_FETCHED_DATA_DIR, "w") as json_file:
            today_str = today.strftime("%Y-%m-%d")
            json_file.write(today_str)

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
        merged_df = merged_df.drop(merged_df.columns[1], axis=1)
        trainer = Trainer()
        with self.assertRaises(Exception) as context:
            trainer.fit(merged_df)
        self.assertIn(type(context.exception), POSSIBLE_EXPECTED_EXCEPTIONS)


class PredictionDataHandlerTestCase(NeuralNetworkTestCase):
    def setUp(self):
        today = datetime.today()
        os.makedirs(os.path.dirname(settings.LAST_FETCHED_DATA_DIR), exist_ok=True)
        with open(settings.LAST_FETCHED_DATA_DIR, "w") as json_file:
            today_str = today.strftime("%Y-%m-%d")
            json_file.write(today_str)

    def train_neural_network(self, total_persons=None):
        self.MAX_TOTAL_PERSONS = total_persons or 50
        self.MAX_TOTAL_PROJECTS = (
            100  # there could be repetitions in the generated data
        )
        allow_repetitions = False if total_persons else True
        mck.create_person_ids(self, allow_repetitions=allow_repetitions)
        mck.create_project_ids(self)
        with mck.mock_recoleccion_data(self):
            call_command("train_neural_network")

    def mock_legislators_data(self, total_legislators=1, use_existing=False):
        columns = {"person": int}
        df = create_fake_df(columns, n=total_legislators, as_dict=False)
        if use_existing:
            if len(self.person_ids) == total_legislators:
                df["person"] = self.person_ids
            else:
                df["person"] = [
                    random.choice(self.person_ids) for _ in range(total_legislators)
                ]
        return df

    def mock_authors_data(self, total_parties=1, use_existing=False):
        columns = {"party": int}
        df = create_fake_df(columns, n=total_parties, as_dict=False)
        if use_existing:
            df["party"] = [random.choice(self.parties) for _ in range(total_parties)]
        return df

    def mock_project_data(self, use_existing=False):
        columns = {
            "project": int,
            "project_year": "year",
            "project_title": "short_text",
            "project_text": "text",
        }
        df = create_fake_df(columns, n=1, as_dict=False)
        if use_existing:
            df["project"] = self.project_ids[0]
        return df

    def test_creating_prediction_data(self):
        TOTAL_LEGISLATORS = 72  # there could be repetitions in the generated data
        EXPECTED_COLUMNS = {
            "voter_id",
            "party_authors",
            "project",
            "project_year",
            "project_title",
            "project_text",
        }
        self.project_ids = [1]
        authors_data = self.mock_authors_data(total_parties=3)
        project_data = self.mock_project_data()
        legislators_data = self.mock_legislators_data(
            total_legislators=TOTAL_LEGISLATORS
        )
        prediction_data = {
            "authors": authors_data,
            "legislators": legislators_data,
            "project": project_data,
        }
        merged_df: pd.DataFrame = PredictionDataHandler.get_prediction_df(
            prediction_data
        )
        expected_df_length = TOTAL_LEGISLATORS
        self.assertEqual(len(merged_df), expected_df_length)
        df_columns = set(merged_df.columns)
        self.assertEqual(df_columns, EXPECTED_COLUMNS)

    def test_neural_network_can_predict_with_project_prediction_data(self):
        self.use_all_persons = (
            True  # Makes sure all legislator_ids are used when generating data
        )
        TOTAL_LEGISLATORS = 72  # there could be repetitions in the generated data
        self.train_neural_network(total_persons=TOTAL_LEGISLATORS)
        authors_data = self.mock_authors_data(total_parties=3, use_existing=True)
        legislators_data = self.mock_legislators_data(
            total_legislators=TOTAL_LEGISLATORS, use_existing=True
        )
        project_data = self.mock_project_data(use_existing=True)
        prediction_data = {
            "authors": authors_data,
            "legislators": legislators_data,
            "project": project_data,
        }
        merged_df: pd.DataFrame = PredictionDataHandler.get_prediction_df(
            prediction_data
        )
        predictor = Predictor()
        predictions = predictor.predict(merged_df)
        self.assertEqual(len(predictions), TOTAL_LEGISLATORS)

    def test_neural_network_can_predict_with_legislator_prediction_data(self):
        self.use_all_persons = (
            True  # Makes sure all legislator_ids are used when generating data
        )
        TOTAL_LEGISLATORS = 1  # there could be repetitions in the generated data
        self.train_neural_network()
        authors_data = self.mock_authors_data(total_parties=3, use_existing=True)
        legislators_data = self.mock_legislators_data(
            total_legislators=TOTAL_LEGISLATORS, use_existing=True
        )
        project_data = self.mock_project_data(use_existing=True)
        prediction_data = {
            "authors": authors_data,
            "legislators": legislators_data,
            "project": project_data,
        }

        merged_df: pd.DataFrame = PredictionDataHandler.get_prediction_df(
            prediction_data
        )
        predictor = Predictor()
        predictions = predictor.predict(merged_df)
        self.assertEqual(len(predictions), TOTAL_LEGISLATORS)
