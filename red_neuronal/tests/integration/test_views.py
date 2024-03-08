from django.core.management import call_command
import random

# Project
from red_neuronal.tests.test_helpers.test_case import CustomAPITestCase
import red_neuronal.tests.test_helpers.mocks as mck
from red_neuronal.tests.test_helpers.faker import create_fake_df
from red_neuronal.utils.enums import VoteChoices


class PredictionTestCase(CustomAPITestCase):
    def train_neural_network(self, total_persons=None):
        self.MAX_TOTAL_PERSONS = total_persons or 50
        self.MAX_TOTAL_PROJECTS = 100  # there could be repetitions in the generated data
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
                df["person"] = [random.choice(self.person_ids) for _ in range(total_legislators)]
        return df.to_dict(orient="records")

    def mock_authors_data(self, total_parties=1, use_existing=False):
        columns = {"party": int}
        df = create_fake_df(columns, n=total_parties, as_dict=False)
        if use_existing:
            df["party"] = [random.choice(self.parties) for _ in range(total_parties)]
        return df.to_dict(orient="records")

    def mock_project_data(self, use_existing=False):
        columns = {"project_id": int, "project_year": "year", "project_title": "short_text", "project_text": "text"}
        df = create_fake_df(columns, n=1, as_dict=False)
        if use_existing:
            df["project"] = self.project_ids[0]
        return df.to_dict(orient="records")

    def test_legislator_prediction_endpoint(self):
        self.train_neural_network()
        url = "/api/predict-legislator-vote/"
        authors_data = self.mock_authors_data(total_parties=3, use_existing=True)
        legislator_data = self.mock_legislators_data(total_legislators=1, use_existing=True)
        project_data = self.mock_project_data(use_existing=True)
        legislator_data, project_data = legislator_data[0], project_data[0]
        payload = {
            "authors": authors_data,
            "legislator": legislator_data,
            "project": project_data,
        }
        response = self.client.post(url, payload, format="json")

        self.assertEqual(response.status_code, 200)
        response_legislator = response.json()["legislator"]
        response_vote = response.json()["vote"]
        self.assertEqual(response_legislator, legislator_data["person"])
        vote_choices = [choice.value for choice in VoteChoices]
        self.assertIn(response_vote, vote_choices)

    def test_project_prediction_endpoint(self):
        self.use_all_persons = True
        TOTAL_LEGISLATORS = 72
        self.train_neural_network(total_persons=TOTAL_LEGISLATORS)
        url = "/api/predict-project-votes/"
        authors_data = self.mock_authors_data(total_parties=3, use_existing=True)
        legislator_data = self.mock_legislators_data(total_legislators=TOTAL_LEGISLATORS, use_existing=True)
        project_data = self.mock_project_data(use_existing=True)
        project_data = project_data[0]
        payload = {
            "authors": authors_data,
            "legislators": legislator_data,
            "project": project_data,
        }
        response = self.client.post(url, payload, format="json")

        self.assertEqual(response.status_code, 200, response.json())
        response_legislators = response.json()
        self.assertEqual(len(response_legislators), TOTAL_LEGISLATORS)
        vote_choices = [choice.value for choice in VoteChoices]
        for response_legislator in response_legislators:
            self.assertIn(response_legislator["legislator"], self.person_ids)
            self.assertIn(response_legislator["vote"], vote_choices)
