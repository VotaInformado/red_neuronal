import requests
import pandas as pd
from datetime import datetime
from django.conf import settings
import os
from tqdm import tqdm
from requests.models import Response
import logging

logger = logging.getLogger(__name__)


class DataHandler:
    session = requests.Session()

    def get_data(self):
        votes = self._get_votes()
        authors = self._get_authors()
        # legislators = self._get_legislators()
        projects = self._get_law_projects()
        final_df: pd.DataFrame = self._merge_data(votes, authors, projects)
        return final_df

    def _get_data_from_source(self, endpoint: str, filters={}) -> pd.DataFrame:
        page_size_filter = {"page_size": settings.DEFAULT_PAGE_SIZE}
        filters.update(page_size_filter)
        base_response = self.session.get(url=endpoint, data=filters)
        if self._is_paginated_response(base_response):
            return self._get_paginated_data(base_response)
        return self._get_non_paginated_data(base_response)

    def _is_paginated_response(self, response: Response) -> bool:
        PAGINATED_RESPONSE_KEYS = ["next", "previous", "results", "count"]
        response_json = response.json()
        response_keys = response_json.keys()
        return sorted(response_keys) == sorted(PAGINATED_RESPONSE_KEYS)

    def _get_non_paginated_data(self, endpoint: str, filters: dict) -> pd.DataFrame:
        response = self.session.get(url=endpoint, data=filters)
        raw_data = response.json()
        raw_df: pd.DataFrame = pd.DataFrame(raw_data)
        return raw_df

    def _calculate_loops_needed(self, total_results: int) -> int:
        # We need this only to be able to use tqdm. In long responses, tqdm is useful to see the progress.
        loops_needed = total_results // settings.DEFAULT_PAGE_SIZE
        if total_results % settings.DEFAULT_PAGE_SIZE != 0:
            # If there is a remainder, we need to do one more loop
            loops_needed += 1
        return loops_needed

    def _get_paginated_data(self, base_response: Response) -> pd.DataFrame:
        response_json = base_response.json()
        received_data = response_json["results"]
        total_results = response_json["count"]
        loops_needed = self._calculate_loops_needed(total_results)
        if response_json["next"] is None:
            received_data = response_json["results"]
        else:
            for _ in tqdm(range(loops_needed)):
                endpoint = response_json["next"]
                if endpoint is None:
                    break
                response = self.session.get(url=endpoint)
                response_json = response.json()
                response_results = response_json["results"]
                received_data.extend(response_results)
        df: pd.DataFrame = pd.DataFrame(received_data)
        return df

    def _get_legislators(self):
        url = settings.LEGISLATORS_DATA_ENDPOINT
        raw_df = self._get_data_from_source(url)
        # TODO: Process the data to leave only the columns we need
        return raw_df

    def _get_votes(self):
        url = settings.VOTES_DATA_ENDPOINT
        # Expected columns: project_id, person_id, vote, date, party
        raw_df = self._get_data_from_source(url)
        # TODO: Process the data to leave only the columns we need
        return raw_df

    def _get_law_projects(self):
        url = settings.PROJECTS_DATA_ENDPOINT
        # Expected columns: project_id, project_title, project_text, project_year
        raw_df = self._get_data_from_source(url)
        # TODO: Process the data to leave only the columns we need
        return raw_df

    def _get_authors(self):
        url = settings.AUTHORS_DATA_ENDPOINT
        # Expected columns: project_id, person_id, party
        raw_df = self._get_data_from_source(url)
        # TODO: Process the data to leave only the columns we need
        return raw_df

    def _merge_data(self, votes, authors, law_projects):
        # Result DF: voter_id, vote, project_id, project_title, project_text, project_year, party_authors
        votes_and_projects = pd.merge(votes, law_projects, on="project_id")
        votes_and_projects = votes_and_projects.drop(["date"], axis=1)
        votes_and_projects = votes_and_projects.rename(columns={"person_id": "voter_id", "party": "voter_party"})
        authors = self._flatten_party_authors(authors)
        votes_projects_and_authors = pd.merge(votes_and_projects, authors, on="project_id")
        # Es un poco complicado mergear para tener el nombre de legisladores, y no aporta nada, por ahora, queda así
        # TODO: ver si el encoder puede recibir ids en vez de nombres
        final_df = votes_projects_and_authors.drop(["voter_party"], axis=1)  # La red neuronal no lo está usando
        return final_df

    def _flatten_party_authors(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.groupby(["project_id"])["party"].apply(lambda x: ";".join(map(str, x))).reset_index()
        df.rename(columns={"party": "party_authors"}, inplace=True)
        df = df.drop_duplicates(subset=["project_id"])
        return df

    def _get_last_fetched_date(self):
        # Training command saves the last fetched date, so the file should exist
        with open(settings.LAST_FETCHED_DATA_DIR, "r") as file:
            last_fetched_date = file.read()
        return last_fetched_date

    def _update_last_fetched_date(self):
        os.makedirs(settings.DATA_HANDLER_FILES_DIR, exist_ok=True)
        with open(settings.LAST_FETCHED_DATA_DIR, "w") as f:
            today = datetime.today()
            today_str = today.strftime("%Y-%m-%d")
            f.write(today_str)


class TrainDataHandler(DataHandler):
    def __init__(self, starting_date: str = None):
        self.starting_date = starting_date

    def _get_filters_for_endpoint(self, endpoint: str, date: str) -> dict:
        if endpoint == settings.VOTES_DATA_ENDPOINT:
            filters = {"date__gte": date}
        elif endpoint == settings.PROJECTS_DATA_ENDPOINT:
            filters = {"publication_date__gte": date}
        elif endpoint == settings.AUTHORS_DATA_ENDPOINT:
            filters = {}  # filters here are not needed
        return filters

    def _get_data_from_source(self, endpoint: str) -> pd.DataFrame:
        if self.starting_date:
            filters = self._get_filters_for_endpoint(endpoint)
        else:
            filters = {}
        return super()._get_data_from_source(endpoint, filters)


class FitDataHandler(DataHandler):
    def __init__(self):
        self.last_fetched_date = self._get_last_fetched_date()

    def _get_data_from_source(self, endpoint: str) -> pd.DataFrame:
        filters = {"created_at__gte": self.last_fetched_date}
        return super()._get_data_from_source(endpoint, filters)
