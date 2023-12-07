import requests
import pandas as pd
from datetime import datetime
from django.conf import settings
import os
from tqdm import tqdm
from rest_framework import status
from requests.models import Response

# Project
from red_neuronal.utils.logger import logger


class DataHandler:
    session = requests.Session()

    def get(self, endpoint, filters={}):
        ACCEPTABLE_RESPONSES = [status.HTTP_200_OK, status.HTTP_201_CREATED]
        response = self.session.get(url=endpoint, params=filters)
        if response.status_code not in ACCEPTABLE_RESPONSES:
            raise Exception(f"Error fetching data from {endpoint}. Response: {response}")
        return response

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
        logger.info(f"Making first request to {endpoint}...")
        logger.info(f"Using filters: {filters}")
        base_response = self.get(endpoint=endpoint, filters=filters)
        if self._is_paginated_response(base_response):
            return self._get_paginated_data(base_response)
        return self._get_non_paginated_data(base_response)

    def _is_paginated_response(self, response: Response) -> bool:
        PAGINATED_RESPONSE_KEYS = ["next", "previous", "results", "count"]
        response_json = response.json()
        if not isinstance(response_json, dict):
            is_paginated = False
        else:
            response_keys = response_json.keys()
            is_paginated = sorted(response_keys) == sorted(PAGINATED_RESPONSE_KEYS)
            logger.info(f"Response is paginated: {is_paginated}")
        return is_paginated

    def _get_non_paginated_data(self, endpoint: str, filters: dict) -> pd.DataFrame:
        response = self.get(endpoint=endpoint, filters=filters)
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
        logger.info("Response is paginated, getting all data...")
        response_json = base_response.json()
        received_data = response_json["results"]
        total_results = response_json["count"]
        loops_needed = self._calculate_loops_needed(total_results)
        logger.info(f"{loops_needed} loops will be needed to fetch {total_results} results")
        if response_json["next"] is None:
            received_data = response_json["results"]
        else:
            for _ in tqdm(range(loops_needed)):
                endpoint = response_json["next"]
                if endpoint is None:
                    break
                response = self.get(endpoint)
                response_json = response.json()
                response_results = response_json["results"]
                received_data.extend(response_results)
        logger.info("All data received, creating DataFrame...")
        df: pd.DataFrame = pd.DataFrame(received_data)
        return df

    def _get_legislators(self):
        url = settings.LEGISLATORS_DATA_ENDPOINT
        raw_df = self._get_data_from_source(url)
        # TODO: Process the data to leave only the columns we need
        return raw_df

    def _get_votes(self):
        url = settings.VOTES_DATA_ENDPOINT
        # Expected columns: project_id, person, vote, date, party
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
        # Expected columns: project, person, party
        raw_df = self._get_data_from_source(url)
        # TODO: Process the data to leave only the columns we need
        return raw_df

    def _merge_data(self, votes, authors, law_projects):
        # TODO: convertir project de law_projects a int
        # Result DF: voter_id, vote, project, project_title, project_text, project_year, party_authors
        # we remove all votes without project

        votes = votes.dropna(subset=["project"])
        votes["project"] = votes["project"].astype(int)
        law_projects["project"] = law_projects["project_id"].astype(int)
        votes_and_projects = pd.merge(votes, law_projects, on="project")
        votes_and_projects = votes_and_projects.drop(["date"], axis=1)
        votes_and_projects = votes_and_projects.rename(columns={"person": "voter_id", "party": "voter_party"})
        authors = self._flatten_party_authors(authors)
        votes_projects_and_authors = pd.merge(votes_and_projects, authors, on="project")
        # Es un poco complicado mergear para tener el nombre de legisladores, y no aporta nada, por ahora, queda así
        # TODO: ver si el encoder puede recibir ids en vez de nombres
        final_df = votes_projects_and_authors.drop(["voter_party"], axis=1)  # La red neuronal no lo está usando
        return final_df

    def _flatten_party_authors(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.groupby(["project"])["party"].apply(lambda x: ";".join(map(str, x))).reset_index()
        df.rename(columns={"party": "party_authors"}, inplace=True)
        df = df.drop_duplicates(subset=["project"])
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

    def _get_filters_for_endpoint(self, endpoint: str) -> dict:
        # Only used when self.starting_date is not None
        if endpoint == settings.VOTES_DATA_ENDPOINT:
            filters = {"date__gte": self.starting_date}
        elif endpoint == settings.PROJECTS_DATA_ENDPOINT:
            filters = {"publication_date__gte": self.starting_date}
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


class PredictionDataHandler(DataHandler):
    @classmethod
    def _flatten_party_authors(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.groupby(df.columns.difference(["party"]).tolist(), as_index=False)["party"].agg(
            lambda x: ";".join(map(str, x))
        )
        df.rename(columns={"party": "party_authors"}, inplace=True)
        return df

    @classmethod
    def replicate_columns(cls, multi_row_df, single_row_df):
        # Repeat the single-row dataframe df2 to match the number of rows in df1
        repeated_df2 = pd.concat([single_row_df] * len(multi_row_df), ignore_index=True)
        # Concatenate along columns (axis=1)
        result_df = pd.concat([multi_row_df, repeated_df2], axis=1)
        return result_df

    @classmethod
    def get_prediction_df(cls, raw_data: dict):
        raw_authors = raw_data["authors"]
        raw_legislators = raw_data.get("legislators") or [raw_data.get("legislator")]
        raw_project = raw_data["project"]
        raw_project = raw_project if isinstance(raw_project, list) else [raw_project]
        project_df = pd.DataFrame(raw_project)
        authors_df = pd.DataFrame(raw_authors)
        legislators_df = pd.DataFrame(raw_legislators)
        project_with_authors = cls.replicate_columns(authors_df, project_df)
        # we remove the one row with party nan
        project_with_authors = cls._flatten_party_authors(project_with_authors)
        merged_df = cls.replicate_columns(legislators_df, project_with_authors)
        merged_df = merged_df.rename(columns={"person": "voter_id"})
        return merged_df
