import requests
import pandas as pd

from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO


class DataHandler:
    session = requests.Session()

    def get_data(self, endpoints_information: EndpointsDTO):
        self.endpoints_information = endpoints_information
        votes = self._get_votes()
        authors = self._get_authors()
        legislators = self._get_legislators()
        projects = self._get_law_projects()
        final_df: pd.DataFrame = self._merge_data(votes, authors, legislators, projects)
        return final_df

    def _get_data_from_source(self, endpoint: str) -> pd.DataFrame:
        response = self.session.get(endpoint)
        raw_data = response.json()
        raw_df: pd.DataFrame = pd.DataFrame(raw_data)
        return raw_df

    def _get_legislators(self):
        raw_df = self._get_data_from_source(self.endpoints_information.legislators_endpoint)
        # TODO: Process the data to leave only the columns we need
        return raw_df

    def _get_votes(self):
        # Expected columns: project_id, person_id, vote, date, party
        raw_df = self._get_data_from_source(self.endpoints_information.votes_endpoint)
        # TODO: Process the data to leave only the columns we need
        return raw_df

    def _get_law_projects(self):
        # Expected columns: project_id, project_title, project_text, project_year
        raw_df = self._get_data_from_source(self.endpoints_information.projects_endpoint)
        # TODO: Process the data to leave only the columns we need
        return raw_df

    def _get_authors(self):
        raw_df = self._get_data_from_source(self.endpoints_information.authors_endpoint)
        # TODO: Process the data to leave only the columns we need
        return raw_df

    def _merge_data(self, votes, authors, legislators, law_projects):
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
