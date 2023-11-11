# External libraries
from datetime import datetime
import pandas as pd
import random
import requests
import time
from django.conf import settings
from contextlib import contextmanager
from django.test import TestCase
from faker import Faker

# Unittest mock
from unittest.mock import MagicMock, patch

# Env variables
from django.conf import settings

# Project
from red_neuronal.tests.test_helpers.faker import create_fake_df
from red_neuronal.utils.enums import VoteChoices
from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO
from red_neuronal.components.data_handler import DataHandler, TrainDataHandler

DEFAULT_NEW_OBJECTS = 50


class FakeClass:
    def fake_method(test_case: TestCase):
        pass


class FakeResponse:
    def __init__(test_case, status_code, content):
        test_case.status_code = status_code
        test_case.content = content


class MockResponse(MagicMock):
    def __init__(self, status_code, data, headers):
        super().__init__()
        self.status_code = status_code
        self.data = data
        self.headers = headers
        self.content = data
        self.error = {"GENERIC_EXCEPTION"}

    def json(self):
        return self.data


def is_internal_mock_enabled():
    return getattr(settings, "INTERNAL_MOCK_ENABLED", True)


def mock_method(mocked_class, method_name, return_value=None, new_callable=None):
    """Mocks a method of a class"""
    mocked_class_name = mocked_class.__name__
    if is_internal_mock_enabled():
        patcher = patch.object(mocked_class, method_name, return_value=return_value, new_callable=new_callable)
        return patcher
    else:
        # This function needs to return a context manager, so this must be used
        patcher = patch.object(FakeClass, "fake_method", return_value=return_value)
        return patcher


def mock_method_side_effect(mocked_class, method_name, side_effect):
    """Mocks a method of a class"""
    # Does not deppend on internal_mock_enabled, since we'll always want the side effects to happen
    patcher = patch.object(mocked_class, method_name, side_effect=side_effect)
    return patcher


def mock_class_attribute(mocked_class, attribute_name, new_attribute_value):
    """Mocks an attribute of a class"""
    mocked_class_name = mocked_class.__name__
    if is_internal_mock_enabled():
        patcher = patch.object(mocked_class, attribute_name, new=new_attribute_value)
        return patcher
    else:
        # This function needs to return a context manager, so this must be used
        patcher = patch.object(FakeClass, "fake_method", return_value=new_attribute_value)
        return patcher


def mock_data_source_json(src_file):
    """Mocks a data source json file"""
    base_dir = "recoleccion/tests/test_helpers/files/"
    file_dir = base_dir + src_file
    return pd.read_json(file_dir)


def mock_data_source_csv(src_file):
    """Mocks a data source json file"""
    base_dir = "recoleccion/tests/test_helpers/files/"
    file_dir = base_dir + src_file
    return pd.read_csv(file_dir)


def get_file_data_length(src_file):
    if ".json" in src_file:
        data = mock_data_source_json(src_file)
    else:
        data = mock_data_source_csv(src_file)
    return len(data)


@contextmanager
def mock_recoleccion_data(test_case: TestCase, use_existing_data: bool = False):
    votes_data = mock_votes_data(test_case, use_existing_data)
    authors_data = mock_authors_data(test_case, use_existing_data)
    # legislators_data = mock_legislators_data(test_case, existing_data)
    projects_data = mock_projects_data(test_case, use_existing_data)
    with mock_method(DataHandler, "_get_votes", return_value=votes_data):
        with mock_method(DataHandler, "_get_authors", return_value=authors_data):
            with mock_method(DataHandler, "_get_law_projects", return_value=projects_data):
                yield


def mock_votes_data(test_case: TestCase, use_existing_data: bool) -> pd.DataFrame:
    if use_existing_data:
        return mock_vote_existing_data(test_case)
    else:
        return mock_new_votes_data(test_case)


def mock_vote_existing_data(test_case: TestCase):
    new_votes = getattr(test_case, "NEW_VOTES", DEFAULT_NEW_OBJECTS)
    existing_data_columns = ["project", "person", "vote"]
    selected_data = test_case.existing_votes[existing_data_columns]
    limited_data = selected_data.head(new_votes)

    other_columns = {
        "date": "date",
        "party": "str",
    }
    second_df: pd.DataFrame = create_fake_df(other_columns, n=new_votes, as_dict=False)
    limited_data["party"] = second_df["party"]
    limited_data["date"] = second_df["date"]
    return limited_data


def mock_new_votes_data(test_case: TestCase, total_results=None):
    VOTES_PER_PROJECT = 5  # setear un TOTAL_RESULTS múltiplo de VOTES_PER_PROJECT
    # Generate random votes for each project and person combination
    votes_data = []
    projects_len = total_results // VOTES_PER_PROJECT if total_results else len(test_case.project_ids)
    total_projects = test_case.project_ids[:projects_len] if test_case else settings.project_ids[:projects_len]
    for project_id in total_projects:
        for _ in range(VOTES_PER_PROJECT):
            person_id = random.choice(test_case.person_ids)
            vote = random.choice(VoteChoices.values)
            votes_data.append([project_id, person_id, vote])
            if getattr(test_case, "total_votes", None) is not None:
                test_case.total_votes += 1

    # Create a dataframe from the generated data
    df = pd.DataFrame(votes_data, columns=["project", "person", "vote"])

    total_votes = len(votes_data)

    columns = {
        "date": "date",
        "party": "str",
    }
    second_df: pd.DataFrame = create_fake_df(columns, n=total_votes, as_dict=False)
    df["party"] = second_df["party"]
    df["date"] = second_df["date"]
    test_case.existing_votes = df  # used for fitting tests
    return df


def mock_authors_data(test_case: TestCase, use_existing_data: bool) -> pd.DataFrame:
    if use_existing_data:
        return mock_authors_existing_data(test_case)
    else:
        return mock_new_authors_data(test_case)


def mock_authors_existing_data(test_case: TestCase):
    new_authors = getattr(test_case, "NEW_AUTHORS", DEFAULT_NEW_OBJECTS)
    existing_data_columns = ["project", "party"]
    selected_data = test_case.existing_authors[existing_data_columns]
    limited_data = selected_data.head(new_authors)
    return limited_data


def mock_new_authors_data(test_case: TestCase, total_results: int = None) -> pd.DataFrame:
    AUTHORS_PER_PROJECT = 4  # setear un TOTAL_RESULTS múltiplo de AUTHORS_PER_PROJECT
    authors_data = []
    total_project_len = total_results // AUTHORS_PER_PROJECT if total_results else len(test_case.project_ids)
    total_projects = test_case.project_ids[:total_project_len]
    print(f"Total projects: {total_project_len}")
    from tqdm import tqdm

    for project_id in tqdm(total_projects):
        for _ in range(AUTHORS_PER_PROJECT):
            party = Faker().name()
            authors_data.append([project_id, party])

    # Create a dataframe from the generated data
    df = pd.DataFrame(authors_data, columns=["project", "party"])
    test_case.existing_authors = df  # used for fitting tests
    return df


def mock_legislators_data(test_case: TestCase) -> pd.DataFrame:
    # TODO: de los legisladores sacamos nada más los nombres, podríamos intentar no usarlo pero hay que ver si funciona en la red
    n = len(test_case.person_ids)
    columns = {
        "person_full_name": "str",
    }
    df: pd.DataFrame = create_fake_df(columns, n=n, as_dict=False)
    df["person"] = test_case.person_ids
    return df


def mock_projects_data(test_case: TestCase, use_existing_data: bool) -> pd.DataFrame:
    return mock_new_projects_data(test_case)  # there should be no problem with new projects


def mock_projects_existing_data(test_case: TestCase):
    new_projects = getattr(test_case, "NEW_PROJECTS", DEFAULT_NEW_OBJECTS)
    existing_df_columns = ["project_title", "project_text", "project_year"]
    selected_data = test_case.existing_projects[existing_df_columns]
    limited_data = selected_data.head(new_projects)
    return limited_data


def mock_new_projects_data(test_case: TestCase, total_project_len: int = None) -> pd.DataFrame:
    total_project_len = total_project_len or len(test_case.project_ids)
    columns = {
        "project_title": "short_text",
        "project_text": "text",
        "project_year": "year",
    }
    df: pd.DataFrame = create_fake_df(columns, n=total_project_len, as_dict=False)
    df["project"] = test_case.project_ids
    test_case.existing_projects = df  # used for fitting tests
    return df


def create_project_ids(test_case: TestCase):
    columns = {"project": "int"}
    df: pd.DataFrame = create_fake_df(columns, n=test_case.MAX_TOTAL_PROJECTS, as_dict=False)
    test_case.project_ids = list(df["project"].unique())


def create_person_ids(test_case: TestCase):
    columns = {"person": "int"}
    df: pd.DataFrame = create_fake_df(columns, n=test_case.MAX_TOTAL_PERSONS, as_dict=False)
    test_case.person_ids = list(df["person"].unique())


def create_fake_endpoints_data(test_case: TestCase):
    return EndpointsDTO()


def paginate_data(df: pd.DataFrame):
    PAGE_SIZE = 1000
    total_rows = len(df)
    total_pages = total_rows // PAGE_SIZE
    if total_rows % PAGE_SIZE != 0:
        total_pages += 1
    for page in range(total_pages):
        start = page * PAGE_SIZE
        end = start + PAGE_SIZE
        yield df[start:end]


@contextmanager
def mock_method_paginated_data_retrieval(paginated_response=True):
    if paginated_response:
        with mock_method_side_effect(requests.Session, "get", side_effect=_mock_method_paginated_data_retrieval):
            yield
    else:
        with mock_method_side_effect(requests.Session, "get", side_effect=_mock_method_non_paginated_data_retrieval):
            yield


def _get_current_page(url):
    if "page=" not in url:
        return 1
    page_param = url.split("page=")[1]
    current_page = page_param.split("&")[0]
    return int(current_page)


def _calculate_next_page(current_page):
    # If the current page is the last one, we return None
    total_pages = settings.TOTAL_RESULTS // settings.PAGE_SIZE
    total_pages = total_pages if settings.TOTAL_RESULTS % settings.PAGE_SIZE == 0 else total_pages + 1
    if current_page == total_pages:
        return None
    # Otherwise, we return the next page
    return current_page + 1


def _get_next_page_url(url, next_page):
    if next_page is None:
        return None
    # the page in the url is page={page_number}&
    # we need to replace the current page with the next one
    if "page=" not in url:  # first page url
        next_page_url = url[:-1] if url.endswith("/") else url  # remove the last /
        next_page_url += f"?page=2&page_size={settings.DEFAULT_PAGE_SIZE}"
    else:
        next_page_url = url.replace(f"page={next_page - 1}", f"page={next_page}")
    return next_page_url


def _get_request_results(url, paginated_response):
    data_len = settings.PAGE_SIZE if paginated_response else settings.TOTAL_RESULTS
    if "votes" in url:
        data = mock_new_votes_data(settings, data_len)
    elif "authors" in url:
        data = mock_new_authors_data(settings, data_len)
    elif "projects" in url:
        data = mock_new_projects_data(settings, data_len)
    return data


def _mock_method_paginated_data_retrieval(url, params=None):
    current_page = _get_current_page(url)
    next_page = _calculate_next_page(current_page)
    next_page_url = _get_next_page_url(url, next_page)
    results_df = _get_request_results(url, paginated_response=True)
    results_json = results_df.to_dict(orient="records")

    data = {
        "count": settings.TOTAL_RESULTS,
        "next": next_page_url,
        "previous": None,
        "results": results_json,
    }
    return MockResponse(status_code=200, data=data, headers={})


def _mock_method_non_paginated_data_retrieval(url, data=None):
    results_df = _get_request_results(url, paginated_response=False)
    results_json = results_df.to_dict(orient="records")
    return MockResponse(status_code=200, data=results_json, headers={})
