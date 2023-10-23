# External libraries
from datetime import datetime
import pandas as pd
import random
import string
import time
from django.conf import settings
from contextlib import contextmanager
from django.test import TestCase
from faker import Faker

# Unittest mock
from unittest.mock import patch

# Env variables
from django.conf import settings

# Project
from red_neuronal.tests.test_helpers.faker import create_fake_df
from red_neuronal.utils.enums import VoteChoices
from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO
from red_neuronal.components.data_handler import DataHandler

BASE_URL = "http://localhost:8000"


class FakeClass:
    def fake_method(test_case: TestCase):
        pass


class FakeResponse:
    def __init__(test_case, status_code, content):
        test_case.status_code = status_code
        test_case.content = content


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
def mock_recoleccion_data(test_case: TestCase):
    votes_data = mock_votes_data(test_case)
    authors_data = mock_authors_data(test_case)
    legislators_data = mock_legislators_data(test_case)
    projects_data = mock_projects_data(test_case)
    with mock_method(DataHandler, "_get_votes", return_value=votes_data):
        with mock_method(DataHandler, "_get_authors", return_value=authors_data):
            with mock_method(DataHandler, "_get_legislators", return_value=legislators_data):
                with mock_method(DataHandler, "_get_law_projects", return_value=projects_data):
                    yield


def mock_votes_data(test_case: TestCase) -> pd.DataFrame:
    MAX_VOTES = 5

    # Generate random votes for each project and person combination
    votes_data = []
    test_case.total_votes = 0
    for project_id in test_case.project_ids:
        for _ in range(random.randint(1, MAX_VOTES)):
            person_id = random.choice(test_case.person_ids)
            vote = random.choice(VoteChoices.values)
            votes_data.append([project_id, person_id, vote])
            test_case.total_votes += 1

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


def mock_authors_data(test_case: TestCase) -> pd.DataFrame:
    MAX_AUTHORS = 3
    authors_data = []
    for project_id in test_case.project_ids:
        for _ in range(random.randint(1, MAX_AUTHORS)):
            party = Faker().name()
            authors_data.append([project_id, party])

    # Create a dataframe from the generated data
    df = pd.DataFrame(authors_data, columns=["project_id", "party"])
    return df


def mock_legislators_data(test_case: TestCase) -> pd.DataFrame:
    # TODO: de los legisladores sacamos nada más los nombres, podríamos intentar no usarlo pero hay que ver si funciona en la red
    n = len(test_case.person_ids)
    columns = {
        "person_full_name": "str",
    }
    df: pd.DataFrame = create_fake_df(columns, n=n, as_dict=False)
    df["person_id"] = test_case.person_ids
    return df


def mock_projects_data(test_case: TestCase) -> pd.DataFrame:
    total_projects = len(test_case.project_ids)
    columns = {
        "project_title": "short_text",
        "project_text": "text",
        "project_year": "year",
    }
    df: pd.DataFrame = create_fake_df(columns, n=total_projects, as_dict=False)
    df["project_id"] = test_case.project_ids
    return df


def create_project_ids(test_case: TestCase):
    columns = {"project_id": "int"}
    df: pd.DataFrame = create_fake_df(columns, n=test_case.MAX_TOTAL_PROJECTS, as_dict=False)
    test_case.project_ids = list(df["project_id"].unique())


def create_person_ids(test_case: TestCase):
    columns = {"person_id": "int"}
    df: pd.DataFrame = create_fake_df(columns, n=test_case.MAX_TOTAL_PERSONS, as_dict=False)
    test_case.person_ids = list(df["person_id"].unique())


def create_fake_endpoints_data(test_case: TestCase):
    return EndpointsDTO()
