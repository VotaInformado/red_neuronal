# External libraries
from datetime import datetime
import pandas as pd
import random
import string
import time

# Unittest mock
from unittest.mock import patch

# Env variables
from django.conf import settings


BASE_URL = "http://localhost:8000"


class FakeClass:
    def fake_method(self):
        pass


class FakeResponse:
    def __init__(self, status_code, content):
        self.status_code = status_code
        self.content = content


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
