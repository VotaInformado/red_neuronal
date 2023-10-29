import os
import warnings
import tensorflow as tf
import shutil
from django.test import TestCase
from rest_framework.test import APITestCase
from django.conf import settings


class CustomTestCase(TestCase):
    def setUp(self):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        super().setUp()
        self.remove_persistence_files()
        self.remove_data_handler_files()

    def tearDown(self) -> None:
        self.remove_persistence_files()
        self.remove_data_handler_files()

    def remove_persistence_files(self):
        if os.path.exists(settings.FILES_COMMON_DIR):
            shutil.rmtree(settings.FILES_COMMON_DIR)

    def remove_data_handler_files(self):
        if os.path.exists(settings.DATA_HANDLER_FILES_DIR):
            shutil.rmtree(settings.DATA_HANDLER_FILES_DIR)


class CustomAPITestCase(APITestCase):
    def setUp(self):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.remove_persistence_files()

    def tearDown(self) -> None:
        self.remove_persistence_files()

    def remove_persistence_files(self):
        if os.path.exists(settings.FILES_COMMON_DIR):
            shutil.rmtree(settings.FILES_COMMON_DIR)
