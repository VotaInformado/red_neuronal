import os
import shutil
from django.test import TestCase
from django.conf import settings


class CustomTestCase(TestCase):
    def setUp(self):
        self.remove_persistence_files()

    def tearDown(self) -> None:
        self.remove_persistence_files()

    def remove_persistence_files(self):
        if os.path.exists(settings.FILES_COMMON_DIR):
            shutil.rmtree(settings.FILES_COMMON_DIR)
