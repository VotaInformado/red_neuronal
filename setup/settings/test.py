from .base import *

# Neural network persistence


FILES_COMMON_DIR = "red_neuronal/tests/persistence"
MODEL_SAVING_DIR = f"{FILES_COMMON_DIR}/model"
ENCODING_SAVING_DIR = f"{FILES_COMMON_DIR}/encoder"

# We override the endpoints which won't be used because of the mocks, but still with some requirements

VOTES_DATA_ENDPOINT = "votes"
LEGISLATORS_DATA_ENDPOINT = "legislators"
PROJECTS_DATA_ENDPOINT = "projects"
AUTHORS_DATA_ENDPOINT = "authors"
