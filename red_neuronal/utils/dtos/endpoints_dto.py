from typing import Dict
from django.conf import settings


class EndpointsDTO:
    def __init__(self):
        self.votes_endpoint = settings.VOTES_ENDPOINT  # vote, voter_id, project_id
        self.legislators_endpoint = settings.LEGISLATORS_ENDPOINT  # legislator, party
        self.projects_endpoint = settings.PROJECTS_ENDPOINT  # title, text, year
        self.authors_endpoint = settings.AUTHORS_ENDPOINT  # author
