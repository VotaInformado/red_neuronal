from typing import Dict


class EndpointsDTO:
    def __init__(self, endpoints: Dict[str, str]):
        self.votes_endpoint = endpoints["votes"]
        self.legislators_endpoint = endpoints["legislators"]  # legislator, party
        self.projects_endpoint = endpoints["projects"]  # title, text, year
        self.authors_endpoint = endpoints["authors"]  # author
