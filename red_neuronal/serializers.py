from rest_framework import serializers
from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO


# Expected columns: project, person, party
class IndividualLegislatorSerializer(serializers.Serializer):
    person = serializers.IntegerField()


class ProjectSerializer(serializers.Serializer):
    project_id = serializers.CharField()
    project_title = serializers.CharField()
    project_text = serializers.CharField()
    project_year = serializers.IntegerField()


class AuthorSerializer(serializers.Serializer):
    party = serializers.CharField()


class LegislatorPredictionRequest(serializers.Serializer):
    legislator = IndividualLegislatorSerializer()
    project = ProjectSerializer()
    authors = serializers.ListField(child=AuthorSerializer())


class PredictionResponse(serializers.Serializer):
    legislator = serializers.IntegerField()
    vote = serializers.CharField()


class ProjectPredictionRequest(serializers.Serializer):
    legislators = serializers.ListField(child=IndividualLegislatorSerializer())
    project = ProjectSerializer()
    authors = serializers.ListField(child=AuthorSerializer())
