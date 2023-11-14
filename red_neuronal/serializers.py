from rest_framework import serializers
from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO


# Expected columns: project, person, party
class IndividualLegislatorSerializer(serializers.Serializer):
    # project = serializers.CharField()
    person = serializers.CharField()
    party = serializers.CharField()


class ProjectSerializer(serializers.Serializer):
    project = serializers.CharField()
    project_title = serializers.CharField()
    project_text = serializers.CharField()
    project_year = serializers.IntegerField()


class AuthorSerializer(serializers.Serializer):
    party = serializers.CharField()


class LegislatorPredictionRequest(serializers.Serializer):
    legislator = IndividualLegislatorSerializer()
    project = ProjectSerializer()
    authors = serializers.ListField(child=AuthorSerializer())


class LegislatorPredictionResponse(serializers.Serializer):
    predicted_vote = serializers.CharField()


class ProjectPredictionRequest(serializers.Serializer):
    legislators = serializers.ListField(child=IndividualLegislatorSerializer())
    project = ProjectSerializer()


class ProjectPredictionResponse(serializers.Serializer):
    predicted_votes = serializers.ListField(child=LegislatorPredictionResponse())
