from rest_framework import serializers
from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO


class NetworkTrainingSerializer(serializers.Serializer):
    votes_endpoint = serializers.CharField()
    legislators_endpoint = serializers.CharField()
    projects_endpoint = serializers.CharField()
    authors_endpoint = serializers.CharField()

    def create(self, validated_data):
        dto = EndpointsDTO(validated_data)
        return dto
