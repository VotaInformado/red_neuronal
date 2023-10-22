import pandas as pd

# Django rest framework
from rest_framework.response import Response
from rest_framework import status, viewsets
from rest_framework.decorators import action
from drf_yasg.utils import swagger_auto_schema

# Project
from red_neuronal.serializers import NetworkTrainingSerializer
from red_neuronal.utils.dtos.endpoints_dto import EndpointsDTO
from red_neuronal.components.data_handler import DataHandler
from red_neuronal.components.neural_network import NeuralNetwork


class NeuralNetworkView(viewsets.ViewSet):
    @swagger_auto_schema(
        responses={status.HTTP_204_NO_CONTENT: None},
    )
    @action(detail=False, methods=["post"])
    def train(self, request):
        serializer = NetworkTrainingSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        endpoints_info: EndpointsDTO = serializer.save()
        self.train_neural_network(endpoints_info)
        return Response(status=status.HTTP_204_NO_CONTENT)

    async def train_neural_network(self, endpoints_info: EndpointsDTO):
        df: pd.DataFrame = DataHandler.get_data(endpoints_info)
        neural_network = NeuralNetwork()
        neural_network.train(df)
