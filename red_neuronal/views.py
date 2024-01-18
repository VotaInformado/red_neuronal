# Django rest framework
import pandas as pd
from rest_framework.response import Response
from rest_framework import viewsets
from rest_framework.decorators import action
from drf_yasg.utils import swagger_auto_schema

# Project
from red_neuronal.serializers import (
    LegislatorPredictionRequest,
    PredictionResponse,
    ProjectPredictionRequest,
)
from red_neuronal.components.neural_network.predictor import Predictor
from red_neuronal.components.data_handler import PredictionDataHandler


class PredictionViewSet(viewsets.GenericViewSet):
    @swagger_auto_schema(
        method="post",
        serializer=LegislatorPredictionRequest,
        responses={200: PredictionResponse},
        operation_description="Returns the predicted votes for a given legislator and project",
    )
    @action(detail=False, methods=["post"], url_path="predict-legislator-vote")
    def predict_legislator_vote(self, request, *args, **kwargs):
        """
        Returns the active senators
        """
        serializer = LegislatorPredictionRequest(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        df: pd.DataFrame = PredictionDataHandler.get_prediction_df(data)
        predictor = Predictor()
        prediction = predictor.predict(df)[0]
        prediction_data = PredictionResponse(prediction).data
        return Response(prediction_data)

    @swagger_auto_schema(
        method="post",
        serializer=ProjectPredictionRequest,
        responses={200: PredictionResponse},
        operation_description="Returns the predicted votes for a given legislator and project",
    )
    @action(detail=False, methods=["post"], url_path="predict-project-votes")
    def predict_project_votes(self, request, *args, **kwargs):
        """
        Returns the active senators
        """
        serializer = ProjectPredictionRequest(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        # import pdb; pdb.set_trace()
        df: pd.DataFrame = PredictionDataHandler.get_prediction_df(data)
        predictor = Predictor()
        prediction = predictor.predict(df)
        prediction_data = PredictionResponse(prediction, many=True).data
        return Response(prediction_data)
