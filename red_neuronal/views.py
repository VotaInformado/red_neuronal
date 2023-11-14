# Django rest framework
import pandas as pd
from rest_framework.response import Response
from rest_framework import viewsets, mixins
from rest_framework.decorators import action
from drf_yasg.utils import swagger_auto_schema

# Project
from red_neuronal.serializers import (
    LegislatorPredictionRequest,
    LegislatorPredictionResponse,
    ProjectPredictionRequest,
    ProjectPredictionResponse,
)
from red_neuronal.components.neural_network.predictor import Predictor
from red_neuronal.components.data_handler import PredictionDataHandler


class SenateViewSet(viewsets.GenericViewSet):
    @swagger_auto_schema(
        method="post",
        serializer=LegislatorPredictionRequest,
        responses={200: LegislatorPredictionResponse},
        operation_description="Returns the predicted votes for a given legislator and project",
    )
    @action(detail=False, methods=["post"], url_path="predict-legislator-vote")
    def predict_legislator_vote(self, request, *args, **kwargs):
        """
        Returns the active senators
        """
        serializer = LegislatorPredictionRequest(request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        df: pd.DataFrame = PredictionDataHandler.get_prediction_df(data)
        predictor = Predictor()
        prediction = predictor.predict(df)
        prediction_data = LegislatorPredictionResponse(prediction)
        return Response(prediction_data)

    @swagger_auto_schema(
        method="post",
        serializer=ProjectPredictionRequest,
        responses={200: ProjectPredictionResponse},
        operation_description="Returns the predicted votes for a given legislator and project",
    )
    @action(detail=False, methods=["post"], url_path="predict-project-votes")
    def predict_project_votes(self, request, *args, **kwargs):
        """
        Returns the active senators
        """
        serializer = ProjectPredictionRequest(request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        df: pd.DataFrame = PredictionDataHandler.get_prediction_df(data)
        predictor = Predictor()
        prediction = predictor.predict(df)
        prediction_data = ProjectPredictionResponse(prediction)
        return Response(prediction_data)