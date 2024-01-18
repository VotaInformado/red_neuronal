import os
import shutil
import random
import pandas as pd
from django.db import models
from red_neuronal.tests.test_helpers.test_case import CustomTestCase
from django.conf import settings

# Project
from red_neuronal.utils.enums import VoteChoices
from red_neuronal.components.encoder import VotesEncoder
from red_neuronal.components.neural_network.neural_network import NeuralNetwork
from red_neuronal.tests.test_helpers.faker import create_fake_df
from red_neuronal.utils.exceptions.exceptions import TransformingUnseenData, UntrainedNeuralNetwork


class EncoderTestCase(CustomTestCase):
    def get_random_vote(self):
        return random.choice(VoteChoices.values)

    def test_basic_encoding(self):
        votes_data = [self.get_random_vote() for _ in range(20)]
        fit_data = [{"voto": vote} for vote in votes_data]
        fit_df = pd.DataFrame(fit_data)
        transform_df = pd.DataFrame(fit_df["voto"].to_frame())
        encoder = VotesEncoder(is_training=True)
        encoder.fit(fit_df)
        transformed_votes = encoder.transform(transform_df)
        self.assertEqual(len(transformed_votes), len(fit_data))

    def test_load_encoder_from_file(self):
        votes_data = [self.get_random_vote() for _ in range(20)]
        fit_data = [{"voto": vote} for vote in votes_data]
        fit_df = pd.DataFrame(fit_data)
        transform_df = pd.DataFrame(fit_df["voto"].to_frame())
        encoder = VotesEncoder(is_training=True)
        encoder.fit(fit_df)
        expected_transformed_votes = encoder.transform(transform_df)
        encoder = VotesEncoder(is_training=False)
        transformed_votes = encoder.transform(transform_df)
        self.assertEqual(transformed_votes.tolist(), expected_transformed_votes.tolist())

    def test_transforming_unseen_data_raises_an_error(self):
        votes_data = [self.get_random_vote() for _ in range(20)]
        fit_data = [{"voto": vote} for vote in votes_data]
        fit_df = pd.DataFrame(fit_data)
        transform_data = fit_data.copy()
        transform_data.append({"voto": "UNSEEN_DATA"})
        transform_df = pd.DataFrame(transform_data)
        encoder = VotesEncoder(is_training=True)
        encoder.fit(fit_df)
        encoder = VotesEncoder(is_training=False)
        with self.assertRaises(TransformingUnseenData) as context:
            transformed_votes = encoder.transform(transform_df)

    def test_transforming_unseen_data_does_not_raise_an_error_for_null_values(self):
        votes_data = [self.get_random_vote() for _ in range(20)]
        fit_data = [{"voto": vote} for vote in votes_data]
        fit_df = pd.DataFrame(fit_data)
        transform_data = fit_data.copy()
        transform_data.append({"voto": None})
        transform_df = pd.DataFrame(transform_data)
        encoder = VotesEncoder(is_training=True)
        encoder.fit(fit_df)
        encoder = VotesEncoder(is_training=False)
        transformed_votes = encoder.transform(transform_df)
