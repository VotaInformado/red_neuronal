import os
import random
import pandas as pd
from django.db import models
from django.test import TestCase
from django.conf import settings

# Project
from red_neuronal.components.encoder import Encoder
from red_neuronal.components.neural_network import NeuralNetwork
from red_neuronal.tests.utils.faker import create_fake_df
from red_neuronal.utils.exceptions.exceptions import TransformingUnseenData, UntrainedNeuralNetwork


class VoteChoices(models.TextChoices):
    # Ongoing status
    ABSENT = "ABSENT", "Ausente"
    ABSTENTION = "ABSTENTION", "Abstención"
    NEGATIVE = "NEGATIVE", "Negativo"
    POSITIVE = "POSITIVE", "Afirmativo"
    PRESIDENT = ("PRESIDENT", "Presidente")


class EncoderTestCase(TestCase):
    def setUp(self):
        self.vote_choices = VoteChoices.values
        self.remove_persistence_files()

    def remove_persistence_files(self):
        files = os.listdir(settings.ENCODING_SAVING_DIR)
        # Iterate through the files and remove them
        for file in files:
            file_path = os.path.join(settings.ENCODING_SAVING_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def get_random_vote(self):
        return random.choice(self.vote_choices)

    def test_basic_encoding(self):
        votes_data = [self.get_random_vote() for _ in range(20)]
        fit_data = [{"voto": vote} for vote in votes_data]
        fit_df = pd.DataFrame(fit_data)
        transform_df = pd.DataFrame(fit_df["voto"].to_frame())
        encoder = Encoder(is_training=True)
        encoder.fit(fit_df)
        transformed_votes = encoder.transform(transform_df)
        self.assertEqual(len(transformed_votes), len(fit_data))

    def test_load_encoder_from_file(self):
        votes_data = [self.get_random_vote() for _ in range(20)]
        fit_data = [{"voto": vote} for vote in votes_data]
        fit_df = pd.DataFrame(fit_data)
        transform_df = pd.DataFrame(fit_df["voto"].to_frame())
        encoder = Encoder(is_training=True)
        encoder.fit(fit_df)
        expected_transformed_votes = encoder.transform(transform_df)
        encoder = Encoder(is_training=False)
        transformed_votes = encoder.transform(transform_df)
        self.assertEqual(transformed_votes.tolist(), expected_transformed_votes.tolist())

    def test_transforming_unseen_data_raises_an_error(self):
        votes_data = [self.get_random_vote() for _ in range(20)]
        fit_data = [{"voto": vote} for vote in votes_data]
        fit_df = pd.DataFrame(fit_data)
        transform_data = fit_data.copy()
        transform_data.append({"voto": "UNSEEN_DATA"})
        transform_df = pd.DataFrame(transform_data)
        encoder = Encoder(is_training=True)
        encoder.fit(fit_df)
        encoder = Encoder(is_training=False)
        with self.assertRaises(TransformingUnseenData) as context:
            transformed_votes = encoder.transform(transform_df)
