import os
import joblib
import pandas as pd
from django.conf import settings

# Encoders
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

# Project
from red_neuronal.utils.exceptions.exceptions import EncoderDataNotFound, TransformingUnseenData


class Encoder:
    class Meta:
        abstract = True

    def save_encoder(self):
        os.makedirs(os.path.dirname(self.ENCODING_FILE_DIR), exist_ok=True)
        joblib.dump(self.encoder, self.ENCODING_FILE_DIR)

    def __init__(self, is_training: bool):
        self.enconder: OneHotEncoder = None
        self.load_encoder(is_training)
        self.is_training = is_training

    def _load_encoder_for_training(self):
        """For training, we create the encoder from scratch"""
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def _load_encoder_for_fit(self):
        """For fitting, we load the encoder from disk. It should already have been saved"""
        try:
            self.encoder = joblib.load(self.ENCODING_FILE_DIR)
        except FileNotFoundError:
            raise EncoderDataNotFound()

    def load_encoder(self, is_training: bool):
        if is_training:
            self._load_encoder_for_training()
        else:
            self._load_encoder_for_fit()

    def fit(self, data: pd.DataFrame):
        self.encoder = self.encoder.fit(data)
        if self.is_training:
            # Creo que no tiene mucho sentido guardar el encoder en fits
            self.save_encoder()

    def _assert_no_new_data(self, data: pd.DataFrame):
        data_values = set(list(data[data.columns[0]]))
        loaded_categories = set(self.get_categories())
        if not data_values.issubset(loaded_categories):
            extra_values = data_values.difference(loaded_categories)
            if all([pd.isna(value) for value in extra_values]):
                return
            raise TransformingUnseenData(extra_values)

    def transform(self, data: pd.DataFrame):
        self._assert_no_new_data(data)
        return self.encoder.transform(data)

    def get_feature_names(self):
        return self.encoder.get_feature_names_out()

    def get_categories(self):
        return self.encoder.categories_[0]


class VotesEncoder(Encoder):
    def __init__(self, is_training: bool):
        ENCODING_FILE_NAME = "vote_encoder.pkl"
        self.ENCODING_FILE_DIR = f"{settings.ENCODING_SAVING_DIR}/{ENCODING_FILE_NAME}"
        super().__init__(is_training)


class LegislatorsEncoder(Encoder):
    def __init__(self, is_training: bool):
        ENCODING_FILE_NAME = "legislators_encoder.pkl"
        self.ENCODING_FILE_DIR = f"{settings.ENCODING_SAVING_DIR}/{ENCODING_FILE_NAME}"
        super().__init__(is_training)


class PartiesEncoder(Encoder):
    def __init__(self, is_training: bool):
        ENCODING_FILE_NAME = "authors_encoder.pkl"
        self.ENCODING_FILE_DIR = f"{settings.ENCODING_SAVING_DIR}/{ENCODING_FILE_NAME}"
        super().__init__(is_training)

    def _load_encoder_for_training(self):
        """For training, we create the encoder from scratch"""
        self.encoder = DictVectorizer(sparse=False, separator="_")

    def _assert_no_new_data(self, data: list):
        KEY_WORD = "party_authors"
        values_list = [v[0] for d in data for k, v in d.items()]
        data_values = set([f"{KEY_WORD}_{value}" for value in values_list])
        loaded_features = set(self.get_feature_names())
        if not data_values.issubset(loaded_features):
            extra_values = data_values.difference(loaded_features)
            if all([pd.isna(value) for value in extra_values]):
                # If all the extra values are NaN, we can ignore them
                return
            raise TransformingUnseenData()
