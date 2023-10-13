import os
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from django.conf import settings

from red_neuronal.utils.exceptions.exceptions import EncoderDataNotFound, TransformingUnseenData


class Encoder:
    class Meta:
        abstract = True

    ENCODING_FILE_NAME = None

    # Assuming self.encoded_votes is your OneHotEncoder object
    def save_encoder(self):
        os.makedirs(os.path.dirname(self.ENCODING_FILE_DIR), exist_ok=True)
        joblib.dump(self.encoder, self.ENCODING_FILE_DIR)

    def __init__(self, is_training: bool):
        self.enconder: OneHotEncoder = None
        self.load_encoder(is_training)
        self.is_training = is_training

    def _load_encoder_for_training(self):
        """For training, we create the encoder from scratch"""
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

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

    def transform(self, data: pd.DataFrame):
        # Assumes the encoder has already been loaded
        data_values = set(list(data[data.columns[0]]))
        loaded_categories = set(self.get_categories())
        if not data_values.issubset(loaded_categories):
            import pdb

            pdb.set_trace()
            raise TransformingUnseenData()
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
