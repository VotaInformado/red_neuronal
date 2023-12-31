import tensorflow_hub as hub
import tensorflow_text
import numpy as np


class UniversalEmbedding:
    """
    Creates an instance of a Universal Encoder.
    """

    model_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/multilingual/versions/2"

    def __init__(self):
        self.model = hub.load(self.model_url)

    def encode_multiple(self, sentences):
        encodings = self.model(sentences)
        return encodings

    def encode_single(self, sentence):
        encodings = self.model(sentence)
        return encodings

    def create_law_text_embedding(self, texto):
        oraciones = str(texto).split(".")
        embeddings = self.encode_multiple(oraciones)
        reduced = np.mean(embeddings, axis=0)
        return reduced
