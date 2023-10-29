# rest_framework
from rest_framework.exceptions import APIException


class CustomException(APIException):
    def __init__(self, code: str, status_code: int, description: str):
        self.status_code = status_code  # needed for DRF to work
        self.detail = {
            "code": code,
            "status_code": status_code,
            "description": description,
        }

    def __str__(self):
        info = "\n"
        for key, value in self.detail.items():
            if key != "extra_details":
                info += f"{key}: {value}\n"
            else:
                if value:
                    info += f"{key}:\n"
                    for key, value in value.items():
                        info += f"\t{key}: {value}\n"
        return info


class UntrainedNeuralNetwork(CustomException):
    def __init__(self):
        code = "UNTRAINED_NEURAL_NETWORK"
        status_code = 400
        description = "The neural network has not been trained yet."
        super().__init__(code, status_code, description)


class EncodingException(CustomException):
    class Meta:
        abstract = True


class EncoderDataNotFound(EncodingException):
    def __init__(self):
        code = "ENCODER_DATA_NOT_FOUND"
        status_code = 400
        description = "The encoder data was not found. You must train the neural network before fitting it."
        super().__init__(code, status_code, description)


class TransformingUnseenData(EncodingException):
    def __init__(self, extra_values: list = []):
        code = "TRANSFORMING_UNSEEN_DATA"
        status_code = 400
        extra_values_str = ", ".join([str(value) for value in extra_values])
        description = (
            f"The encoder has found values that were not present in the fitted data. Extra values: {extra_values_str}"
        )
        super().__init__(code, status_code, description)
